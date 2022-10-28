from sentence_transformers import SentenceTransformer
import spacy 
import os 
from tqdm import tqdm 
import requests 
import json 
import re 
from datasets import Dataset
import pandas as pd
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from torch import bfloat16
import time


# Initializing models
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('./model').to('cuda').to(bfloat16)
MAX_LENGTH = model.max_seq_length
VECTOR_SIZE = model[1].word_embedding_dimension
tokenizer = model.tokenizer

# Elastic Search settings 
#SETTINGS =  {"number_of_shards": 2, "number_of_replicas": 1}
MAPPINGS =  {
        "dynamic"   : "false",
        "properties": {
            "source"        : {"type": "keyword", "index": "false" , "ignore_above": 32766},
            "url"           : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "title"         : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "text"          : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "subHead"       : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "thumbnail"     : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "vector"        : {"type": "dense_vector", "dims": VECTOR_SIZE}
        }
    }

# Creating a folder to store the data. Will also be ingested into ES
DATA_PATH = './cleaned_new'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

def get_token_length(txt: str) -> int:
    """Returns the number of tokens in a string"""
    return tokenizer(txt, return_tensors='pt')['input_ids'].shape[1]

def process_long_text(txt: str, chunk_length: int = 3) -> list:
    """Splits longer texts into chunks of 3 sentences. Returns individual sentences if this is not possible"""
    sentences = [x for sen in nlp(txt).sents if (x:=sen.text.strip())]
    if len(sentences) >= chunk_length:
        return [' '.join(sentences[i:i+chunk_length])for i in range(0, len(sentences), chunk_length)]
    else:
        return sentences
    
    
def clean_header(header: str) -> str:
    """Cleans header items found in some of the scraped datasets"""
    return re.sub('([A-Z][^A-Z]+)', r'\1 ', header).replace('- ', '-').strip()


not_relevant_headers = ['description', 'introduction', 'summary', 'question', 'answer', 'conclusion']
def prepend_heading_text(header: str) -> str:
    """Prepends the header if it contains useful information"""
    header = clean_header(header)
    lowered_header = header.lower()
    for txt in not_relevant_headers:
        if txt in lowered_header:
            return ''
    return f"{header}\n"


def clean_text(text: str) -> str:
    """Takes in description and uses various regular expressions to clean it up somewhat."""
    text = re.sub(' +', ' ', text)
    text = text.replace(u'\xa0', u' ')
    text = re.sub('On [A-Z][a-z]*(?:| \d*)\, ([A-Z].+|\d{4}.+)', '', text).strip() # Weird ask error
    text = re.sub('\-+(?:| )(?:Forwarded|Original).+', '', text).strip() 
    text = re.sub('\. [A-Z][a-z]+$', '.', text)
    text = re.sub(' From: [A-Z].+', '', text)
    text = re.sub(' Sent from [A-Z].+', '', text)
    text = text.replace('My 4-H ________ Project Record (000-00R)', '')
    text = re.sub('((?:\n|\n | \n)[^\w]*)', r'\n', text)
    text = re.sub(' ([.,!?])', r'\1', text)
    return text.strip()

def chunk_data(txt: str) -> list:
    """Checks if text is over the maximum model length. If so, splits into chunks based on paragraphs"""
    assert isinstance(txt, str)
    token_length = get_token_length(txt)
    if token_length >= MAX_LENGTH and '\n' in txt:
        txts = txt.split('\n')
        chunked = [chunk_data(t) for t in txts]
        flattened = [t for chunk in chunked for t in chunk if t]
        return flattened
    elif token_length >= MAX_LENGTH:
        return process_long_text(txt)
    elif len(txt.split()) <= 12:
        return []
    else:
        return [txt]

def clean_dict_item(item: dict) -> str:
    """Assumes item has natural paragraph structure. 
        Breaks items into paragraphs, further splits into chunks if needed, and adds header if it's not generic"""
    header = prepend_heading_text(item['header'])
    texts = item['text'].split('\n')
    chunked_text = []
    for text in texts:
        text = clean_text(text)
        chunked_text.extend(chunk_data(text))
    if chunked_text:
        chunked_text[0] = f"{header}{chunked_text[0]}"
    return chunked_text 

def get_all_os_ticket_items(start_year: int = 2006, end_year: int = 2024):
    """Calls OS ticket API to get all ask extension data"""
    for i in tqdm(range(start_year, end_year), desc='Calling OS Ticket API'):
        start = str(i) 
        end = str(i+1)
        url = f'https://qa.osticket.eduworks.com/api/knowledge/{start}-01-01/{end}-01-01'
        try:
            r = requests.get(url, timeout=40)
            items = r.json()
        except requests.exceptions.Timeout: 
            print(f"{start}, {end}")
            continue

        if items:
            for item in items:
                yield item    
                
def get_ask_extension_data() -> list:
    """Attempts to load from json. Though, will call os ticket API if not available"""    
    try:
        with open('raw_ask_extension_data.json', 'r', encoding='utf-8') as r:
            return json.load(r)
    except:
        return list(get_all_os_ticket_items())       
    
    
def get_ask_text(data: list) -> list:
    unique_text = set()
    for item in tqdm(data, desc="Adding Ask Extension Data"):
        url = get_link(item)
        thumbnail = item['attachments'][0] if 'attachments' in item else ''
        subhead = re.sub('\#.+', '', item['title']).strip()
        for k in item['answer']:
            text = clean_text(item['answer'][k]['response'])
            text = re.sub('([a-z,”:.])\n([a-z“–])', r'\1 \2', text) # Weird line break in some answers
            chunked_items = chunk_data(text)
            for chunk in chunked_items:
                chunk = re.sub('\s+', ' ', chunk)
                if chunk not in unique_text:
                    unique_text.add(chunk)
                    
                    yield {'source': "ask_extension_kb", 'title': item['title'], 'url': url, 'text': chunk, 'thumbnail': thumbnail, "subHead": subhead}
                     
                
def get_link(item: dict) -> str:
    """Creates ask url from faq-id"""
    faq_id = item['faq-id']
    return f"https://ask2.extension.org/kb/faq.php?id={faq_id}"  


def load_json(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as r:
        return json.load(r)
    
def is_qa_format(item: dict) -> bool:
    """Checks what type of format the dataset is in"""
    if 'question' in item.keys() and 'answer' in item.keys():
        return True
    else:
        return False
def get_source(url: str) -> str:
    if 'clemson.edu' in url:
        return 'clemson'
    elif 'okstate.edu' in url:
        return 'oklahoma_state'
    elif 'oregonstate.edu' in url:
        return 'oregon_state'
    elif 'ipm.' in url or 'youtu' in url:
        return 'uc_ipm'
    else:
        print(url)
        raise Exception
    
def get_title(item: dict) -> str:
    """Finds title of item. If it doesn't exist, uses question"""
    if 'title' in item:
        return item['title']
    elif 'question' in item:
        return item['question']
    else:
        return '' 
        
def get_thumbnail(item: dict) -> str:
    if 'thumbnail' in item:
        return item['thumbnail']
    elif 'images' in item:
        return item['images'][0]['src']
    else:
        return ''
    
def get_subheader(content_item):
    if 'subHead' in content_item:
        return content_item['subHead']
    elif 'header' in content_item and (x:= prepend_heading_text(content_item['header'])):
        return x
    elif 'title' in content_item:
        return content_item['title']
    else:
        return ''
    
        
def get_final_format(item: dict, chunk: str, content) -> dict:
    url = item['url'] if 'url' in item else item['link']
    source = get_source(url)
    title = get_title(item)
    thumbnail = item['thumbnail'] if 'thumbnail' in item else ''
    subhead = get_subheader(item)
    return {'source': source, 'title': title, 'url': url, 'text': chunk, 'thumbnail': thumbnail, "subHead": subhead}
        
def parse_qa_data(data: list) -> list:
    if data:
        relevant_text = [get_final_format(item, clean_text(v), item) for item in data for k,v in item.items() if k in ['question', 'answer']]
        return  [get_final_format(item, chunk, item) for item in tqdm(relevant_text, desc='parsing qa data') for chunk in chunk_data(item['text']) if chunk]
    else:
        return []

def extract_qa_from_headers(data):
    content, qa = [], []
    for item in data:
        if 'ask-expert' in item['link']:
            qa.append(item)
        else:
            content.append(item)
    return content, qa 

def extract_formatted_data(data: list):
    if is_qa_format(data[0]):
        return parse_qa_data(data)
    else: 
        content_data, qa_data = extract_qa_from_headers(data)
        qa_data = parse_qa_data(qa_data)
        content_data = [get_final_format(item, chunk, content) for item in content_data for content in item['content'] for chunk in clean_dict_item(content) if chunk]
        return qa_data + content_data
    
    
def ingest_into_es(data: list, index: str):
    """Deleting any existing index and then ingesting the new data"""
    def gen_data():
        for item in tqdm(data, desc='Ingesting into Elasticsearch'):
            yield {'_index': index, '_type': '_doc', **item}
            
    es_hosts = ['http://localhost:9200']
    for es_host in es_hosts:
        es = Elasticsearch([es_host], http_auth=('elastic', 'changeme'), timeout=140)
        if es.indices.exists(index):
            es.indices.delete(
                index   = index, 
                ignore  = 404)
            es.indices.refresh()
        es.indices.create(index=index, mappings=MAPPINGS)
        bulk(es, gen_data(), chunk_size=1000, request_timeout=120)
    
def save_data(path: str, all_data: list):
    """Saves as json and as a HuggingFace dataset for easy testing of the model"""
    with open(f"./{DATA_PATH}/{path}.json", 'w', encoding = 'utf-8') as w:
        json.dump(all_data, w, indent=4, ensure_ascii=False)
        
    ds = Dataset.from_pandas(pd.DataFrame(all_data))
    ds.save_to_disk(f'./{DATA_PATH}/{path}')
    if 'test_' not in path:
        all_data = get_vectors(all_data)
        
    ingest_into_es(all_data, path)
    
    print(f"fully ingested {path} into elasticsearch")
    
def parse_ask_extension_data():
    ask_extension_data = get_ask_extension_data()
    relevant_text = list(get_ask_text(ask_extension_data))
    return relevant_text

def get_vectors(all_data: list) -> list:
    """Vectorizing text in the dataset and then adding as a key in the list of dicts."""
    print(f"\nVectorizing {len(all_data)} items\n")
    vectors = model.encode([item['text'] for item in all_data], batch_size=32, show_progress_bar=True).tolist()
    return [{**item, "vector": vector} for item, vector in zip(all_data, vectors)]
    
def get_all_data():
    paths = [f'./data/{f}'for f in os.listdir('./data')]
    all_data = parse_ask_extension_data()
    for path in paths:
        if '.json' in path:
            data = load_json(path)
            formatted = extract_formatted_data(data)
            all_data.extend(formatted)
    save_data('chatbot_data', all_data)
    return list(set([item['url'] for item in all_data]))
    
def parse_test_data(file: str, sheet_names: list, all_urls: list):
    """Retrieves questions and answer links from excel file. Stores in elastic search and saves to disk"""
    for sheet_name in sheet_names:
        df = pd.read_excel(file, sheet_name=sheet_name)
        test_questions = []
        for i, row in df.iterrows(): 
            
            original_url = row['resource'] if 'resource' in df.columns else row['URL']
            if isinstance(original_url, str):
                url = f"https://{original_url}" if "http" not in original_url else original_url
                question = row['question'] if 'question' in df.columns else row['Question']
                if url and question and url in all_urls:
                    test_questions.append({"question": question, "url": url, "row": i+2})
        name = f'test_data_{sheet_name.lower()}'
        save_data(path=name, all_data=test_questions)
        
def main():
    all_links = get_all_data()
    sheets = ['made_up_OK_OR', 'UC_IPM_chatbot']
    parse_test_data(file='./data/AE_test_QA_chatbot_v2.xlsx', sheet_names=sheets, all_urls=all_links)
    
if __name__ == '__main__':
    main()