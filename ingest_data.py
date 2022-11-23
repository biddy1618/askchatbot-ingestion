from sentence_transformers import SentenceTransformer
import spacy 
import os 
from tqdm import tqdm 
import json 
import re 
from datasets import Dataset
import pandas as pd
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from torch import bfloat16
import json 
from ask_extension import get_ask_extension_data
from markdownify import markdownify
from markdown2 import markdown as to_html

nlp = spacy.load('en_core_web_sm')

# Loading fine-tuned model
embed_url = "JeffEduworks/generalized_chatbot_model"
auth_token = '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'
model = embed = SentenceTransformer(
    model_name_or_path  = embed_url         ,
    use_auth_token      = auth_token        ,
    device              = 'cuda'             ).to(bfloat16)
tokenizer = model.tokenizer

MAX_LENGTH = model.max_seq_length
VECTOR_SIZE = model[1].word_embedding_dimension

# Elastic Search settings 
SETTINGS =  {"number_of_shards": 2, "number_of_replicas": 1}
MAPPINGS =  {
        "dynamic"   : "false",
        "properties": {
            "source"        : {"type": "keyword", "index": "false" , "ignore_above": 32766},
            "url"           : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "title"         : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "text"          : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "subHead"       : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "thumbnail"     : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "state"         : {"type": "keyword", "index": "false", "ignore_above": 32766},
            "vector"        : {"type": "dense_vector", "dims": VECTOR_SIZE}
        }
    }

# Creating a folder to store the data. Will also be ingested into ES
DATA_PATH = './chunked_data'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

def get_token_length(txt: str) -> int:
    """Returns the number of tokens in a string"""
    return tokenizer(txt, return_tensors='pt')['input_ids'].shape[1]

def process_long_text(txt: str, chunk_length: int = 3) -> list:
    """Splits longer texts into chunks of 3 sentences. Returns individual sentences if this is not possible"""
    sentences = [x for sen in nlp(txt).sents if (x:=sen.text.strip())]
    if sentences and len(sentences) > 1:
        chunks = []
        combined_sentences = [sentences[0]]
        for i, sen in enumerate(sentences): 
            if i == 0:
                continue 
            
            possible_new_chunk = ' '.join(combined_sentences + [sen])
            if get_token_length(possible_new_chunk) < MAX_LENGTH:
                combined_sentences.append(sen)
            else:
                old_chunk = ' '.join(combined_sentences)
                chunks.append(old_chunk)
                combined_sentences = [sen]
        if combined_sentences:
            chunks.append(' '.join(combined_sentences))
        return chunks
    return sentences 
    
    
def clean_header(header: str) -> str:
    """Cleans header items found in some of the scraped datasets"""
    if len(header) > 1:
        header = header[0].upper() + header[1:]
    return re.sub('([A-Z][^A-Z]+)', r'\1 ', header).replace('- ', '-').strip()


not_relevant_headers = ['description', 'introduction', 'summary', 'question', 'answer', 'conclusion']
def prepend_heading_text(header: str) -> str:
    """Prepends the header if it contains useful information"""
    header = clean_header(header)
    lowered_header = header.lower()
    for txt in not_relevant_headers:
        if txt in lowered_header:
            return ''
    return f"{header}: "


def clean_text(text: str) -> str:
    """Takes in description and uses various regular expressions to clean it up somewhat."""
    text = text.replace("<p>", "").replace("</p>", "")
    text = re.sub(' +', ' ', text)
    text = text.replace(u'\xa0', u' ')
    
    text = re.sub('On [A-Z][a-z]*(?:| \d*)\, ([A-Z].+|\d{4}.+)', '', text).strip() # Weird ask error
    text = re.sub('\-+(?:| )(?:Forwarded|Original).+', '', text).strip() 
    text = re.sub('\. [A-Z][a-z]+$', '.', text)
    text = re.sub(' From: [A-Z].+', '', text)
    text = re.sub(' Sent from [A-Z].+', '', text)
    text = text.replace('My 4-H ________ Project Record (000-00R)', '')
    text = re.sub('((?:\n|\n | \n)[^\w<]*)', r'\n', text)
    text = re.sub(' ([.,!:;?])', r'\1', text)
    text = re.sub('([:;,!?])(\w)', r'\1 \2', text)
    text = text.replace('&amp;', '&')
    text = re.sub('[.]([A-Z])', r'. \1', text)
    text = re.sub('([:!.,?])\:', r'\1', text)
    text = re.sub("^(Hi|Hello)(?:,|\.)", '', text)
    text = re.sub('[.](com|org|pdf)([A-Z])', r'.\1\n\2', text)
    
    text = to_markdown(text) # Only converts if there are html tags present in text 
    return text.strip()

def to_markdown(text: str) -> str:
    """Using markdown rather than HTML for vectorization due to chunking/vectorization being more efficient"""
    text = text.strip()
    if '<ul>' in text or '<ol>' in text or "<a" in text or "<strong>" in text:
        if not re.search("^<(?:u|o)l>", text):
            text = f"<p>{text}"
            text = re.sub("(<(?:u|o)l>)", r"</p>\1", text)
        if  not re.search("</(?:u|o)l>$", text):
            text = re.sub("(<(?:u|o)l>)", r"\1<p>", text)
            text = f"{text}</p>"
        text = markdownify(text)
        text = text.strip()
        text = re.sub('\n\n+', '\n\n', text)
    return text 
    
    
def chunk_data(txt: str, used_markdown=False) -> list:
    """Checks if text is over the maximum model length. If so, splits into chunks based on paragraphs"""
    assert isinstance(txt, str)
    token_length = get_token_length(txt)
    if token_length >= MAX_LENGTH and '\n' in txt and not used_markdown:
        txts = txt.split('\n')
        chunked = [chunk_data(t, False) for t in txts]
        flattened = [t for chunk in chunked for t in chunk if t]
        return flattened
    elif token_length >= MAX_LENGTH and used_markdown:
        txts = txt.split("\n\n")
        chunked = [chunk_data(x, False) for t in txts if (x:=t.strip())]
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
    text = f"{header}{item['text']}"
    using_markdown = True if "<li>" in text or "<a" in text or "<strong>" in text else False
    text = clean_text(text)
    chunked_text = chunk_data(text, using_markdown)
    return chunked_text 

    
def get_text_dict_format(item, k):
    if item['answer'][k]['author'] == 'The Question Asker':
        return None
    else: 
        return clean_text(item['answer'][k]['response'])
    
def get_text_list_format(a):
    if 'attachments' in a or ('author' in a and a['author'] == 'The Question Asker'):
        return None 
    else:
        return clean_text(a['response'])
def remove_irrelevant_info(text):
    # Remove greeting at beginning of sentence
    text = re.sub("^w+[,.](\w)", r'\1', text )
    return text 
    

def get_ask_text(data: list) -> list:
    unique_text = set()
    for item in tqdm(data, desc="Adding Ask Extension Data"):
        url = get_link(item)
        thumbnail = item['attachments']['-1'] if 'attachments' in item else ''
        title = re.sub('\#.*', '', item['title']).strip()
        title = re.sub("<[^>]*>", '', title).strip()
        if 'question' not in item:
            continue
        subhead = clean_text(item['question'])
        state = item['state']
        is_dict= True if isinstance(item['answer'], dict) else False
        for a in item['answer']:
            text = get_text_dict_format(item, a) if is_dict else get_text_list_format(a)
            if text:
                text = re.sub('([a-z,”:.])\n([a-z“–])', r'\1 \2', text) # Weird line break in some answers
                text = re.sub('\.([A-Z])', r'.\n\1', text)
                chunked_items = chunk_data(text, used_markdown=False)
                for chunk in chunked_items:
                    chunk = re.sub('\s+', ' ', chunk)
                    if chunk not in unique_text:
                        unique_text.add(chunk)
                        yield {'source': "ask_extension_kb", "state": state, 'title': title, 'url': url, 'text': chunk, 'thumbnail': thumbnail, "subHead": subhead}
                        
                
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
        return 'clemson', "South Carolina"
    elif 'okstate.edu' in url:
        return 'oklahoma_state', "Oklahoma"
    elif 'oregonstate.edu' in url:
        return 'oregon_state', "Oregon"
    elif 'ipm.' in url or 'youtu' in url:
        return 'uc_ipm', "California"
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
    elif 'images' in item and item['images']:
        return item['images'][0]['src']
    else:
        return ''
    
def get_subheader(content_item):
    if 'subHead' in content_item:
        return content_item['subHead']
    elif 'header' in content_item and (x:= prepend_heading_text(content_item['header'])) and 'title' in content_item and content_item['title']:
        return f"{content_item['title']}: {x}"
    elif 'header' in content_item and (x:= prepend_heading_text(content_item['header'])):
        return x
    elif 'title' in content_item:
        return content_item['title']
    else:
        return ''

def get_final_format(item: dict, chunk: str, content) -> dict:
    url = item['url'] if 'url' in item else item['link']
    source, state = get_source(url)
    title = get_title(item)
    thumbnail = get_thumbnail(item)
    subhead = get_subheader(item)
    return {'source': source,  "state": state, 'title': title, 'url': url, 'text': chunk, 'thumbnail': thumbnail, "subHead": subhead}
        
        
def parse_qa_data(data: list) -> list:
    if data:
        relevant_text = [get_final_format(item, clean_text(v), item) for item in data for k,v in item.items() if k in ['question', 'answer']]
        return  [get_final_format(item, chunk, item) for item in tqdm(relevant_text, desc='parsing qa data') for chunk in chunk_data(item['text'], used_markdown=False) if chunk]
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
            
    es_hosts = ['https://dev.es.chat.ask.eduworks.com/', 'https://qa.es.chat.ask.eduworks.com/']
    #es_hosts = ['http://localhost:9200/', 'https://dev.es.chat.ask.eduworks.com/']
    for es_host in es_hosts:
        es = Elasticsearch([es_host], http_auth=('elastic', 'changeme'), timeout=140)
        if es.indices.exists(index):
            es.indices.delete(
                index   = index, 
                ignore  = 404)
            es.indices.refresh()
        es.indices.create(index=index, mappings=MAPPINGS, settings=SETTINGS)
        bulk(es, gen_data(), chunk_size=1000, request_timeout=120)

def prep_to_save(all_data: list) -> list:
    """Converts markdown text to html for display purposes"""
    for item in tqdm(all_data, desc='Converting markdown to html'):
        item['text'] = to_html(item['text'])
        yield item 

def save_data(path: str, all_data: list):
    """Saves as json and as a HuggingFace dataset for easy testing of the model"""
    with open(f"./{DATA_PATH}/{path}.json", 'w', encoding = 'utf-8') as w:
        json.dump(all_data, w, indent=4, ensure_ascii=False)
        
        
    # You can comment this out. It's just so I can easily view and test the model
    ds = Dataset.from_pandas(pd.DataFrame(all_data))  
    ds.save_to_disk(f'./{DATA_PATH}/{path}')
    
    all_data = get_vectors(all_data)    
    all_data = list(prep_to_save(all_data))
    ingest_into_es(all_data, path)
    
    print(f"fully ingested {path} into elasticsearch")
    
def parse_ask_extension_data():
    ask_extension_data = get_ask_extension_data()
    relevant_text = list(get_ask_text(ask_extension_data))
    return relevant_text

def get_vectors(all_data: list) -> list:
    """Vectorizing text in the dataset and then adding as a key in the list of dicts."""
    print(f"\nVectorizing {len(all_data)} items\n")
    vectors = model.encode([item['text'] for item in all_data], batch_size=40, show_progress_bar=True).tolist()
    return [{**item, "vector": vector} for item, vector in zip(all_data, vectors)]
    
def get_all_data():
    paths = [f'./data/{f}'for f in os.listdir('./data')]
    all_data = parse_ask_extension_data()
    for path in paths:
        if '.json' in path:
            data = load_json(path)
            formatted = extract_formatted_data(data)
            all_data.extend(formatted)
    all_data = [item for item in all_data if item['url'] and item['text']]
    save_data('chatbot_data', all_data)
    return list(set([item['url'] for item in all_data]))

def clean_and_save_ask():
    all_data = parse_ask_extension_data()
    with open('./chunked_data/ask_only.json', 'w', encoding='utf-8') as w:
        json.dump(all_data, w, indent=4, ensure_ascii=False)

def main():
    #all_links = get_all_data()
    clean_and_save_ask()
    
if __name__ == '__main__':
    main()