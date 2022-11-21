'''
Module for embedding data to vectors and ingesting the result to ES instance.

Author: Dauren Baitursyn
Date: 01.11.22
'''
import sys
import logging
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from collections import deque
from spacy.lang.en import English
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from sentence_transformers import SentenceTransformer


ES_USERNAME = 'elastic'
ES_PASSWORD = 'changeme'
ES_INDEX = 'test'
ES_HOST = 'http://localhost:9200/'
# ES_HOST = 'https://dev.es.chat.ask.eduworks.com/'
# ES_HOST = 'https://qa.es.chat.ask.eduworks.com/'
EMBED_CACHE_URL = '/var/tmp/models'
# MODEL_URL = 'all-distilroberta-v1'
MODEL_URL = "JeffEduworks/generalized_chatbot_model"
AUTH_TOKEN = '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'

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

logger.info(f'Start loading model - {MODEL_URL}')
model = SentenceTransformer(
    model_name_or_path=MODEL_URL,
    use_auth_token=AUTH_TOKEN,
    cache_folder= EMBED_CACHE_URL,
    device='cuda'            
)
logger.info(f'Done loading model')
MAX_SEQ_SIZE = model.max_seq_length
VECTOR_SIZE = model[1].word_embedding_dimension
BATCH_SIZE = 64
MAX_STRING_SIZE = 32766
MAPPING  = {
    "settings": {"number_of_shards": 2, "number_of_replicas": 1},
    "mappings": {
        "dynamic"       : "false"   ,
        "date_detection": "false"   ,
        "_source"   : {"enabled": "true"},
        "properties": {
            "source"        : {"type": "keyword", "index": "true" , "ignore_above": MAX_STRING_SIZE},
            "url"           : {"type": "keyword", "index": "true" , "ignore_above": MAX_STRING_SIZE},

            "title"         : {"type": "keyword", "index": "false", "ignore_above": MAX_STRING_SIZE},
            "images"        : {"type": "keyword", "index": "false", "ignore_above": MAX_STRING_SIZE},
            "vectors"       : {
                "type"      : "nested",
                "properties": {
                    "vector": {
                        "type": "dense_vector", 
                        "dims": VECTOR_SIZE
                    },
                    "field" : {"type": "keyword", "index": "false", "ignore_above": MAX_STRING_SIZE},
                    "name"  : {"type": "keyword", "index": "false", "ignore_above": MAX_STRING_SIZE},
                    "im_src": {"type": "keyword", "index": "false", "ignore_above": MAX_STRING_SIZE},
                    "text"  : {"type": "keyword", "index": "false", "ignore_above": MAX_STRING_SIZE},
                }
            }
        }
    }
}

nlp = English()
nlp.add_pipe('sentencizer')


def get_chunks(texts, max_seq_length):
    '''
    Chunk the text into fragments no longer than maximum sequence length.
    '''
    texts_new = []

    for item in texts:
        text, field, name, im_src = item['text'], item['field'], item['name'], item['im_src']
        doc = nlp(text)
        sents = [sent for sent in doc.sents]
        
        if len(text) == 0:
            print(item)
        start, end = 0, 1
        while end != len(sents):
            if start == end:
                end +=1
            elif len(' '.join([sent.text for sent in sents[start:end+1]])) > max_seq_length:
                texts_new.append({
                    'text': ' '.join([sent.text for sent in sents[start:end]]),
                    'field': field,
                    'name': name,
                    'im_src': im_src
                })
                start += 1
            else:
                end += 1

        texts_new.append({
            'text': ' '.join([sent.text for sent in sents[start:end]]),
            'field': field,
            'name': name,
            'im_src': im_src
        })

    return texts_new


def embed_data(df):
    '''
    Embed the text into vectors given the transformed data.
    '''
    logger.info(f'STARTING TRANSFORMING...')
    df_texts = []

    for i, row in tqdm(df.iterrows()):
        texts = row['texts']
        df_texts.append(get_chunks(texts, MAX_SEQ_SIZE))
        if (i+1) % 500 == 0:
            logger.info(f'Finished transforming of {i+1} rows of dataframe')
        
        
    logger.info(f'Finished transforming of {i+1} rows of dataframe')
    logger.info(f'FINISHED TRANSFORMING')
    texts = [item['text'] for row in df_texts for item in row]
    logger.info(f'STARTING EMBEDDING - BATCH_SIZE = {BATCH_SIZE}')
    logger.info(f'Number of texts to be embedded = {len(texts)}')
    # Sentence Encoder model        
    vectors = model.encode(
        sentences           = texts     ,
        batch_size          = BATCH_SIZE,
        show_progress_bar   = True
    ).tolist()

    index = 0
    for i, row in enumerate(df_texts):
        for i1, item in enumerate(row):
            item['vector'] = vectors[index]
            item['field'] = item['field'] + str(i1)
            assert texts[index] == item['text']
            index += 1

    logger.info(f'FINISHED EMBEDDING')
    df['texts'] = df_texts
    logger.info(f'The number of vectors to be ingested: {len([item["vector"] for row in df["texts"] for item in row])}')

    return df


def ingest(df):
    '''
    Given the data ingest it into ES instance.
    '''
    # increase the timeout if necessary
    es_client = Elasticsearch([ES_HOST], http_auth=(ES_USERNAME, ES_PASSWORD), timeout=20)
    es_client.indices.delete(index=ES_INDEX, ignore=404)
    es_client.indices.create(index=ES_INDEX, settings=MAPPING['settings'], mappings=MAPPING['mappings'])
    # play with chunk size parameter for timed out problem
    final_json = df.to_dict(orient='records')
    deque(parallel_bulk(es_client, actions=final_json, index=ES_INDEX, max_chunk_bytes=5*1024*1024), maxlen=0)
    es_client.indices.refresh()


def get_transformed_data():
    DATA_PATH = Path.joinpath(Path(__file__).parents[1], 'data/transformed')

    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            (
                'Folder \'/data/transformed\' not available.'
                ' Data folder is empty or not created. Make sure to create data folder.'
                ' Follow the instruction in the \'README-es-ingesting-data.md\' file.'
            )
        )
    
    DATA_FILE_NAMES = sorted(DATA_PATH.iterdir())
    
    df = pd.DataFrame()
    for f in DATA_FILE_NAMES:
        df = pd.concat([df, pd.read_json(f)], ignore_index = True, axis = 0)

    return df


if __name__ == '__main__':
    df = get_transformed_data()
    df = embed_data(df)
    ingest(df)


    

    
    