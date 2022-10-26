import os
import sys
import logging

from elasticsearch import AsyncElasticsearch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

es_username     = os.getenv('ES_USERNAME'       , 'elastic'                 )
es_password     = os.getenv('ES_PASSWORD'       , 'changeme'                )
es_host         = os.getenv('ES_HOST'           , 'http://localhost:9200/'  )
embed_cache_dir = os.getenv('TFHUB_CACHE_DIR'   , '/var/tmp/models'         )

# embed_url = 'https://tfhub.dev/google/universal-sentence-encoder/4' # 512
# embed_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5' # 512
# embed_url = 'all-MiniLM-L6-v2' # 384
# embed_url = 'paraphrase-MiniLM-L6-v2'# 384
# embed_url = 'bert-base-nli-mean-tokens' # 768
# embed_url = 'stsb-distilbert-base' # 768
# embed_url = 'paraphrase-multilingual-MiniLM-L12-v2' # 384
# embed_url = 'all-mpnet-base-v2' # 768
## Best one so far
embed_url = 'all-distilroberta-v1' # 768
# embed_url = 'distiluse-base-multilingual-cased-v2' # 512
# embed_url = 'paraphrase-mpnet-base-v2' # 768
# embed_url = 'paraphrase-MiniLM-L12-v2' # 384
# embed_url = 'paraphrase-xlm-r-multilingual-v1' # 768
# embed_url = 'distiluse-base-multilingual-cased' # 512
# embed_url = 'paraphrase-distilroberta-base-v1' # 768
# embed_url = 'allenai-specter' # 768
# embed_url = 'paraphrase-multilingual-mpnet-base-v2' # 768
# embed_url = 'paraphrase-distilroberta-base-v2' # 768
# embed_url = 'multi-qa-MiniLM-L6-cos-v1' # 384
# embed_url = 'LaBSE' # 768
# embed_url = 'distilbert-base-nli-stsb-mean-tokens' # 768
# embed_url = 'multi-qa-mpnet-base-dot-v1' # 768
# embed_url = 'paraphrase-MiniLM-L3-v2' # 384
# embed_url = 'sentence-t5-large' # 768
# embed_url = 'stsb-roberta-base-v2' # 768
# embed_url = 'distiluse-base-multilingual-cased-v1' # 512
# embed_url = 'msmarco-distilbert-dot-v5' # 768
# embed_url = 'multi-qa-mpnet-base-cos-v1' # 768
# embed_url = 'all-mpnet-base-v1' # 768

es_combined_index   = 'test'
es_logging_index    = 'logs'
es_field_limit      = 32766

es_search_size  = 100
es_cut_off      = 0.4
es_top_n        = 10
es_downweight   = 1

# import tensorflow_hub as tf_hub
from sentence_transformers import SentenceTransformer

logger.info('----------------------------------------------')
logger.info('Elasticsearch configuration:')
logger.info(f'- host                = {es_host          }')
logger.info(f'- embed_url           = {embed_url        }')
logger.info(f'- embed_cache_dir     = {embed_cache_dir  }')
logger.info('----------------------------------------------')

logger.info('----------------------------------------------')
logger.info('Elasticsearch indexes:')
logger.info(f'- combined index      = {es_combined_index}'  )
logger.info(f'- logging index       = {es_logging_index}'   )
logger.info('----------------------------------------------')

logger.info('Initializing the Elasticsearch client')
es_client = AsyncElasticsearch(
    [es_host], http_auth=(es_username, es_password))
logger.info('Done initiliazing ElasticSearch client')

logger.info(f'Start loading embedding module - {embed_url}')
# embed = tf_hub.load(embed_url)
embed = SentenceTransformer(
    model_name_or_path  = embed_url         ,
    cache_folder        = embed_cache_dir   ,
    device              = 'cpu'             )
logger.info(f'Done loading embedding module - {embed_url}')
# -------------------------------------------------------------
