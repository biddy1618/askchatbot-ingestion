'''
Script for creating logging index in ES instance.

Author: Dauren Baitursyn
Date: 01.11.22
'''

pipeline_id = "transform_id"
mapping  = {
    "pipeline": {
        "id"    : pipeline_id,
        "body"  : {
            "description"   : "Replace the _id with chat_id for the logs index",
            "processors"    : [{
                "set": {
                    "field": "_id",
                    "value": "{{chat_id}}"
                }
            }]
        }
    },
    "settings": {
        "number_of_shards"  : 2, 
        "number_of_replicas": 1,
        "default_pipeline"  : pipeline_id
    },
    "mappings": {
        "dynamic"   : "false",
        "_source"   : {"enabled": "true"},
        "properties": {
            "chat_id"       : {"type": "keyword", "index": "true", "doc_values": "false", "ignore_above": 256},
            "timestamp"     : {"type": "date"   , "index": "true", "doc_values": "true"},
            "chat_history"  : {
                "dynamic"       : "false",
                "type"          : "nested",
                "properties"    : {
                    "agent"     : {"type": "keyword"        , "index": "false", "doc_values": "false", "ignore_above": 256  },
                    "timestamp" : {"type": "date"           , "index": "false", "doc_values": "false"                       },
                    "text"      : {"type": "match_only_text"                                                                },
                    "intent"    : {"type": "keyword"        , "index": "false", "doc_values": "false", "ignore_above": 256  },
                    "results"   : {
                        "dynamic"   : "false",
                        "type"      : "nested",
                        "properties": {
                            "score"     : {"type": "keyword"        , "index": "false", "doc_values": "false", "ignore_above": 256  },
                            "url"       : {"type": "keyword"        , "index": "false", "doc_values": "false", "ignore_above": 256  }
                        }
                    }
                }
            }
        }
    }
}

from elasticsearch import Elasticsearch
from elasticsearch.client import IngestClient

ES_USERNAME = 'elastic'
ES_PASSWORD = 'changeme'
ES_INDEX    = 'logs'
ES_HOST     = 'http://localhost:9200/'



if __name__ == '__main__':
    # increase the timeout if necessary
    es_client = Elasticsearch([ES_HOST], http_auth=(ES_USERNAME, ES_PASSWORD), timeout = 20)
    es_ingest = IngestClient(es_client)

    # create pipeline for replacing _id with chat_id
    es_ingest.put_pipeline(
        id   = mapping['pipeline']['id'     ],
        body = mapping['pipeline']['body'   ])

    # create index
    es_client.indices.delete(
        index   = ES_INDEX.es_logging_index, 
        ignore  = 404)
    es_client.indices.create(
        index       = ES_INDEX   , 
        settings    = mapping['settings']       , 
        mappings    = mapping['mappings']       )

    es_client.indices.refresh()