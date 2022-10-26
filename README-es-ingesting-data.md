# Ingesting data into the ES

## Downloading data

### AskExtension knowledge base data

Download data as indicated in this [notebook](./scripts/es_download_ae_kb.ipynb) and data should appear in folder `actions/es/data/askextension_kb`.

#### Person of contact

For details, contact [Autunm Greenley](autumn.greenley@eduworks.com).

### UC IPM crawled data

Data can be obtained through DVC ([installation guide](https://wiki.eduworks.com/Information_Technology/MLOps/DATA-Installing-DVC)). Clone the [repository](https://git.eduworks.us/data/ask-extension/uc-ipm-web-scrape) for scraped data, install Google Cloud Client - `gcloud` (more in installation guide), authenticate, and pull the data through dvc. Please, contact admin for access rights.

The current version of data corresponds to this commit - [7a9aa45ed3797921973ca60f2c75d62bbef13114](https://git.eduworks.us/data/ask-extension/uc-ipm-web-scrape/-/commit/7a9aa45ed3797921973ca60f2c75d62bbef13114) at [data repo](https://git.eduworks.us/data/ask-extension/uc-ipm-web-scrape).

Copy the downloaded folders through DVC - `scrape_cleaned_Dec2021` and `scrape_cleaned_Apr2022` - into folders `data/uc-ipm/scrape_cleaned_Dec2021` and `data/uc-ipm/scrape_cleaned_Apr2022` correspondingly.

### Oklahome State University crawled data

Clone [data repo](https://git.eduworks.us/data/ask-extension/oku-xmltojson) and download data using DVC.

The current version of data corresponds to this commit - [39a7adacd302fcaf66ef3168b76d8cbcb7d2c2cf](https://git.eduworks.us/data/ask-extension/oku-xmltojson/-/commit/39a7adacd302fcaf66ef3168b76d8cbcb7d2c2cf) at [data repo](https://git.eduworks.us/data/ask-extension/oku-xmltojson).

Copy the downloaded data `fact-sheets-out-cleaner.json` to folder `data/okstate`.

### Oregon State University crawled data

Clone [data repo](https://git.eduworks.us/data/ask-extension/oregon-state-extension-content) and download data using DVC.

The current version of data corresponds to this commit - (63ae4450124ab06cd6c98222c2af62de69c2ee82)[https://git.eduworks.us/data/ask-extension/oregon-state-extension-content/-/commit/63ae4450124ab06cd6c98222c2af62de69c2ee82] at [data repo](https://git.eduworks.us/data/ask-extension/oregon-state-extension-content).

Copy the downloaded data `OSU-Out-Cleaner.json` to `data/orstate`.

## EDA of data

Can be found in this [notebook](./scripts/es_eda.ipynb)

## Ingesting data

Run the scripts in the [notebook](./scripts/es_ingest_data.ipynb) to create indexes and ingest data sources from AE knowledge base (California only) and UC IPM.

Run the scripts in the [notebook](./scripts/es_chat_logging.ipynb) to create index for chat history.

