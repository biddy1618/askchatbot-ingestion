'''
Module for transforming raw source data to format flexible to be ingested.

Author: Dauren Baitursyn
Date: 31.10.2020
'''
import re
import sys
import html
import json
import logging

import pandas as pd

from copy import deepcopy
from pathlib import Path
from string import punctuation as pn


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# Helper functions
def clean(text):
    '''
    Fix encodings and remove escape and redundant whitespace characters from text.
    '''
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'\s+', ' ', text).strip()
    text = html.unescape(text)
    return text


def get_text_from_fields(row, fields):
    '''
    Get text from text fields.
    '''
    row_items = []
    for field in fields:
        text = row[field]

        if len(text) == 0:
            continue
        
        row_items.append({
            'text': clean(text),
            'field': field,
            'name': field.replace('_', ' ').capitalize(),
            'im_src': ''
        })

    return row_items


def get_text_from_list_field(row, field, subfield, title = False, im_src = None):
    '''
    Get text from list fields.
    '''
    row_items = []
    for item in row[field]:
        text = item[subfield]

        if len(text) == 0:
            continue

        if title:
            text = row['title'] + ' - ' + text
        else: 
            text = text
        
        if im_src and len(item[im_src]) > 0:
            src = item[im_src]
        else: 
            src = ''
        
        row_items.append({
            'text': clean(text),
            'field': field,
            'name': field.replace('_', ' ').capitalize(),
            'im_src': src
        })
    
    return row_items


def get_images(row, field, im_src):
    '''
    Get images from list image fields.
    '''
    row_images = []
    for item in row[field]:
        if len(item[im_src]) > 0:
            row_images.append(item[im_src])
    
    return row_images


def transform_data(df, list_fields, list_text_fields, image_fields, limit = None):
    '''
    Given data transform the data into the required format.
    '''
    if limit:
        df = df.sample(limit).copy(deep=True)
    
    df['source'] = 'ucipm'
    
    for field in list_fields:
        df[field] = df[field].apply(lambda d: d if isinstance(d, list) else [])
    
    cols = [col for col in df.columns[df.applymap(lambda x: isinstance(x, str)).all(0)] if col not in ['url', 'source']]

    texts = []
    images = []
    for _, row in df.iterrows():
        row_texts = []
        row_images = []
        
        row_texts.extend(get_text_from_fields(row, cols))

        for field, subfield, concat_title, im_src in list_text_fields:
            row_texts.extend(get_text_from_list_field(row, field, subfield, title=concat_title, im_src=im_src))
        
        for field, subfield in image_fields:
            row_images.extend(get_images(row, field, subfield))

        texts.append(row_texts)
        images.append(row_images)
    
    df['texts'] = texts
    df['images'] = images

    df = df.loc[:, ['source', 'url', 'title', 'texts', 'images']]

    return df


def transform_table(row):
    '''
    Rename the 'tips_table' key values to title with title and header concatenation.
    '''
    if len(row['tips_table']) > 0:
        items = row['tips_table']
        assert 'header' in items[0] 
        header_title = row['title'] + ' - ' + items[0]['header']
        row['tips_table'] = header_title
    else:
        row['tips_table'] = ''


def transform_pesticide(row):
    '''
    Merge pesticide subfield into main field - information.
    '''
    information = row['information'][0]
    texts = []
    for k, v in information.items():
        texts.append(k.replace('_', ' ').capitalize() + ': ' + v + '. ')
    row['information'] = '. '.join(texts)


def transform_answer(answer_dict):
    '''
    Convert answer field from a dictionary to a list.
    '''
    answers = [{}] * len(answer_dict)
    
    for k, v in answer_dict.items():
        # clean the response up
        v = {
            'response' : clean(v['response']),
        }
        answers[int(k) - 1] = v
    
    return answers


def transform_title(title):
    '''
    Remove question ID from title, and append '.' in the end
    if no punctuation was detected.

    Example with '#' - 437259
    Example with '...' - 437264
    '''
    title = ''.join(title.split('#')[:-1]).strip().strip('...')
    
    # add a '.' if it does not yet end with a punctuation
    title = title if (title and title[-1] in pn) else title + '.'
    
    return title


def merge_title_question(df):
    '''
    Create new column from questions and title,
    but only if it is not already exactly in the question.
    '''
    titles      = df['title'    ].tolist()
    questions   = df['question' ].tolist()
    
    tqs = [
        question
        if (title and question.startswith(title[:-1]))
        else title + " " + question
        for (title, question) in zip(titles, questions)
    ]

    return tqs


def get_title_and_description(row, thumbnail = False):
    '''
    Transform the title and description fields.
    '''
    title = clean(row['title'])
    description = clean(row['description'])
    if thumbnail:
        im_src = row['thumbnail']
    else:
        im_src = ''

    texts = []
    texts.append({
        'text': title,
        'field': 'title',
        'name': 'Title',
        'im_src': im_src
    })
    if len(description) > 0:
        texts.append({
            'text': description,
            'field': 'description',
            'name': 'Description',
            'im_src': im_src
        })

    return texts

def get_contents_and_images(row, thumbnail=False):
    '''
    Transform the content field by concatenating title with header, and perform cleaning. Drop the unncessary columns.
    '''
    title = clean(row['title'])
    if thumbnail:
        im_src = row['thumbnail']
    else:
        im_src = ''

    texts = []
    images = []

    if thumbnail:
        images.append(im_src)
    for content in row['content']:
        item = {}
        header = clean(content['header'])
        if len(header) == 0 or header == 'Introduction-w/o-header':
            header = clean(title)
        else:
            header = clean(title + ' - ' + header)
        item['text'] = header
        item['field'] = 'content'
        item['name'] = 'Paragraph'
        if len(content['images']['image_urls']) > 0:
            item['im_src'] = content['images']['image_urls'][0]
        elif thumbnail:
            item['im_src'] = im_src
        else:
            item['im_src'] = ''
        texts.append(item)
        
        text = clean(content['text'])
        if len(text) > 0:
            item = deepcopy(item)
            item['text'] = clean(content['text'])
        
            texts.append(item)
        
        for url, caption in zip(content['images']['image_urls'], content['images']['image_captions']):
            item = {}
            if len(caption) > 0:
                item['text'] = clean(header + ' - ' + caption)
                item['field'] = 'image'
                item['name'] = 'Image'
                item['im_src'] = url
                texts.append(item)
            if len(url) > 0:
                images.append(url)
    
    return texts, images

# Main functions
def transform_uc_ipm_dec21():
    '''
    UC IPM December 2021 data transformation.

    Data is fetched from 'data/uc-ipm/scrape_cleaned_Dec2021' folder.
    The transformed data is saved in 'data/transformed/ucipm-Dec2021.json' file  
    '''
    DATA_PATH = Path.joinpath(
        Path(__file__).parents[1],
        'data/uc-ipm/scrape_cleaned_Dec2021'
    )

    SAVE_FILE = Path.joinpath(
        Path(__file__).parents[1],
        'data/transformed/ucipm-Dec2021.json'
    )
    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            (
                'Folder \'/data/uc-ipm/scrape_cleaned_Dec2021\' not available.'
                ' Data folder is empty or not created. Make sure to create data folder.'
                ' Follow the instruction in the \'README-es-ingesting-data.md\' file.'
            )
        )

    DATA_FILE_NAMES = {
        'exoticPests.json',
        'fruitItems_new.json',
        'fruitVeggieEnvironItems_new.json',
        'pestDiseaseItems_new.json',
        'plantFlowerItems.json',
        'turfPests.json',
        'veggieItems_new.json',
        'weedItems.json'
    }

    try:
        assert set(data_file.name for data_file in DATA_PATH.iterdir()) == DATA_FILE_NAMES
    except AssertionError:
        raise FileNotFoundError(
            (
                'Data folder \'scrape_cleaned_dec2021\' doesn\' contain all the files.'
                ' Please check the commit hash of the data source and make sure it'
                ' corresponds to the one in the \'README-es-ingesting-data.md\' file.'
            )
        )

    logger.info(f'Transforming UC IPM Dec 2021 crawl...')

    final_df = pd.DataFrame()

    FILE_NAME = 'exoticPests.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title'}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images', 'related_links'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
            ('related_links', 'text', True, None),        
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'fruitItems_new.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title'}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['cultural_tips', 'pests_and_disorders'],
        list_text_fields=[
            ('cultural_tips', 'tip', True, None),
            ('pests_and_disorders', 'problem', True, None),        
        ],
        image_fields=[],
        limit=10
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'fruitVeggieEnvironItems_new.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title',}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),        
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'pestDiseaseItems_new.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title',}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),        
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'plantFlowerItems.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title',}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images', 'pests_and_disorders'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
            ('pests_and_disorders', 'problem', True, None),        
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'turfPests.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title',}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),        
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'veggieItems_new.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name'  : 'title'}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images', 'pests_and_disorders'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
            ('pests_and_disorders', 'problem', True, None),        
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'weedItems.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title',}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, None),        
        ],
        image_fields=[],
        limit=30
    )

    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    logger.info(f'Final shape is :{final_df.shape}')

    final_df.to_json(SAVE_FILE, orient='records')
    logger.info(f'Saved file to {SAVE_FILE}')


def transform_uc_ipm_apr22():
    '''
    UC IPM December 2022 data transformation.

    Data is fetched from 'data/uc-ipm/scrape_cleaned_Apr2022' folder.
    The transformed data is saved in 'data/transformed/ucipm-Dec2021.json' file  
    '''
    
    DATA_PATH = Path.joinpath(
        Path(__file__).parents[1],
        'data/uc-ipm/scrape_cleaned_Apr2022'
    )

    SAVE_FILE = Path.joinpath(
        Path(__file__).parents[1],
        'data/transformed/ucipm-Apr2022.json'
    )
    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            (
                'Folder \'/data/uc-ipm/scrape_cleaned_Dec2021\' not available.'
                ' Data folder is empty or not created. Make sure to create data folder.'
                ' Follow the instruction in the \'README-es-ingesting-data.md\' file.'
            )
        )

    DATA_FILE_NAMES = {
        'FruitVegCulturalItems.json',
        'GardenControlsPestItems.json',
        'GardenControlsPesticideItems.json',
        'PestNotes.json',
        'QuickTips.json',
        'Videos.json',
        'WeedIdItems.json'
    }

    try:
        assert set(data_file.name for data_file in DATA_PATH.iterdir()) == DATA_FILE_NAMES
    except AssertionError:
        raise FileNotFoundError(
            (
                'Data folder \'scrape_cleaned_dec2021\' doesn\' contain all the files.'
                ' Please check the commit hash of the data source and make sure it'
                ' corresponds to the one in the \'README-es-ingesting-data.md\' file.'
            )
        )
    
    logger.info(f'Transforming UC IPM Apr 2022 crawl...')

    final_df = pd.DataFrame()

    FILE_NAME = 'FruitVegCulturalItems.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title'}, inplace = True)
    df['tips_table'] = df['tips_table'].apply(lambda d: d if isinstance(d, list) else [])
    df.apply(lambda r: transform_table(r), axis = 1)


    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'GardenControlsPestItems.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name': 'title'}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
        ],
        image_fields=[('images', 'src')],
        limit=10
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)


    FILE_NAME = 'GardenControlsPesticideItems.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df['title'] = df[['active_ingredient', 'pesticide_type']].agg(' - '.join, axis=1)
    df.drop(['active_ingredient', 'pesticide_type'], axis=1, inplace=True)
    df.apply(lambda r: transform_pesticide(r), axis = 1)

    df = transform_data(
        df, 
        list_fields=[],
        list_text_fields=[],
        image_fields=[],
        limit=10
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'PestNotes.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {
        'urlPestNote'           : 'url'         ,
        'name'                  : 'title'       ,
        'descriptionPestNote'   : 'description' ,
        'lifecyclePestNote'     : 'lifecycle'   ,
        'damagePestNote'        : 'damage'      ,
        'managementPestNote'    : 'management'  ,
        'imagePestNote'         : 'images'      ,
    }, inplace = True)
    df.drop('tablePestNote', axis=1, inplace=True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'QuickTips.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {
        'urlQuickTip'           : 'url'     ,
        'name'                  : 'title'   ,
        'contentQuickTips'      : 'content' ,
        'imageQuickTips'        : 'images'  ,
    }, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
        ],
        image_fields=[('images', 'src')],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'Videos.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))

    df = transform_data(
        df, 
        list_fields=[],
        list_text_fields=[],
        image_fields=[],
        limit=30
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    FILE_NAME = 'WeedIdItems.json'
    logger.info(f'Transforming "{FILE_NAME}"...')
    df = pd.read_json(Path.joinpath(DATA_PATH, FILE_NAME))
    df.rename(columns = {'name'  : 'title',}, inplace = True)

    df = transform_data(
        df, 
        list_fields=['images'],
        list_text_fields=[
            ('images', 'caption', True, 'src'),
        ],
        image_fields=[('images', 'src')],
        limit=10
    )
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    logger.info(f'Final shape is :{final_df.shape}')

    final_df.to_json(SAVE_FILE, orient='records')
    logger.info(f'Saved file to {SAVE_FILE}')


def transform_ae_kb():
    '''
    AE KB data transformation.

    Data is fetched from 'data/askextension_kb' folder.
    The transformed data is saved in 'data/transformed/ae_kb.json' file  
    '''
    DATA_PATH = Path.joinpath(
        Path(__file__).parents[1],
        'data/askextension_kb'
    )

    SAVE_FILE = Path.joinpath(
        Path(__file__).parents[1],
        'data/transformed/ae_kb.json'
    )
    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            (
                'Folder \'/data/askextension_kb\' not available.'
                ' Data folder is empty or not created. Make sure to create data folder.'
                ' Follow the instruction in the \'README-es-ingesting-data.md\' file.'
            )
        )
    DATA_FILE_NAMES = sorted(DATA_PATH.iterdir())

    logger.info(f'Transforming AE KB data...')

    logger.info(f'List of files:\n{[data_file.name for data_file in DATA_FILE_NAMES]}')

    # Combines the data files into one and returns it.
    df = pd.DataFrame()
    for f in DATA_FILE_NAMES:
        df = pd.concat([df, pd.read_json(f)], ignore_index = True, axis = 0)


    # Modify STATE_FILTER and MIN_WORD_COUNT variables accordingly
    # STATE_FILTER    = ['California', 'Oklahoma', 'Oregon']
    STATE_FILTER    = ['California']
    MIN_WORD_COUNT  = 3

    ASKEXTENSION_QUESTION_URL = 'https://ask2.extension.org/kb/faq.php?id='

    df['source'] = 'ae-kb'
    df['faq-id'] = df['faq-id'].astype(str)
    df = df[df['state'].isin(STATE_FILTER)]
    df['url'] = [
        f"{ASKEXTENSION_QUESTION_URL}{ticket_no}" if len(ticket_no) == 6 else ""
        for ticket_no in df['title'].str.split('#').str[-1]
    ]
    df['ticket-no'] = [
        ticket_no if len(ticket_no) == 6 else ""
        for ticket_no in df['title'].str.split('#').str[-1]
    ]
    df['attachments'] = df['attachments'].apply(lambda d: d if isinstance(d, list) else [])
    df['attachments'] = df['attachments'].apply(lambda d: [{'src': link} for link in d])
    df.rename(columns = {'faq-id': 'faq_id', 'ticket-no': 'ticket_no'}, inplace = True)

    df['answers'] = df['answer'].apply(transform_answer)
    df['title'] = df['title'].apply(transform_title)
    df['question'] = merge_title_question(df)

    if MIN_WORD_COUNT:
        df = df[df['question'].str.split().str.len() > MIN_WORD_COUNT]

    df = df.loc[:, ['source', 'url', 'title', 'question', 'answers', 'attachments']]
    df.reset_index(drop=True, inplace=True)

    df = transform_data(
        df, 
        list_fields=['answers',],
        list_text_fields=[
            ('answers', 'response', False, None),
        ],
        image_fields=[('attachments', 'src')],
        limit=30
    )

    logger.info(f'Final shape is :{df.shape}')

    df.to_json(SAVE_FILE, orient='records')
    logger.info(f'Saved file to {SAVE_FILE}')


def transform_okstate():
    '''
    Oklahome State University data transformation.

    Data is fetched from 'data/okstate/fact-sheets-out-cleaner.json' file.
    The transformed data is saved in 'data/transformed/okstate.json' file  
    '''
    FILE_PATH = Path.joinpath(
        Path(__file__).parents[1],
        'data/okstate/fact-sheets-out-cleaner.json'
    )

    SAVE_FILE = Path.joinpath(
        Path(__file__).parents[1],
        'data/transformed/okstate.json'
    )
    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not FILE_PATH.is_file():
        raise FileNotFoundError(
            (
                'File \'data/okstate/fact-sheets-out-cleaner.json\' not available.'
                ' File folder is empty or not created. Make sure to create data folder.'
                ' Follow the instruction in the \'README-es-ingesting-data.md\' file.'
            )
        )


    logger.info(f'Transforming Oklahome State University data...')
    df = pd.read_json(FILE_PATH)

    df['source'] = 'okstate'

    df['title'] = df['title'].apply(clean).fillna('Auxilary')
    df.rename(columns={'link': 'url'}, inplace=True)
    df.drop(columns=['author', 'pubdate', 'category', 'displaydate'], inplace=True)

    texts = []
    images = []
    for _, row in df.iterrows():
        title_description_texts = get_title_and_description(row, thumbnail=True)
        row_texts, row_images = get_contents_and_images(row, thumbnail=True)
        title_description_texts.extend(row_texts)
        
        texts.append(row_texts)
        images.append(row_images)

    df['texts'] = texts
    df['images'] = images

    df = df.loc[:, ['source', 'url', 'title', 'texts', 'images']]
    df = df.sample(50).reset_index(drop=True)

    logger.info(f'Final shape is :{df.shape}')

    df.to_json(SAVE_FILE, orient='records')
    logger.info(f'Saved file to {SAVE_FILE}')


def transform_orstate():
    '''
    Oregon State University data transformation.

    Data is fetched from 'data/orstate/OSU-Out-Cleaner.json' file.
    The transformed data is saved in 'data/transformed/orstate.json' file  
    '''
    FILE_PATH = Path.joinpath(
        Path(__file__).parents[1],
        'data/orstate/OSU-Out-Cleaner.json'
    )

    SAVE_FILE = Path.joinpath(
        Path(__file__).parents[1],
        'data/transformed/orstate.json'
    )
    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not FILE_PATH.is_file():
        raise FileNotFoundError(
            (
                'File \'data/orstate/OSU-Out-Cleaner.json\' not available.'
                ' File folder is empty or not created. Make sure to create data folder.'
                ' Follow the instruction in the \'README-es-ingesting-data.md\' file.'
            )
        )


    logger.info(f'Transforming Oregon State University data...')
    
    df = pd.read_json(FILE_PATH)
    df['source'] = 'orstate'

    df['title'] = df['title'].apply(clean).fillna('Auxilary')
    df.rename(columns={'link': 'url'}, inplace=True)
    df.drop(columns=['author', 'pubdate', 'category', 'displaydate'], inplace=True)

    texts = []
    images = []
    for i, row in df.iterrows():
        title_description_texts = get_title_and_description(row)
        row_texts, row_images = get_contents_and_images(row)
        title_description_texts.extend(row_texts)
        
        texts.append(row_texts)
        images.append(row_images)

    df['texts'] = texts
    df['images'] = images

    df = df.loc[:, ['source', 'url', 'title', 'texts', 'images']]
    df = df.sample(50).reset_index(drop=True)

    logger.info(f'Final shape is :{df.shape}')

    df.to_json(SAVE_FILE, orient='records')
    logger.info(f'Saved file to {SAVE_FILE}')
    
    
if __name__ == '__main__':
    transform_uc_ipm_dec21()
    transform_uc_ipm_apr22()
    transform_ae_kb()
    transform_okstate()
    transform_orstate()


    
