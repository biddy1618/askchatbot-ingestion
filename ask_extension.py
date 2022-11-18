import requests
from tqdm import tqdm 
from datetime import date, timedelta
import json

def get_all_os_ticket_items():
    """Slow way of getting data from OS Ticket API."""
    
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = date(2010, 5, 15)
    end_date = date(2024, 1, 1)
    for single_date in tqdm(daterange(start_date, end_date)):
        # Day after single date 
        day_after = single_date + timedelta(days=1)
        try:
            url = f"https://ask2.extension.org/api/knowledge/{single_date}/{day_after}"
            r = requests.get(url).json()
        except:
            print(url)
            continue
        if r:
            for item in r:
                yield item
                
def get_ask_extension_data() -> list:

    try:
        with open('raw_ask_extension_data.json', 'r', encoding='utf-8') as r:
            return json.load(r)
    except:
        return list(get_all_os_ticket_items())   