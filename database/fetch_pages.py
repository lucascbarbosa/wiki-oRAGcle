import cloudscraper
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

scraper = cloudscraper.create_scraper()

def fetch__list_pages():
    """Fetch a list of all pages."""
    url = "https://awoiaf.westeros.org/api.php"
    apcontinue = None
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": "max",
    }
    list__pages_df = []
    print("Listando páginas...")

    with tqdm(desc="Páginas", unit=" lotes") as pbar:
        while apcontinue is not None or len(list__pages_df) == 0:
            if apcontinue is not None:
                params['apcontinue'] = apcontinue
            response = scraper.get(url, params=params)
            pages_data = response.json()
            pages_df = pd.DataFrame(pages_data['query']['allpages'])[['pageid', 'title']]
            pbar.update(1)
            if 'continue' in pages_data:
                apcontinue = pages_data['continue']['apcontinue']
            else:
                apcontinue = None
            list__pages_df.append(pages_df)

    all__pages_df = pd.concat(list__pages_df, ignore_index=True)
    return all__pages_df


def fetch__page_content(pageid: int):
    """Fetch a page content based on page id."""
    def _wikitext_to_markdown(wikitext):
        wikitext = re.sub(r'\{\{[^{}]*\}\}', '', wikitext)
        wikitext = re.sub(r'<ref[^>]*>.*?<\/ref>', '', wikitext, flags=re.DOTALL)
        wikitext = re.sub(r'<ref[^\/>]*/>', '', wikitext)
        wikitext = re.sub(r'^(=+)\s*(.*?)\s*\1$', lambda m: "#" * len(m.group(1)) + " " + m.group(2), wikitext, flags=re.MULTILINE)
        wikitext = re.sub(r"'''''(.*?)'''''", r'***\1***', wikitext)
        wikitext = re.sub(r"'''(.*?)'''", r'**\1**', wikitext)
        wikitext = re.sub(r"''(.*?)''", r'*\1*', wikitext)
        wikitext = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', wikitext)
        wikitext = re.sub(r'\[\[([^\]]+)\]\]', r'\1', wikitext)
        wikitext = re.sub(r'\[https?:\/\/[^\s\]]+\s([^\]]+)\]', r'\1', wikitext)
        wikitext = re.sub(r'\[\[Category:[^\]]+\]\]', '', wikitext)
        wikitext = re.sub(r'\n{3,}', '\n\n', wikitext)
        return wikitext.strip()

    url = "https://awoiaf.westeros.org/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "pageids": pageid,
        "redirect": 1,
    }
    response = scraper.get(url, params=params)
    data = response.json()
    try:
        content = data['query']['pages'][str(pageid)]['revisions'][0]['*']
    except (KeyError, IndexError):
        return ""
    return {'pageid': pageid, 'content': _wikitext_to_markdown(content)}


# Main script
pages_df = fetch__list_pages()
page_ids = pages_df['pageid'].tolist()

# Parallel fetch with progress bar
list_contents_df = []
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = {
        executor.submit(fetch__page_content, pid): pid for pid in page_ids
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="Baixando conteúdo", unit="página"):
        list_contents_df.append(future.result())

contents_df = pd.DataFrame(list_contents_df)

# Merge
pages_df = pages_df.merge(contents_df, on='pageid')

# Delete redirects
pages_df = pages_df[~pages_df['content'].str.startswith("#REDIRECT")]

# Export parquet
pages_df = pages_df.to_parquet("database.parquet")