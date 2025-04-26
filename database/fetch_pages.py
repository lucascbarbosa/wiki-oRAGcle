"""Script to fetch all pages."""
import cloudscraper
import pandas as pd
import re

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
    while apcontinue is not None or len(list__pages_df) == 0:
        # Add apcontinue if exists
        if apcontinue is not None:
            params['apcontinue'] = apcontinue
        response = scraper.get(url, params=params)
        pages_data = response.json()
        pages_df = pd.DataFrame(pages_data['query']['allpages'])[['pageid', 'title']]
        print(f"""
            Lista {len(list__pages_df) + 1}.
            Apcontinue: {apcontinue}.
            Start: {pages_data['query']['allpages'][0]}.
            End: {pages_data['query']['allpages'][-1]}.\n
            """)
        # Updated apcontinue if exists
        if 'continue' in pages_data:
            apcontinue = pages_data['continue']['apcontinue']
        else:
            apcontinue = None
        list__pages_df.append(pages_df)

    all__pages_df = pd.concat(list__pages_df)
    return all__pages_df


def fetch__page_content(pageid: int):
    """Fetch a page content based on page id.

    Args:
        pageid (int): page id.
    """
    def _wikitext_to_markdown(wikitext):
        # 1. Remove templates like {{Infobox television}} and {{References}}
        wikitext = re.sub(r'\{\{[^{}]*\}\}', '', wikitext)

        # 2. Remove <ref>...</ref> references (single-line and multi-line)
        wikitext = re.sub(r'<ref[^>]*>.*?<\/ref>', '', wikitext, flags=re.DOTALL)

        # 3. Remove any remaining <ref .../> self-closing tags
        wikitext = re.sub(r'<ref[^\/>]*/>', '', wikitext)

        # 4. Convert section titles (== Title == → ## Title)
        wikitext = re.sub(
            r'^(=+)\s*(.*?)\s*\1$', lambda m: "#" * len(m.group(1)) + " " +
            m.group(2), wikitext, flags=re.MULTILINE)

        # 5. Replace ''italic'' and '''bold'''
        wikitext = re.sub(r"'''''(.*?)'''''", r'***\1***', wikitext)
        wikitext = re.sub(r"'''(.*?)'''", r'**\1**', wikitext)
        wikitext = re.sub(r"''(.*?)''", r'*\1*', wikitext)

        # 6. Convert wiki links
        wikitext = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', wikitext)
        wikitext = re.sub(r'\[\[([^\]]+)\]\]', r'\1', wikitext)

        # 7. Remove external link references [https://url Title]
        wikitext = re.sub(
            r'\[https?:\/\/[^\s\]]+\s([^\]]+)\]', r'\1', wikitext)

        # 8. Remove categories
        wikitext = re.sub(r'\[\[Category:[^\]]+\]\]', '', wikitext)

        # 9. Clean up multiple newlines
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
    content = data['query']['pages'][str(pageid)]['revisions'][0]['*']
    content = _wikitext_to_markdown(content)
    return content


pages_df = fetch__list_pages()
