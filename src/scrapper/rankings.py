# scrapper/rankings.py
import re
from bs4 import BeautifulSoup
from .base_scraper import fetch_page

RANKINGS_MEN_URL = "https://www.worldsurfleague.com/athletes/tour/mct?year=2025"
RANKINGS_WOMEN_URL = "https://www.worldsurfleague.com/athletes/tour/wct?year=2025"

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def parse_rankings(url, category):
    html = fetch_page(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    rankings = []
    
    table = soup.find('table', class_='table')
    if not table:
        table = soup.find('table')
    if not table:
        print(f"No se encontró tabla en {url}")
        return []
    
    rows = table.find_all('tr')
    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) >= 3:
            rank_text = clean_text(cols[0].get_text())
            name_text = clean_text(cols[1].get_text())
            points_text = clean_text(cols[-1].get_text())
            
            rank_match = re.search(r'(\d+)', rank_text)
            rank = int(rank_match.group(1)) if rank_match else 999
            
            rankings.append({
                'rank': rank,
                'name': name_text,
                'points': points_text,
                'category': category,
                'source_url': url
            })
    return rankings

def get_men_rankings():
    return parse_rankings(RANKINGS_MEN_URL, 'men')

def get_women_rankings():
    return parse_rankings(RANKINGS_WOMEN_URL, 'women')