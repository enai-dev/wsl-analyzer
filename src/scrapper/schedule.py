# scrapper/schedule.py
import re
from bs4 import BeautifulSoup
from .base_scraper import fetch_page

SCHEDULE_URL = "https://www.worldsurfleague.com/posts/546310/world-surf-league-releases-2026-championship-tour-schedule-and-formats"

def parse_schedule():
    html = fetch_page(SCHEDULE_URL)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    events = []
    
    body = soup.find('div', class_='post-body')
    if not body:
        body = soup.find('article')
    
    if body:
        text = body.get_text()
        lines = text.split('\n')
        for line in lines:
            if 'Stop No.' in line:
                match = re.search(r'Stop No\. \d+ - ([^:]+):', line)
                if match:
                    wave_name = match.group(1).strip()
                    events.append({
                        'wave': wave_name,
                        'raw_line': line.strip()
                    })
    return events

def get_next_event():
    events = parse_schedule()
    if events:
        return events[0]
    return None