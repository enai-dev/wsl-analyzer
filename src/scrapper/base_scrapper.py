# scrapper/base_scraper.py
import time
import random
import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}
DELAY = 2

def fetch_page(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                time.sleep(random.uniform(DELAY, DELAY + 1))
                return response.text
            else:
                print(f"Error {response.status_code} en {url}")
        except Exception as e:
            print(f"Intento {attempt+1} fallido para {url}: {e}")
            time.sleep(5)
    return None

def fetch_dynamic_page(url):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            content = page.content()
            browser.close()
            return content
    except ImportError:
        print("Playwright no instalado. Usando fetch_page normal.")
        return fetch_page(url)
