# scrapper/historical.py
import json
import os
from .base_scraper import fetch_page

CACHE_DIR = "data/historical"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_historical_scores(wave_name, force_refresh=False):
    cache_file = os.path.join(CACHE_DIR, f"{wave_name.replace(' ', '_')}.json")
    
    if not force_refresh and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Datos mock mientras no se implemente scraping real
    print(f"Obteniendo histórico para {wave_name} (usando datos de ejemplo)")
    mock_scores = {
        "John John Florence": 16.5,
        "Gabriel Medina": 15.8,
        "Filipe Toledo": 16.2,
        "Italo Ferreira": 15.5,
        "Carissa Moore": 17.2,
        "Tyler Wright": 15.9,
        "Stephanie Gilmore": 16.8,
        "Molly Picklum": 16.0,
    }
    
    with open(cache_file, 'w') as f:
        json.dump(mock_scores, f)
    
    return mock_scores

def update_historical_for_all_events():
    from .schedule import parse_schedule
    events = parse_schedule()
    for event in events:
        wave = event['wave']
        print(f"Actualizando histórico para {wave}...")
        get_historical_scores(wave, force_refresh=True)