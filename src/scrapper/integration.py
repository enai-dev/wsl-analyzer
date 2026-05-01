# scrapper/integration.py
import pandas as pd
from .rankings import get_men_rankings, get_women_rankings
from .schedule import get_next_event
from .historical import get_historical_scores

def assign_tier_by_rank(rank):
    if rank <= 5:
        return 'A'
    elif rank <= 15:
        return 'B'
    else:
        return 'C'

def get_surfers_dataframe(wave_name=None, use_mock_if_no_data=True):
    hombres = get_men_rankings()
    mujeres = get_women_rankings()
    
    if not hombres and use_mock_if_no_data:
        print("No se pudieron obtener rankings reales. Usando datos mock para hombres.")
        hombres = [
            {'rank': 1, 'name': 'John John Florence', 'points': '100', 'category': 'men'},
            {'rank': 2, 'name': 'Gabriel Medina', 'points': '95', 'category': 'men'},
            {'rank': 3, 'name': 'Filipe Toledo', 'points': '92', 'category': 'men'},
            {'rank': 4, 'name': 'Italo Ferreira', 'points': '88', 'category': 'men'},
            {'rank': 5, 'name': 'Griffin Colapinto', 'points': '85', 'category': 'men'},
            {'rank': 6, 'name': 'Ethan Ewing', 'points': '80', 'category': 'men'},
            {'rank': 7, 'name': 'Jack Robinson', 'points': '78', 'category': 'men'},
            {'rank': 8, 'name': 'Kanoa Igarashi', 'points': '75', 'category': 'men'},
        ]
    if not mujeres and use_mock_if_no_data:
        print("No se pudieron obtener rankings reales. Usando datos mock para mujeres.")
        mujeres = [
            {'rank': 1, 'name': 'Carissa Moore', 'points': '100', 'category': 'women'},
            {'rank': 2, 'name': 'Stephanie Gilmore', 'points': '96', 'category': 'women'},
            {'rank': 3, 'name': 'Tyler Wright', 'points': '92', 'category': 'women'},
            {'rank': 4, 'name': 'Molly Picklum', 'points': '88', 'category': 'women'},
            {'rank': 5, 'name': 'Lakey Peterson', 'points': '85', 'category': 'women'},
            {'rank': 6, 'name': 'Tatiana Weston-Webb', 'points': '81', 'category': 'women'},
        ]
    
    for s in hombres:
        s['tier'] = assign_tier_by_rank(s['rank'])
    for s in mujeres:
        s['tier'] = assign_tier_by_rank(s['rank'])
    
    if wave_name is None:
        next_event = get_next_event()
        if next_event:
            wave_name = next_event['wave']
            print(f"Próxima ola detectada: {wave_name}")
        else:
            wave_name = "Desconocida"
    
    historical = get_historical_scores(wave_name)
    
    for s in hombres + mujeres:
        base_score = max(0, 20 - (s['rank'] - 1) * 0.5)
        name = s['name']
        if name in historical:
            hist_avg = historical[name]
            s['total_score'] = 0.6 * hist_avg + 0.4 * base_score
        else:
            s['total_score'] = base_score
    
    df = pd.DataFrame(hombres + mujeres)
    df = df.rename(columns={'name': 'surfer'})
    df['gender'] = df['category'].map({'men': 'M', 'women': 'F'})
    
    return df[['surfer', 'gender', 'tier', 'total_score']]

def suggest_power_surfers(df):
    hombres = df[df.gender == 'M'].nlargest(1, 'total_score')
    mujeres = df[df.gender == 'F'].nlargest(1, 'total_score')
    power_man = hombres['surfer'].iloc[0] if not hombres.empty else None
    power_woman = mujeres['surfer'].iloc[0] if not mujeres.empty else None
    return power_man, power_woman