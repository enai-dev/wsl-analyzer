# src/optimizer/team_selector.py

import pulp
from src.scrapper.integration import get_surfers_dataframe, suggest_power_surfers

def optimize_team(df, power_surfer_man=None, power_surfer_woman=None):
    """
    Selecciona el equipo óptimo de 12 surfistas (6 hombres + 6 mujeres)
    con exactamente 2 por cada tier (A, B, C) en cada género.
    Opcionalmente, duplica la puntuación de los Power Surfers.
    
    Parámetros:
        df: DataFrame con columnas ['surfer', 'gender', 'tier', 'total_score']
        power_surfer_man: nombre del surfista hombre que será Power Surfer (duplica su total_score)
        power_surfer_woman: nombre de la surfista mujer que será Power Surfer
    
    Retorna:
        DataFrame con los surfistas seleccionados (12 filas)
    """
    # Copia para no modificar el original
    data = df.copy()
    
    # Aplicar Power Surfers (duplicar puntuación)
    if power_surfer_man:
        mask_man = (data['surfer'] == power_surfer_man) & (data['gender'] == 'M')
        if mask_man.any():
            data.loc[mask_man, 'total_score'] *= 2
        else:
            print(f"Advertencia: {power_surfer_man} no encontrado en hombres.")
    
    if power_surfer_woman:
        mask_woman = (data['surfer'] == power_surfer_woman) & (data['gender'] == 'F')
        if mask_woman.any():
            data.loc[mask_woman, 'total_score'] *= 2
        else:
            print(f"Advertencia: {power_surfer_woman} no encontrada en mujeres.")
    
    # Crear problema de optimización lineal
    prob = pulp.LpProblem("Fantasy_WSL_Optimizer", pulp.LpMaximize)
    
    # Variables binarias: 1 si se selecciona al surfista i
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in data.index}
    
    # Función objetivo: maximizar la suma de total_score * x_i
    prob += pulp.lpSum(data.loc[i, 'total_score'] * x[i] for i in data.index)
    
    # Restricciones: exactamente 6 hombres y 6 mujeres
    prob += pulp.lpSum(x[i] for i in data[data.gender == 'M'].index) == 6
    prob += pulp.lpSum(x[i] for i in data[data.gender == 'F'].index) == 6
    
    # Restricciones: por cada género y cada tier, exactamente 2 surfistas
    for gender in ['M', 'F']:
        for tier in ['A', 'B', 'C']:
            indices = data[(data.gender == gender) & (data.tier == tier)].index
            if len(indices) >= 2:
                prob += pulp.lpSum(x[i] for i in indices) == 2
            else:
                print(f"ADVERTENCIA: No hay suficientes surfistas para {gender} tier {tier} (solo {len(indices)}). Se relajará la restricción.")
                # Si no hay suficientes, permitimos que elija los que hay (<=)
                prob += pulp.lpSum(x[i] for i in indices) <= len(indices)
    
    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extraer los seleccionados (donde x_i == 1)
    seleccionados = data.loc[[i for i in data.index if x[i].value() == 1]]
    return seleccionados

def main():
    # Obtener datos a través del scraper de tu amigo
    # Esto automáticamente detecta la próxima ola y trae rankings + histórico
    print("Obteniendo datos de surfistas (rankings + histórico)...")
    df = get_surfers_dataframe()
    
    if df.empty:
        print("Error: No se pudieron cargar los datos. Verifica tu conexión o los mocks.")
        return
    
    # Sugerir Power Surfers basados en los mejores total_score
    power_man, power_woman = suggest_power_surfers(df)
    print(f"Power Surfers sugeridos: {power_man} (hombre), {power_woman} (mujer)")
    
    # Optimizar el equipo
    equipo = optimize_team(df, power_man, power_woman)
    
    # Mostrar resultados
    print("\n=== EQUIPO RECOMENDADO ===")
    for genero, nombre_genero in [('M', 'Hombres'), ('F', 'Mujeres')]:
        print(f"\n--- {nombre_genero}")
        for tier in ['A', 'B', 'C']:
            subset = equipo[(equipo.gender == genero) & (equipo.tier == tier)]
            if not subset.empty:
                nombres = ", ".join(subset['surfer'].values)
                # Mostrar también la puntuación de cada uno (opcional)
                puntuaciones = ", ".join([f"{s:.1f}" for s in subset['total_score'].values])
                print(f"  Tier {tier}: {nombres}  (pts: {puntuaciones})")
    
    print("\n=== POWER SURFERS ===")
    print(f"Hombre: {power_man if power_man else 'Ninguno'}")
    print(f"Mujer: {power_woman if power_woman else 'Ninguno'}")
    
    print(f"\nPuntuación total esperada del equipo: {equipo['total_score'].sum():.2f} puntos")
    print("(Incluye el efecto de doble puntuación de Power Surfers si los hay)")

if __name__ == "__main__":
    main()
