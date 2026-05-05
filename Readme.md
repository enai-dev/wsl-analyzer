# WSL Team Optimizer

> El asistente inteligente para ganar en el Fantasy de la World Surf League

 # ¿Qué es esto?

Si eres fan del surf y juegas al **Fantasy de la WSL**, sabrás que elegir el equipo no es fácil: hay reglas de niveles (Tiers), surfistas que rinden mejor en según qué ola, rachas cambiantes, y además ahora también hay que elegir **2 surfistas por cada Tier** tanto en hombres como en mujeres.

Este programa te quita el dolor de cabeza. Analiza datos reales (resultados históricos, características de la ola, estado de forma) y te dice **exactamente qué 12 surfistas elegir** para maximizar tus puntos.

No es magia, es **datos + optimización**. 

# ¿Cómo funciona por dentro?

El programa hace tres cosas sencillas pero muy potentes:

1. **Recolecta datos** → Rastrea páginas oficiales (WSL, rankings, etc.) y construye un histórico de puntuaciones por surfista, ola y evento.
2. **Predice** → Para cada surfista y evento, calcula una **puntuación esperada** mezclando:
   - Rendimiento histórico en esa ola (el factor más importante)
   - Racha de los últimos eventos
   - Características de la ola (tubos, aéreos, izquierda/derecha)
   - Ventaja de ser local (si aplica)
3. **Optimiza** → Con las reglas del Fantasy (2 surfistas por cada Tier A, B, C en hombres y mujeres, más los 2 Power Surfers), prueba millones de combinaciones y se queda con el equipo que suma **más puntos esperados totales**.

El resultado es una alineación legal y matemáticamente óptima.

## 📅 Adaptado a la temporada 2026

La WSL ha cambiado las reglas: ahora hay **2 picks por cada Tier** en ambas categorías. Es decir:

| Categoría | Tier A | Tier B | Tier C | Total |
|-----------|--------|--------|--------|-------|
| Hombres   | 2      | 2      | 2      | 6     |
| Mujeres   | 2      | 2      | 2      | 6     |
| **Total** | 4      | 4      | 4      | 12    |

El programa ya tiene integrada esta restricción. También respeta la selección automática de **Power Surfers** (un hombre y una mujer) que duplican sus puntos.

## 🛠️ Tecnologías usadas

- **Python 3.9+** → Lenguaje principal.
- **BeautifulSoup + Requests** → Web scraping.
- **Pandas** → Limpieza y manipulación de datos.
- **Scikit-learn / XGBoost** → Modelo predictivo (opcional, según la versión).
- **PuLP / OR-Tools** → Optimizador de equipos (resuelve restricciones de Tiers).
- **Git + GitHub** → Control de versiones y colaboración.

