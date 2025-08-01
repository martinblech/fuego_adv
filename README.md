# Fuego Austral "¿A Dónde Vamos?" dataset

Base places CSV extracted from https://es.wikipedia.org/wiki/Anexo:Áreas_urbanas_de_Argentina_con_más_de_30_000_habitantes

Augment with geo data (lat/lon, distance to BA) and weather data (rain, wind, temp).

## Generated Columns

The script adds the following columns to each place:

**Geographic Data:**
- `latitude`, `longitude` - Geocoded coordinates
- `ba_distance` - Distance to Buenos Aires in km

**February Weather Data (based on historical data since start year):**
- `feb_precip_mm` - Average precipitation in mm
- `feb_rainy_days_rate` - Proportion of rainy days (>threshold)
- `feb_very_rainy_days_rate` - Proportion of very rainy days (>higher threshold)
- `feb_p5_temp_c`, `feb_p50_temp_c`, `feb_p95_temp_c` - Temperature percentiles in Celsius
- `feb_avg_wind_speed_ms`, `feb_p95_wind_speed_ms` - Wind speed statistics in m/s
- `feb_windy_days_rate`, `feb_very_windy_days_rate`, `feb_wind_gust_days_rate` - Wind-related metrics

## Usage:

`$ uv run geo_weather_csv.py`

(pass --help to see options, but defaults should be fine)