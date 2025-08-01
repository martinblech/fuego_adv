import sys
import datetime
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import ee

PROJECT_ID = "fuego-adv"

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)


geolocator = Nominatim(user_agent=f"{PROJECT_ID}-geocoder", timeout=10)
geolocator.geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

BA_LAT_LON = (-34.6174444, -58.4383458)


def geocode(place: dict) -> tuple[float, float]:
    """Geocode a place."""
    loc = geolocator.geocode(
        f"{place['main_city']}, {place['jurisdiction']}, Argentina"
    )
    if loc is None:
        print(
            f"Failed to geocode {place['main_city']}, {place['jurisdiction']}",
            file=sys.stderr,
        )
        return (None, None)
    return (loc.latitude, loc.longitude)


def get_places(csv_path) -> list[dict]:
    """Load places from local CSV file."""
    print(f"Loading places from {csv_path}...", file=sys.stderr)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows:
            row["population"] = int(row["population"])
        print(f"Loaded {len(rows)} places", file=sys.stderr)
        return sorted(rows, key=lambda x: x["population"], reverse=True)


def get_precipitation(
    *,
    month: int,
    latitude: float,
    longitude: float,
    radius_km: float,
    start_year: int = 2015,
) -> float:
    """Get cumulative precipitation for a given month across all years from start_year to current year."""
    # Create a point geometry for the location
    point = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the point (radius in meters)
    buffer = point.buffer(radius_km * 1000)

    # Load CHIRPS precipitation dataset
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

    # Get current year for end date
    current_year = datetime.date.today().year

    # Create date range for the specific month across all years
    month_start = datetime.date(start_year, month, 1)

    # Handle edge case for December
    if month == 12:
        month_end = datetime.date(current_year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        month_end = datetime.date(current_year, month + 1, 1) - datetime.timedelta(
            days=1
        )

    # Filter the collection by date range (all years for the specified month)
    filtered_chirps = chirps.filterDate(
        month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
    )

    # Calculate mean precipitation across all years for the month
    monthly_precip = filtered_chirps.mean()

    # Extract the mean precipitation value for the buffer area
    stats = monthly_precip.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=5000,  # 5km resolution
        maxPixels=1e6,
    )

    # Get the precipitation value (in mm)
    precip_mm = stats.get("precipitation").getInfo()

    if precip_mm is None:
        print(
            f"WARNING: No precipitation data available for month {month} at coordinates ({latitude}, {longitude})",
            file=sys.stderr,
        )
        return None

    return float(precip_mm)


def get_rainy_days(
    *,
    month: int,
    latitude: float,
    longitude: float,
    radius_km: float,
    start_year: int = 2015,
    threshold_mm: float = 2.0,
) -> float:
    """Get the proportion of rainy days for a given month across all years from start_year to current year.

    Args:
        month: Month (1-12)
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        radius_km: Radius in kilometers for the buffer area
        start_year: Start year for the analysis period
        threshold_mm: Precipitation threshold in mm to consider a day as "rainy"

    Returns:
        Proportion of rainy days (0.0 to 1.0) for the specified month across all years.
        For example, 0.25 means 25% of days in this month were rainy across all years.
    """
    # Create a point geometry for the location
    point = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the point (radius in meters)
    buffer = point.buffer(radius_km * 1000)

    # Load CHIRPS precipitation dataset
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

    # Get current year for end date
    current_year = datetime.date.today().year

    # Create date range for the specific month across all years
    month_start = datetime.date(start_year, month, 1)

    # Handle edge case for December
    if month == 12:
        month_end = datetime.date(current_year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        month_end = datetime.date(current_year, month + 1, 1) - datetime.timedelta(
            days=1
        )

    # Filter the collection by date range (all years for the specified month)
    filtered_chirps = chirps.filterDate(
        month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
    )

    # Create a binary mask for days with precipitation above threshold
    rainy_days_mask = filtered_chirps.map(
        lambda image: image.gt(threshold_mm).rename("rainy_day")
    )

    # Calculate the mean rainy days (this gives us the proportion across all years)
    avg_rainy_days = rainy_days_mask.mean()

    # Extract the proportion of rainy days for the buffer area
    stats = avg_rainy_days.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=5000,  # 5km resolution
        maxPixels=1e6,
    )

    # Get the rainy days proportion
    rainy_days_proportion = stats.get("rainy_day").getInfo()

    # Return the proportion, or raise ValueError if no data
    if rainy_days_proportion is None:
        print(
            f"WARNING: No rainy days data available for month {month} at coordinates ({latitude}, {longitude})",
            file=sys.stderr,
        )
        return None

    return float(rainy_days_proportion)


def get_temperatures(
    *,
    month: int,
    latitude: float,
    longitude: float,
    radius_km: float,
    start_year: int = 2015,
) -> dict:
    """Get min, avg, and max temperatures for a given month across all years from start_year to current year.

    Args:
        month: Month (1-12)
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        radius_km: Radius in kilometers for the buffer area
        start_year: Start year for the analysis period

    Returns:
        Dictionary with 'min_temp_c', 'avg_temp_c', and 'max_temp_c' keys
    """
    # Create a point geometry for the location
    point = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the point (radius in meters)
    buffer = point.buffer(radius_km * 1000)

    # Load ERA5 temperature dataset
    era5 = ee.ImageCollection("ECMWF/ERA5/DAILY")

    # Get current year for end date
    current_year = datetime.date.today().year

    # Create date range for the specific month across all years
    month_start = datetime.date(start_year, month, 1)

    # Handle edge case for December
    if month == 12:
        month_end = datetime.date(current_year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        month_end = datetime.date(current_year, month + 1, 1) - datetime.timedelta(
            days=1
        )

    # Filter the collection by date range (all data for the specified month)
    filtered_era5 = era5.filterDate(
        month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
    )

    # Get temperature bands (2m temperature in Kelvin)
    temp_2m = filtered_era5.select("mean_2m_air_temperature")

    # Convert Kelvin to Celsius
    temp_celsius = temp_2m.map(lambda img: img.subtract(273.15).rename("temp_c"))

    # Calculate percentiles across all years for the month
    percentiles = temp_celsius.reduce(reducer=ee.Reducer.percentile([5, 50, 95]))

    # Extract temperature values for the buffer area
    stats = percentiles.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=25000,  # 25km resolution for ERA5
        maxPixels=1e6,
    )

    # Get temperature values
    p5_temp_c = stats.get("temp_c_p5").getInfo()
    p50_temp_c = stats.get("temp_c_p50").getInfo()
    p95_temp_c = stats.get("temp_c_p95").getInfo()

    # Check if data is available
    if any(temp is None for temp in [p5_temp_c, p50_temp_c, p95_temp_c]):
        print(
            f"WARNING: No temperature data available for month {month} at coordinates ({latitude}, {longitude})",
            file=sys.stderr,
        )

    return {
        "p5_temp_c": float(p5_temp_c) if p5_temp_c is not None else None,
        "p50_temp_c": float(p50_temp_c) if p50_temp_c is not None else None,
        "p95_temp_c": float(p95_temp_c) if p95_temp_c is not None else None,
    }


def get_wind_data(
    *,
    month: int,
    latitude: float,
    longitude: float,
    radius_km: float,
    start_year: int = 2015,
    windy_threshold_ms: float = 5.5,
    very_windy_threshold_ms: float = 8.3,
    wind_gust_threshold_ms: float = 10.8,
) -> dict:
    """Get wind speed statistics and windy days for a given month across all years.

    Args:
        month: Month (1-12)
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        radius_km: Radius in kilometers for the buffer area
        start_year: Start year for the analysis period

    Returns:
        Dictionary with wind statistics including:
        - avg_wind_speed_ms: Average wind speed in m/s
        - p95_wind_speed_ms: 95th percentile wind speed in m/s
        - windy_days_rate: Proportion of days with wind speed > 5.5 m/s (20 km/h)
        - very_windy_days_rate: Proportion of days with wind speed > 8.3 m/s (30 km/h)
        - wind_gust_days_rate: Proportion of days with wind gusts > 10.8 m/s (39 km/h)
    """
    # Create a point geometry for the location
    point = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the point (radius in meters)
    buffer = point.buffer(radius_km * 1000)

    # Load ERA5 wind dataset
    era5 = ee.ImageCollection("ECMWF/ERA5/DAILY")

    # Get current year for end date
    current_year = datetime.date.today().year

    # Create date range for the specific month across all years
    month_start = datetime.date(start_year, month, 1)

    # Handle edge case for December
    if month == 12:
        month_end = datetime.date(current_year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        month_end = datetime.date(current_year, month + 1, 1) - datetime.timedelta(
            days=1
        )

    # Filter the collection by date range (all data for the specified month)
    filtered_era5 = era5.filterDate(
        month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
    )

    # Calculate wind speed from u and v components: sqrt(u² + v²)
    # Join the collections and map the calculation
    wind_speed = filtered_era5.map(
        lambda img: img.select("u_component_of_wind_10m")
        .pow(2)
        .add(img.select("v_component_of_wind_10m").pow(2))
        .sqrt()
        .rename("wind_speed")
    )

    # Calculate wind speed statistics across all years for the month
    wind_stats = wind_speed.reduce(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.percentile([95]), sharedInputs=True
        )
    )

    # Extract wind speed values for the buffer area
    stats = wind_stats.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=25000,  # 25km resolution for ERA5
        maxPixels=1e6,
    )

    # Get wind speed values
    avg_wind_speed_ms = stats.get("wind_speed_mean").getInfo()
    p95_wind_speed_ms = stats.get("wind_speed_p95").getInfo()

    # Create masks for different wind speed thresholds
    # Windy: > threshold m/s - can affect outdoor activities
    windy_mask = wind_speed.map(
        lambda img: img.gt(windy_threshold_ms).rename("windy_day")
    )

    # Very windy: > threshold m/s - can be hazardous for outdoor events
    very_windy_mask = wind_speed.map(
        lambda img: img.gt(very_windy_threshold_ms).rename("very_windy_day")
    )

    # Wind gusts: > threshold m/s - significant hazard for outdoor events
    wind_gust_mask = wind_speed.map(
        lambda img: img.gt(wind_gust_threshold_ms).rename("wind_gust_day")
    )

    # Calculate proportions of windy days
    windy_days = windy_mask.mean()
    very_windy_days = very_windy_mask.mean()
    wind_gust_days = wind_gust_mask.mean()

    # Extract proportions for the buffer area
    windy_stats = windy_days.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=25000,
        maxPixels=1e6,
    )

    very_windy_stats = very_windy_days.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=25000,
        maxPixels=1e6,
    )

    wind_gust_stats = wind_gust_days.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=25000,
        maxPixels=1e6,
    )

    # Get proportions
    windy_days_rate = windy_stats.get("windy_day").getInfo()
    very_windy_days_rate = very_windy_stats.get("very_windy_day").getInfo()
    wind_gust_days_rate = wind_gust_stats.get("wind_gust_day").getInfo()

    # Check if data is available
    if any(
        val is None
        for val in [
            avg_wind_speed_ms,
            p95_wind_speed_ms,
            windy_days_rate,
            very_windy_days_rate,
            wind_gust_days_rate,
        ]
    ):
        print(
            f"WARNING: No wind data available for month {month} at coordinates ({latitude}, {longitude})",
            file=sys.stderr,
        )

    return {
        "avg_wind_speed_ms": float(avg_wind_speed_ms)
        if avg_wind_speed_ms is not None
        else None,
        "p95_wind_speed_ms": float(p95_wind_speed_ms)
        if p95_wind_speed_ms is not None
        else None,
        "windy_days_rate": float(windy_days_rate)
        if windy_days_rate is not None
        else None,
        "very_windy_days_rate": float(very_windy_days_rate)
        if very_windy_days_rate is not None
        else None,
        "wind_gust_days_rate": float(wind_gust_days_rate)
        if wind_gust_days_rate is not None
        else None,
    }


def process_places(places: list, args: argparse.Namespace) -> list:
    """Process the places."""

    def process_place(place: dict):
        place["latitude"], place["longitude"] = geocode(place)
        place["ba_distance"] = geodesic(
            BA_LAT_LON, (place["latitude"], place["longitude"])
        ).km
        place["feb_precip_mm"] = get_precipitation(
            month=2,
            latitude=place["latitude"],
            longitude=place["longitude"],
            radius_km=args.weather_radius_km,
            start_year=args.start_year,
        )
        place["feb_rainy_days_rate"] = get_rainy_days(
            month=2,
            latitude=place["latitude"],
            longitude=place["longitude"],
            radius_km=args.weather_radius_km,
            start_year=args.start_year,
            threshold_mm=args.rainy_day_threshold_mm,
        )
        place["feb_very_rainy_days_rate"] = get_rainy_days(
            month=2,
            latitude=place["latitude"],
            longitude=place["longitude"],
            radius_km=args.weather_radius_km,
            start_year=args.start_year,
            threshold_mm=args.very_rainy_day_threshold_mm,
        )

        # Get February temperatures (all at once to avoid multiple API calls)
        feb_temps = get_temperatures(
            month=2,
            latitude=place["latitude"],
            longitude=place["longitude"],
            radius_km=args.weather_radius_km,
            start_year=args.start_year,
        )
        place["feb_p5_temp_c"] = feb_temps["p5_temp_c"]
        place["feb_p50_temp_c"] = feb_temps["p50_temp_c"]
        place["feb_p95_temp_c"] = feb_temps["p95_temp_c"]

        # Get wind data (all at once to avoid multiple API calls)
        feb_wind_data = get_wind_data(
            month=2,
            latitude=place["latitude"],
            longitude=place["longitude"],
            radius_km=args.weather_radius_km,
            start_year=args.start_year,
            windy_threshold_ms=args.windy_day_threshold_ms,
            very_windy_threshold_ms=args.very_windy_day_threshold_ms,
            wind_gust_threshold_ms=args.wind_gust_threshold_ms,
        )
        place["feb_avg_wind_speed_ms"] = feb_wind_data["avg_wind_speed_ms"]
        place["feb_p95_wind_speed_ms"] = feb_wind_data["p95_wind_speed_ms"]
        place["feb_windy_days_rate"] = feb_wind_data["windy_days_rate"]
        place["feb_very_windy_days_rate"] = feb_wind_data["very_windy_days_rate"]
        place["feb_wind_gust_days_rate"] = feb_wind_data["wind_gust_days_rate"]

        print(f"Processed {place}")
        return place

    p = ThreadPoolExecutor(max_workers=10)
    places = list(p.map(process_place, places))

    # Filter out places with population outside the specified range
    filtered_places = [place for place in places if place is not None]

    print(
        f"Filtered {len(places)} to {len(filtered_places)} places",
        file=sys.stderr,
    )
    return filtered_places


def export_places(places: list[dict]) -> None:
    """Export the places to CSV format on stdout."""
    writer = csv.DictWriter(sys.stdout, places[0].keys())
    writer.writeheader()
    for place in places:
        writer.writerow(place)


def main():
    parser = argparse.ArgumentParser(
        description="Augment places CSV with geo and weather data"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/places.csv",
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Start year for weather data analysis (default: 2010)",
    )
    parser.add_argument(
        "--weather-radius-km",
        type=float,
        default=50,
        help="Weather radius in km (default: 50)",
    )
    parser.add_argument(
        "--rainy-day-threshold-mm",
        type=float,
        default=1,
        help="Rainy day threshold in mm (default: 1)",
    )
    parser.add_argument(
        "--very-rainy-day-threshold-mm",
        type=float,
        default=5,
        help="Very rainy day threshold in mm (default: 5)",
    )
    parser.add_argument(
        "--windy-day-threshold-ms",
        type=float,
        default=5.5,
        help="Windy day threshold in m/s (default: 5.5, ~20 km/h)",
    )
    parser.add_argument(
        "--very-windy-day-threshold-ms",
        type=float,
        default=8.3,
        help="Very windy day threshold in m/s (default: 8.3, ~30 km/h)",
    )
    parser.add_argument(
        "--wind-gust-threshold-ms",
        type=float,
        default=10.8,
        help="Wind gust threshold in m/s (default: 10.8, ~39 km/h)",
    )

    args = parser.parse_args()

    places = get_places(args.csv_path)
    processed_places = process_places(places, args)
    export_places(processed_places)


if __name__ == "__main__":
    main()
