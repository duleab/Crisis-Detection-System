import math
import json
import time
import re
import requests
from functools import lru_cache

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    _GEOPY_AVAILABLE = True
except ImportError:
    _GEOPY_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    _GEOPANDAS_AVAILABLE = True
except ImportError:
    _GEOPANDAS_AVAILABLE = False


# ── Haversine distance ────────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Gazetteer ─────────────────────────────────────────────────────────────────
def load_gazetteer(path):
    """
    Load place-name gazetteer from CSV.
    Expected columns: name, alt_names, lat, lon, level (province/district/city)
    Returns dict: {canonical_name: {lat, lon, level, alt_names}}.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
        gazetteer = {}
        for _, row in df.iterrows():
            name = str(row.get('name', '')).strip().lower()
            gazetteer[name] = {
                'lat': float(row.get('lat', 0)),
                'lon': float(row.get('lon', 0)),
                'level': str(row.get('level', 'unknown')),
                'canonical': str(row.get('name', '')),
            }
            # Also index alt_names
            for alt in str(row.get('alt_names', '')).split(';'):
                alt = alt.strip().lower()
                if alt:
                    gazetteer[alt] = gazetteer[name]
        print(f"✅ Loaded gazetteer with {len(gazetteer)} place entries.")
        return gazetteer
    except Exception as e:
        print(f"⚠️  Could not load gazetteer from {path}: {e}")
        # Return a minimal built-in gazetteer for Indonesian cities
        return _builtin_indonesia_gazetteer()


def _builtin_indonesia_gazetteer():
    """Minimal built-in gazetteer for major Indonesian cities."""
    places = {
        'jakarta': {'lat': -6.2088, 'lon': 106.8456, 'level': 'city', 'canonical': 'Jakarta'},
        'surabaya': {'lat': -7.2575, 'lon': 112.7521, 'level': 'city', 'canonical': 'Surabaya'},
        'bandung': {'lat': -6.9175, 'lon': 107.6191, 'level': 'city', 'canonical': 'Bandung'},
        'medan': {'lat': 3.5952, 'lon': 98.6722, 'level': 'city', 'canonical': 'Medan'},
        'semarang': {'lat': -6.9932, 'lon': 110.4203, 'level': 'city', 'canonical': 'Semarang'},
        'makassar': {'lat': -5.1477, 'lon': 119.4327, 'level': 'city', 'canonical': 'Makassar'},
        'palembang': {'lat': -2.9761, 'lon': 104.7754, 'level': 'city', 'canonical': 'Palembang'},
        'tangerang': {'lat': -6.1781, 'lon': 106.6298, 'level': 'city', 'canonical': 'Tangerang'},
        'depok': {'lat': -6.4025, 'lon': 106.7942, 'level': 'city', 'canonical': 'Depok'},
        'bekasi': {'lat': -6.2383, 'lon': 106.9756, 'level': 'city', 'canonical': 'Bekasi'},
        'bogor': {'lat': -6.5971, 'lon': 106.8060, 'level': 'city', 'canonical': 'Bogor'},
        'yogyakarta': {'lat': -7.7956, 'lon': 110.3695, 'level': 'city', 'canonical': 'Yogyakarta'},
        'solo': {'lat': -7.5755, 'lon': 110.8243, 'level': 'city', 'canonical': 'Solo'},
        'malang': {'lat': -7.9797, 'lon': 112.6304, 'level': 'city', 'canonical': 'Malang'},
        'pekanbaru': {'lat': 0.5335, 'lon': 101.4474, 'level': 'city', 'canonical': 'Pekanbaru'},
        'banjarmasin': {'lat': -3.3186, 'lon': 114.5944, 'level': 'city', 'canonical': 'Banjarmasin'},
        'pontianak': {'lat': -0.0263, 'lon': 109.3425, 'level': 'city', 'canonical': 'Pontianak'},
        'bali': {'lat': -8.4095, 'lon': 115.1889, 'level': 'province', 'canonical': 'Bali'},
        'denpasar': {'lat': -8.6705, 'lon': 115.2126, 'level': 'city', 'canonical': 'Denpasar'},
        'aceh': {'lat': 4.6951, 'lon': 96.7494, 'level': 'province', 'canonical': 'Aceh'},
        'cianjur': {'lat': -6.8203, 'lon': 107.1366, 'level': 'district', 'canonical': 'Cianjur'},
        'ciawi': {'lat': -6.6789, 'lon': 106.8890, 'level': 'subdistrict', 'canonical': 'Ciawi'},
        'kemang': {'lat': -6.2607, 'lon': 106.8138, 'level': 'area', 'canonical': 'Kemang'},
        # Province abbreviations
        'jkt': {'lat': -6.2088, 'lon': 106.8456, 'level': 'city', 'canonical': 'Jakarta'},
        'bdg': {'lat': -6.9175, 'lon': 107.6191, 'level': 'city', 'canonical': 'Bandung'},
        'sby': {'lat': -7.2575, 'lon': 112.7521, 'level': 'city', 'canonical': 'Surabaya'},
        'jogja': {'lat': -7.7956, 'lon': 110.3695, 'level': 'city', 'canonical': 'Yogyakarta'},
        'jawa barat': {'lat': -6.9147, 'lon': 107.6098, 'level': 'province', 'canonical': 'West Java'},
        'jawa tengah': {'lat': -7.0051, 'lon': 110.4381, 'level': 'province', 'canonical': 'Central Java'},
        'jawa timur': {'lat': -7.5361, 'lon': 112.2384, 'level': 'province', 'canonical': 'East Java'},
        'sulawesi selatan': {'lat': -3.6687, 'lon': 119.9741, 'level': 'province', 'canonical': 'South Sulawesi'},
        'kalimantan barat': {'lat': 0.1325, 'lon': 111.0937, 'level': 'province', 'canonical': 'West Kalimantan'},
        'sumatera utara': {'lat': 2.1154, 'lon': 99.5451, 'level': 'province', 'canonical': 'North Sumatra'},
    }
    return places


def fuzzy_match_gazetteer(location_string, gazetteer, threshold=2):
    """
    Fuzzy match a location string against the gazetteer.
    Uses Levenshtein distance. Returns match dict or None.
    """
    if not location_string or not gazetteer:
        return None

    query = location_string.lower().strip()

    # Exact match first
    if query in gazetteer:
        return gazetteer[query]

    # Fuzzy match using simple edit distance
    best_match = None
    best_dist = threshold + 1

    for key in gazetteer:
        dist = _levenshtein(query, key)
        if dist < best_dist and dist <= threshold:
            best_dist = dist
            best_match = gazetteer[key]

    return best_match


def _levenshtein(s1, s2):
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for c1 in s1:
        curr = [prev[0] + 1]
        for i, c2 in enumerate(s2):
            curr.append(min(prev[i + 1] + 1, curr[i] + 1,
                            prev[i] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ── Nominatim Geocoding ───────────────────────────────────────────────────────
_geocoder = None
_geocode_cache = {}

def _get_geocoder(email='crisis_detection@research.ac.id'):
    global _geocoder
    if _geocoder is None and _GEOPY_AVAILABLE:
        _geocoder = Nominatim(user_agent=f"crisis_detection/{email}")
    return _geocoder


def geocode_string(location_string, country_code='ID', email=None):
    """
    Geocode a location string using Nominatim.
    Returns (lat, lon) or (None, None).
    Implements 1-second rate limiting and local caching.
    """
    if not location_string:
        return None, None

    cache_key = f"{location_string.lower()}_{country_code}"
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    if not _GEOPY_AVAILABLE:
        return None, None

    geocoder = _get_geocoder(email or 'research@example.com')
    if geocoder is None:
        return None, None

    try:
        time.sleep(1.1)  # Nominatim rate limit: 1 req/sec
        result = geocoder.geocode(
            f"{location_string}, Indonesia",
            country_codes=country_code.lower(),
            timeout=10
        )
        if result:
            lat, lon = result.latitude, result.longitude
            _geocode_cache[cache_key] = (lat, lon)
            return lat, lon
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"  Geocoding error for '{location_string}': {e}")
    except Exception as e:
        print(f"  Geocoding unexpected error: {e}")

    _geocode_cache[cache_key] = (None, None)
    return None, None


def save_geocode_cache(cache_path):
    """Save geocode cache to JSON for future runs."""
    with open(cache_path, 'w') as f:
        json.dump(_geocode_cache, f, ensure_ascii=False, indent=2)


def load_geocode_cache(cache_path):
    """Load geocode cache from JSON."""
    global _geocode_cache
    try:
        with open(cache_path) as f:
            _geocode_cache = json.load(f)
        print(f"✅ Loaded geocode cache with {len(_geocode_cache)} entries.")
    except FileNotFoundError:
        pass


# ── Geo-confidence scoring ────────────────────────────────────────────────────
GEO_TIER_SCORES = {
    1: 1.00,   # GPS metadata
    2: 0.85,   # Gazetteer district-level
    3: 0.60,   # Gazetteer province-level
    4: 0.50,   # Hashtag expansion
    5: 0.35,   # User profile location
    6: 0.40,   # Organization jurisdiction
    7: 0.25,   # Cluster centroid inheritance
    0: 0.00,   # Unresolved
}

LEVEL_BONUS = {
    'street': 0.10,
    'area': 0.07,
    'subdistrict': 0.07,
    'district': 0.05,
    'city': 0.03,
    'province': 0.00,
    'country': -0.05,
    'unknown': 0.00,
}

def compute_geo_confidence(tier, level='unknown'):
    """Compute geo-confidence score (0.0–1.0) from resolution tier and level."""
    base = GEO_TIER_SCORES.get(tier, 0.0)
    bonus = LEVEL_BONUS.get(level, 0.0)
    return max(0.0, min(1.0, base + bonus))


# ── Point-in-polygon ──────────────────────────────────────────────────────────
def point_in_polygon(lat, lon, gdf):
    """Return the first polygon row in gdf that contains the point."""
    if not _GEOPANDAS_AVAILABLE or gdf is None:
        return None
    point = Point(lon, lat)
    matches = gdf[gdf.geometry.contains(point)]
    if not matches.empty:
        return matches.iloc[0]
    return None
