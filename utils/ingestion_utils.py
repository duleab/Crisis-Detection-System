import pandas as pd
import requests
import feedparser
import tweepy
import time
import json
import re
from datetime import datetime, timezone
from tqdm import tqdm


# ── Twitter / X ──────────────────────────────────────────────────────────────
def fetch_tweets(keywords, start_date=None, end_date=None, bbox=None,
                 bearer_token=None, max_results=100):
    """Fetch tweets using Tweepy v2 recent-search endpoint."""
    if not bearer_token or bearer_token == "YOUR_TWITTER_BEARER_TOKEN":
        print("⚠️  No Twitter bearer token — returning empty DataFrame.")
        return pd.DataFrame()

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    query_parts = [f'({" OR ".join(keywords)})']
    query_parts.append('lang:id OR lang:en')
    query_parts.append('-is:retweet')
    if bbox:
        # bbox = [west, south, east, north]
        query_parts.append(
            f'bounding_box:[{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}]'
        )
    query = ' '.join(query_parts)

    rows = []
    try:
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            tweet_fields=['created_at', 'author_id', 'geo', 'public_metrics', 'text'],
            place_fields=['full_name', 'geo'],
            expansions=['geo.place_id'],
            max_results=min(max_results, 100),
            limit=max_results // 100 + 1,
            start_time=start_date,
            end_time=end_date,
        )
        for response in paginator:
            if response.data is None:
                continue
            places = {p.id: p for p in (response.includes.get('places') or [])}
            for tweet in response.data:
                place_name = None
                if tweet.geo and tweet.geo.get('place_id'):
                    place = places.get(tweet.geo['place_id'])
                    if place:
                        place_name = place.full_name
                m = tweet.public_metrics or {}
                rows.append({
                    'post_id': f"TW_{tweet.id}",
                    'raw_text': tweet.text,
                    'timestamp_raw': str(tweet.created_at),
                    'user_id': str(tweet.author_id),
                    'location_raw': place_name,
                    'geo_lat': None,
                    'geo_lon': None,
                    'media_present': False,
                    'engagement_score': (
                        m.get('retweet_count', 0) +
                        m.get('like_count', 0) +
                        m.get('reply_count', 0)
                    ),
                    'source_platform': 'twitter',
                    'label': None,
                })
    except Exception as e:
        print(f"Twitter API error: {e}")

    print(f"✅ Fetched {len(rows)} tweets.")
    return pd.DataFrame(rows)


# ── Reddit RSS ────────────────────────────────────────────────────────────────
def fetch_reddit_posts(keywords, limit=100):
    """Fetch Reddit posts via public RSS search (no API key required)."""
    import urllib.parse
    print("Fetching Reddit posts via RSS...")
    query = urllib.parse.quote(" OR ".join(keywords))
    urls = [
        f"https://www.reddit.com/search.rss?q={query}&sort=new&limit=25",
        "https://www.reddit.com/r/indonesia/.rss",
        "https://www.reddit.com/r/worldnews/search.rss?q=Indonesia+disaster",
    ]

    rows = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                summary = getattr(entry, 'summary', '') or ''
                clean_summary = re.sub(r'<[^>]+>', ' ', summary)
                rows.append({
                    'post_id': f"RD_{abs(hash(entry.id))}",
                    'raw_text': (entry.title + ' ' + clean_summary).strip(),
                    'timestamp_raw': entry.get('published', ''),
                    'user_id': None,
                    'location_raw': None,
                    'geo_lat': None,
                    'geo_lon': None,
                    'media_present': False,
                    'engagement_score': 0,
                    'source_platform': 'reddit',
                    'label': None,
                })
        except Exception as e:
            print(f"  Reddit RSS error for {url}: {e}")

    print(f"✅ Fetched {len(rows)} Reddit posts.")
    return pd.DataFrame(rows)


# ── RSS News Feeds ────────────────────────────────────────────────────────────
INDONESIAN_NEWS_FEEDS = [
    "https://www.detik.com/feeds/detik-terkini.rss",
    "https://rss.kompas.com/nasional",
    "https://www.antaranews.com/rss/terkini.xml",
    "https://www.cnnindonesia.com/nasional/rss",
    "https://www.tribunnews.com/rss",
]

def fetch_rss_entries(feed_urls=None, keywords=None, since_date=None):
    """Fetch and filter RSS news entries."""
    if feed_urls is None:
        feed_urls = INDONESIAN_NEWS_FEEDS
    if keywords is None:
        keywords = ['banjir', 'gempa', 'bencana', 'kebakaran', 'longsor',
                    'flood', 'earthquake', 'disaster', 'fire']

    rows = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                text = (entry.title + ' ' + getattr(entry, 'summary', '')).lower()
                if any(kw.lower() in text for kw in keywords):
                    rows.append({
                        'post_id': f"NEWS_{abs(hash(entry.get('id', entry.link)))}",
                        'raw_text': (entry.title + '. ' +
                                     re.sub(r'<[^>]+>', ' ',
                                            getattr(entry, 'summary', ''))).strip(),
                        'timestamp_raw': str(entry.get('published', '')),
                        'user_id': None,
                        'location_raw': None,
                        'geo_lat': None,
                        'geo_lon': None,
                        'media_present': False,
                        'engagement_score': 0,
                        'source_platform': 'news_rss',
                        'label': None,
                    })
        except Exception as e:
            print(f"  RSS error for {url}: {e}")

    print(f"✅ Fetched {len(rows)} news RSS entries.")
    return pd.DataFrame(rows)


# ── BMKG ──────────────────────────────────────────────────────────────────────
def fetch_bmkg_alerts(min_magnitude=4.0):
    """Fetch recent earthquake alerts from the BMKG public API."""
    print("Fetching BMKG earthquake alerts...")
    url = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
    rows = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        eq = data.get('Infogempa', {}).get('gempa', {})
        if eq:
            mag = float(eq.get('Magnitude', 0))
            if mag >= min_magnitude:
                rows.append({
                    'post_id': f"BMKG_{eq.get('DateTime', '').replace(' ', '_')}",
                    'raw_text': (
                        f"Earthquake magnitude {mag} at depth {eq.get('Kedalaman', '')} "
                        f"in {eq.get('Wilayah', '')}. {eq.get('Potensi', '')}"
                    ),
                    'timestamp_raw': eq.get('DateTime', ''),
                    'user_id': 'BMKG_official',
                    'location_raw': eq.get('Wilayah', ''),
                    'geo_lat': float(eq.get('Lintang', '0').replace('°LS', '').replace('°LU', '')),
                    'geo_lon': float(eq.get('Bujur', '0').replace('°BT', '')),
                    'media_present': False,
                    'engagement_score': 100,
                    'source_platform': 'bmkg',
                    'label': 'earthquake',
                })
    except Exception as e:
        print(f"  BMKG API error: {e}")

    # Also try the full list
    try:
        url2 = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"
        resp2 = requests.get(url2, timeout=10)
        resp2.raise_for_status()
        data2 = resp2.json()
        for eq in data2.get('Infogempa', {}).get('gempa', []):
            try:
                mag = float(eq.get('Magnitude', 0))
                if mag >= min_magnitude:
                    rows.append({
                        'post_id': f"BMKG_{eq.get('DateTime', '').replace(' ', '_')}_{len(rows)}",
                        'raw_text': (
                            f"Earthquake magnitude {mag} at depth {eq.get('Kedalaman', '')} "
                            f"in {eq.get('Wilayah', '')}."
                        ),
                        'timestamp_raw': eq.get('DateTime', ''),
                        'user_id': 'BMKG_official',
                        'location_raw': eq.get('Wilayah', ''),
                        'geo_lat': None,
                        'geo_lon': None,
                        'media_present': False,
                        'engagement_score': 100,
                        'source_platform': 'bmkg',
                        'label': 'earthquake',
                    })
            except Exception:
                continue
    except Exception as e:
        print(f"  BMKG terkini error: {e}")

    print(f"✅ Fetched {len(rows)} BMKG alerts.")
    return pd.DataFrame(rows)


# ── PetaBencana ───────────────────────────────────────────────────────────────
def fetch_petabencana_reports(start_date=None, end_date=None):
    """Fetch crowdsourced flood reports from PetaBencana.id (ground truth)."""
    print("Fetching PetaBencana.id flood reports...")
    url = "https://data.petabencana.id/reports"
    params = {'city': 'jakarta', 'format': 'json'}
    rows = []
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        features = data.get('result', {}).get('objects', {}).get('output', {}).get('geometries', [])
        for feat in features:
            props = feat.get('properties', {})
            coords = feat.get('coordinates', [None, None])
            rows.append({
                'post_id': f"PB_{props.get('pkey', len(rows))}",
                'raw_text': props.get('text', 'Flood report'),
                'timestamp_raw': props.get('created_at', ''),
                'user_id': None,
                'location_raw': props.get('tags', {}).get('area_name', ''),
                'geo_lat': coords[1] if len(coords) > 1 else None,
                'geo_lon': coords[0] if len(coords) > 0 else None,
                'media_present': bool(props.get('image_url')),
                'engagement_score': 0,
                'source_platform': 'petabencana',
                'label': 'flood',
            })
    except Exception as e:
        print(f"  PetaBencana error: {e}")

    print(f"✅ Fetched {len(rows)} PetaBencana reports.")
    return pd.DataFrame(rows)


# ── Normalization & Deduplication ─────────────────────────────────────────────
UNIFIED_SCHEMA = [
    'post_id', 'source_platform', 'raw_text', 'timestamp_raw',
    'user_id', 'location_raw', 'geo_lat', 'geo_lon',
    'media_present', 'engagement_score', 'label'
]

def normalize_to_schema(df, source_platform):
    """Ensure DataFrame conforms to the unified schema."""
    if df.empty:
        return pd.DataFrame(columns=UNIFIED_SCHEMA)
    df = df.copy()
    df['source_platform'] = source_platform
    for col in UNIFIED_SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df[UNIFIED_SCHEMA]


def deduplicate_posts(df):
    """Remove exact duplicate texts and near-empty posts."""
    if df.empty:
        return df
    original = len(df)
    df = df.dropna(subset=['raw_text'])
    df = df[df['raw_text'].str.len() >= 15]
    df = df[~df['raw_text'].str.match(r'^(https?://\S+|@\w+|#\w+)$')]
    df = df.drop_duplicates(subset=['raw_text'])
    print(f"🧹 Deduplication: {original} → {len(df)} posts "
          f"({original - len(df)} dropped).")
    return df.reset_index(drop=True)
