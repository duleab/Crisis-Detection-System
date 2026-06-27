import re
from datetime import datetime, timedelta, timezone

try:
    import dateparser
    _DATEPARSER_AVAILABLE = True
except ImportError:
    _DATEPARSER_AVAILABLE = False

# ── Indonesian temporal patterns ──────────────────────────────────────────────
ID_TEMPORAL_PATTERNS = [
    # Pattern → (offset_hours, offset_minutes)
    (r'tadi pagi',           -6,  0),   # this morning ~6h ago
    (r'pagi tadi',           -6,  0),
    (r'tadi malam',         -12,  0),   # last night
    (r'kemarin malam',      -24,  0),   # last night
    (r'kemarin sore',       -18,  0),   # yesterday afternoon
    (r'kemarin',            -24,  0),   # yesterday
    (r'subuh tadi',          -8,  0),   # earlier at dawn
    (r'malam tadi',         -10,  0),
    (r'tadi sore',           -5,  0),   # this afternoon
    (r'barusan',              0, -5),   # just now
    (r'baru saja',            0, -5),
    (r'baru ini',             0, -5),
    (r'jam (\d+) tadi',       0,  0),   # at X just now (hour extracted)
    (r'sekitar jam (\d+)',    0,  0),   # around X o'clock
]

HOUR_EXTRACTION = re.compile(
    r'jam\s+(\d{1,2})(?:[:\.](\d{2}))?\s*(?:tadi|WIB|WITA|WIT)?',
    re.IGNORECASE
)

WIB_OFFSET = timedelta(hours=7)   # UTC+7


# ── Main parsing function ─────────────────────────────────────────────────────
def parse_temporal_expression(text, reference_timestamp, lang='id'):
    """
    Parse temporal expressions from text.
    Returns (absolute_utc_datetime or None, expression_raw, confidence).
    """
    if not text or not reference_timestamp:
        return None, None, 0.0

    # Ensure reference is timezone-aware UTC
    if isinstance(reference_timestamp, str):
        try:
            reference_dt = datetime.fromisoformat(
                reference_timestamp.replace('Z', '+00:00')
            )
        except Exception:
            reference_dt = datetime.now(timezone.utc)
    else:
        reference_dt = reference_timestamp
    if reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=timezone.utc)

    text_lower = text.lower()

    # 1. Try Indonesian rule-based patterns first
    for pattern, offset_h, offset_m in ID_TEMPORAL_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            delta = timedelta(hours=offset_h, minutes=offset_m)

            # Check if the pattern captures a specific hour
            if r'(\d+)' in pattern and m.lastindex and m.lastindex >= 1:
                try:
                    hour = int(m.group(1))
                    event_dt = reference_dt.replace(
                        hour=hour, minute=0, second=0, microsecond=0
                    )
                    if event_dt > reference_dt:  # If in the future, must be yesterday
                        event_dt -= timedelta(days=1)
                    return event_dt.astimezone(timezone.utc), pattern, 0.85
                except (ValueError, AttributeError):
                    pass

            event_dt = reference_dt + delta
            confidence = 0.50 if offset_h < 0 else 0.65
            return event_dt.astimezone(timezone.utc), pattern, confidence

    # 2. Try extracting explicit HH:MM patterns
    time_match = HOUR_EXTRACTION.search(text)
    if time_match:
        try:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            event_dt = reference_dt.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            # Convert WIB to UTC
            event_dt = event_dt - WIB_OFFSET + timedelta(hours=0)
            if event_dt > reference_dt:
                event_dt -= timedelta(days=1)
            return event_dt.astimezone(timezone.utc), time_match.group(0), 1.0
        except (ValueError, AttributeError):
            pass

    # 3. Fall back to dateparser
    if _DATEPARSER_AVAILABLE:
        settings = {
            'LANGUAGES': ['id', 'en'],
            'RETURN_AS_TIMEZONE_AWARE': True,
            'PREFER_DAY_OF_MONTH': 'first',
            'RELATIVE_BASE': reference_dt,
        }
        parsed = dateparser.parse(text, settings=settings)
        if parsed:
            return parsed.astimezone(timezone.utc), text[:50], 0.65

    return None, None, 0.0


def resolve_relative_time(expression, reference_dt):
    """Resolve a relative time expression to absolute UTC datetime."""
    result, _, confidence = parse_temporal_expression(expression, reference_dt)
    return result, confidence


def to_utc(dt, source_tz_offset_hours=7):
    """Convert a naive datetime (assumed WIB UTC+7) to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    # Assume WIB (UTC+7)
    wib = timezone(timedelta(hours=source_tz_offset_hours))
    return dt.replace(tzinfo=wib).astimezone(timezone.utc)


def to_wib(dt):
    """Convert UTC datetime to WIB (UTC+7)."""
    if dt is None:
        return None
    wib = timezone(timedelta(hours=7))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(wib)


# ── Confidence scoring ────────────────────────────────────────────────────────
TIME_CONFIDENCE = {
    'explicit_hhmm':  1.00,
    'explicit_hh':    0.85,
    'explicit_date':  0.65,
    'relative':       0.50,
    'proxy':          0.30,
}

def compute_time_confidence(expression_type):
    """Return time confidence score for a resolution method."""
    return TIME_CONFIDENCE.get(expression_type, 0.30)


# ── Temporal proximity ────────────────────────────────────────────────────────
def temporal_proximity(ts1, ts2, window_hours=4):
    """
    Returns a score 0.0–1.0 representing how temporally close two events are.
    Score = 1.0 if same time, 0.0 if beyond window_hours apart.
    """
    if ts1 is None or ts2 is None:
        return 0.5  # Unknown — neutral

    if isinstance(ts1, str):
        ts1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
    if isinstance(ts2, str):
        ts2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))

    diff_hours = abs((ts1 - ts2).total_seconds()) / 3600
    if diff_hours >= window_hours:
        return 0.0
    return 1.0 - (diff_hours / window_hours)
