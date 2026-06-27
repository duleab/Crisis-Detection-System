import re
import json
import unicodedata
import html
from collections import defaultdict

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from langdetect import detect as _langdetect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


# ── Lexicons & constants ──────────────────────────────────────────────────────
CRISIS_KEYWORDS_ID = [
    'banjir', 'gempa', 'kebakaran', 'kecelakaan', 'bencana',
    'longsor', 'tsunami', 'gunung', 'meletus', 'erupsi',
    'banjir bandang', 'korban', 'evakuasi', 'darurat', 'tolong',
    'minta tolong', 'sos', 'segera', 'rusak', 'hancur', 'tenggelam'
]
CRISIS_KEYWORDS_EN = [
    'flood', 'earthquake', 'fire', 'accident', 'disaster',
    'landslide', 'tsunami', 'eruption', 'volcano', 'evacuation',
    'victim', 'emergency', 'help', 'sos', 'urgent', 'damage', 'collapse'
]
URGENCY_SIGNALS = [
    'tolong', 'help', 'sos', 'darurat', 'emergency', 'segera',
    'urgent', 'bahaya', 'danger', 'kritis', 'critical'
]
HEDGE_WORDS_ID = [
    'katanya', 'mungkin', 'sepertinya', 'dikabarkan', 'konon',
    'kabarnya', 'belum dikonfirmasi', 'diduga', 'kemungkinan'
]
HEDGE_WORDS_EN = [
    'reportedly', 'allegedly', 'possibly', 'unverified', 'rumor',
    'rumour', 'claimed', 'said to', 'appears to', 'seemingly'
]

INDONESIAN_SLANG = {
    'yg': 'yang', 'dgn': 'dengan', 'tdk': 'tidak', 'bgt': 'banget',
    'kbakaran': 'kebakaran', 'gempabumi': 'gempa bumi',
    'gempabumi': 'gempa', 'msh': 'masih', 'sdh': 'sudah',
    'udh': 'sudah', 'lg': 'lagi', 'gk': 'tidak', 'ga': 'tidak',
    'krn': 'karena', 'kpd': 'kepada', 'pd': 'pada', 'jd': 'jadi',
    'hrs': 'harus', 'bs': 'bisa', 'sm': 'sama', 'dr': 'dari',
    'utk': 'untuk', 'ttg': 'tentang', 'bnr': 'benar', 'bkn': 'bukan',
    'spy': 'supaya', 'sdgkan': 'sedangkan', 'krg': 'kurang',
    'jwb': 'jawab', 'ckp': 'cukup', 'nih': 'ini', 'tuh': 'itu'
}


# ── Core text cleaning ────────────────────────────────────────────────────────
def clean_text(text, platform='twitter'):
    """
    Platform-specific text cleaning pipeline.
    Returns cleaned string ready for NLP.
    """
    if not isinstance(text, str) or not text.strip():
        return ''

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Normalize unicode
    text = unicodedata.normalize('NFC', text)

    # 3. Platform-specific cleaning
    if platform == 'twitter':
        # Remove RT prefix
        text = re.sub(r'^RT @\w+:\s*', '', text)
        # Remove @mentions but keep following text
        text = re.sub(r'@\w+', '', text)
        # Expand crisis hashtags by splitting camelcase
        def expand_hashtag(m):
            tag = m.group(1)
            expanded = re.sub(r'([A-Z][a-z]+)', r' \1', tag).strip()
            return expanded if expanded else tag
        text = re.sub(r'#([A-Za-z][a-zA-Z0-9]+)', expand_hashtag, text)
        # Remove remaining non-crisis hashtags
        text = re.sub(r'#\S+', '', text)

    elif platform == 'reddit':
        # Remove markdown formatting
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+\s', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'&gt;[^\n]+', '', text)  # blockquotes

    elif platform == 'telegram':
        # Remove forwarded headers
        text = re.sub(r'Forwarded from [^\n]+\n', '', text)

    # 4. Remove URLs (but keep domain for credibility scoring later)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # 5. Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # 6. Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_indonesian_slang(text, slang_dict=None):
    """Replace Indonesian slang and abbreviations with formal equivalents."""
    if slang_dict is None:
        slang_dict = INDONESIAN_SLANG
    tokens = text.split()
    normalized = [slang_dict.get(t.lower(), t) for t in tokens]
    return ' '.join(normalized)


# ── Language detection ────────────────────────────────────────────────────────
def detect_language(text):
    """Detect language: returns 'id', 'en', 'mixed', or 'other'."""
    if not text or len(text.strip()) < 10:
        return 'other'
    if not _LANGDETECT_AVAILABLE:
        # Simple fallback: check for common Indonesian words
        id_words = ['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'ada', 'dengan']
        if any(w in text.lower().split() for w in id_words):
            return 'id'
        return 'en'
    try:
        lang = _langdetect(text)
        return lang if lang in ('id', 'en') else 'other'
    except Exception:
        return 'other'


# ── Named Entity Recognition ──────────────────────────────────────────────────
_NLP_MODELS = {}

def _load_spacy(lang='xx'):
    """Lazy-load spaCy model."""
    if lang in _NLP_MODELS:
        return _NLP_MODELS[lang]
    if not _SPACY_AVAILABLE:
        return None
    model_map = {
        'en': 'en_core_web_sm',
        'xx': 'xx_ent_wiki_sm',
        'id': 'xx_ent_wiki_sm',
    }
    model_name = model_map.get(lang, 'xx_ent_wiki_sm')
    try:
        nlp = spacy.load(model_name)
        _NLP_MODELS[lang] = nlp
        return nlp
    except OSError:
        print(f"⚠️  spaCy model '{model_name}' not found. Run: "
              f"python -m spacy download {model_name}")
        return None


def run_ner(text, lang='id', gazetteer=None):
    """
    Extract named entities from text.
    Returns dict with lists of locations, times, and organizations.
    """
    result = {'LOC': [], 'DATE': [], 'TIME': [], 'ORG': []}
    if not text:
        return result

    nlp = _load_spacy(lang)
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ('GPE', 'LOC'):
                result['LOC'].append(ent.text)
            elif ent.label_ == 'DATE':
                result['DATE'].append(ent.text)
            elif ent.label_ == 'TIME':
                result['TIME'].append(ent.text)
            elif ent.label_ == 'ORG':
                result['ORG'].append(ent.text)

    # Supplement with gazetteer matching for Indonesian place names
    if gazetteer and text:
        text_lower = text.lower()
        for place in gazetteer:
            if place.lower() in text_lower:
                if place not in result['LOC']:
                    result['LOC'].append(place)

    # Deduplicate
    for key in result:
        result[key] = list(dict.fromkeys(result[key]))

    return result


# ── Crisis Keyword Matching ───────────────────────────────────────────────────
def match_crisis_keywords(text, lexicon_en=None, lexicon_id=None):
    """
    Match crisis keywords against post text.
    Returns (matched_keywords_list, hit_count, urgency_flag).
    """
    if lexicon_en is None:
        lexicon_en = CRISIS_KEYWORDS_EN
    if lexicon_id is None:
        lexicon_id = CRISIS_KEYWORDS_ID

    text_lower = text.lower()
    all_keywords = set(lexicon_en + lexicon_id)

    matched = [kw for kw in all_keywords if kw in text_lower]
    urgency = any(sig in text_lower for sig in URGENCY_SIGNALS)

    return matched, len(matched), urgency


# ── Metadata Feature Computation ──────────────────────────────────────────────
def compute_metadata_features(text, has_media=False, engagement_score=0):
    """Compute per-post metadata features for credibility scoring."""
    contains_number = bool(re.search(r'\d+', text))
    contains_url = bool(re.search(r'https?://|www\.', text))
    url_domain = None
    url_match = re.search(r'https?://([^/\s]+)', text)
    if url_match:
        url_domain = url_match.group(1)

    return {
        'text_length': len(text),
        'token_count': len(text.split()),
        'has_media': has_media,
        'engagement_score': engagement_score,
        'contains_number': contains_number,
        'contains_url': contains_url,
        'url_domain': url_domain,
    }


# ── Linguistic Uncertainty ────────────────────────────────────────────────────
def compute_linguistic_uncertainty(text, hedge_en=None, hedge_id=None):
    """
    Returns a score 0.0–1.0 where 1.0 = highly hedged/uncertain language.
    """
    if hedge_en is None:
        hedge_en = HEDGE_WORDS_EN
    if hedge_id is None:
        hedge_id = HEDGE_WORDS_ID

    all_hedge = hedge_en + hedge_id
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    hedge_count = sum(1 for hw in all_hedge if hw in text.lower())
    return min(1.0, hedge_count / max(len(tokens) * 0.1, 1))


# ── Preliminary Relevance Scoring ─────────────────────────────────────────────
def compute_relevance_flag(keyword_hit_count, entity_count, urgency_flag):
    """Return 1 if post is likely crisis-relevant, else 0."""
    if keyword_hit_count >= 1 and (entity_count >= 1 or urgency_flag):
        return 1
    if keyword_hit_count >= 2:
        return 1
    return 0
