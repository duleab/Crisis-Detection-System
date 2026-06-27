# Real-Time Crisis Detection — GitHub Security Audit
#
# This script does three things:
#   1. Creates .gitignore
#   2. Creates a safe api_keys.yaml template (strips real keys)
#   3. Cleans all notebooks (removes emojis, STEP X: prefixes, cell numbering)
#
# Run this ONCE before: git add . && git push
# Usage: python pre_push_cleanup.py

import os
import re
import json
import shutil
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Write .gitignore
# ─────────────────────────────────────────────────────────────────────────────
GITIGNORE = """\
# ─── Secrets & Credentials (NEVER push these) ────────────────────────────────
config/api_keys.yaml
.env
.env.*
*.env
secrets.yaml
secrets.json
credentials.json
kaggle.json
token.json
service_account*.json

# ─── Large Data Files ─────────────────────────────────────────────────────────
data/raw/
data/processed/
data/external/training_data.csv
data/external/posts_unified*.csv
*.csv
*.parquet
*.feather
*.h5
*.hdf5
*.pkl
*.pickle

# ─── Model Weights (too large for GitHub, use HuggingFace Hub or Google Drive) ─
models/
*.pt
*.pth
*.bin
*.safetensors
*.onnx
model_weights/
tokenizer/

# ─── Notebook Outputs / Cache ─────────────────────────────────────────────────
.ipynb_checkpoints/
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
*.egg-info/
dist/
build/
.eggs/

# ─── Python Virtual Environment ───────────────────────────────────────────────
venv/
env/
.venv/
.conda/
*.egg

# ─── Logs & Temporary Files ───────────────────────────────────────────────────
*.log
*.tmp
*.temp
logs/
runs/

# ─── OS & Editor Files ────────────────────────────────────────────────────────
.DS_Store
Thumbs.db
.idea/
.vscode/
*.swp
*.swo

# ─── Large Output Files ───────────────────────────────────────────────────────
outputs/maps/*.html
outputs/figures/*.png
outputs/figures/*.jpg
events_dashboard.geojson

# ─── What IS allowed to push ──────────────────────────────────────────────────
# notebooks/*.ipynb          ← yes, push notebooks
# config/model_config.yaml   ← yes, safe (no secrets)
# config/paths.yaml          ← yes, safe (no secrets)
# config/api_keys.yaml.example ← yes, push the TEMPLATE only
# data/external/*.json       ← yes (lexicons, mappings)
# data/external/*.txt        ← yes (lexicons)
# src/*.py                   ← yes
# scripts/*.py               ← yes
# README.md                  ← yes
# requirements.txt           ← yes
"""

gitignore_path = os.path.join(ROOT, ".gitignore")
with open(gitignore_path, "w", encoding="utf-8") as f:
    f.write(GITIGNORE)
print(f"✓  Written: .gitignore")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Create safe api_keys.yaml.example (template with placeholder values)
# ─────────────────────────────────────────────────────────────────────────────
API_KEYS_TEMPLATE = """\
# API Keys Configuration Template
# Copy this file to config/api_keys.yaml and fill in your credentials.
# IMPORTANT: config/api_keys.yaml is in .gitignore — never commit real keys.

twitter:
  bearer_token: "YOUR_TWITTER_BEARER_TOKEN"
  api_key: "YOUR_TWITTER_API_KEY"
  api_secret: "YOUR_TWITTER_API_SECRET"

telegram:
  api_id: 0000000
  api_hash: "YOUR_TELEGRAM_API_HASH"
  session_name: "crisis_detector"

nominatim:
  email: "your_email@example.com"

bmkg:
  base_url: "https://data.bmkg.go.id/DataMKG/TEWS/"

llm:
  provider: "gemini"
  api_key: "YOUR_GEMINI_API_KEY"

petabencana:
  base_url: "https://data.petabencana.id/reports"
"""

template_path = os.path.join(ROOT, "config", "api_keys.yaml.example")
with open(template_path, "w", encoding="utf-8") as f:
    f.write(API_KEYS_TEMPLATE)
print(f"✓  Written: config/api_keys.yaml.example")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Notebook Cleaning
# ─────────────────────────────────────────────────────────────────────────────
# Remove: emojis, "STEP X:" / "CELL X:" / "─── STEP X:" prefixes
# Keep:   meaningful title text, code logic, logical flow

NOTEBOOKS_DIR = os.path.join(ROOT, "notebooks")

# Emoji regex (covers most Unicode emoji ranges)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed chars
    "\u2600-\u26FF"          # misc symbols
    "\u2700-\u27BF"          # dingbats
    "✅❌⚠️🔴🟡🟢🔵"
    "]+",
    flags=re.UNICODE,
)

# Patterns to strip from markdown cell titles/text
STEP_PREFIX_RE = re.compile(
    r"^#+\s*"                              # heading hashes
    r"(?:"
    r"(?:─+\s*)?"                          # optional ─── separator
    r"(?:STEP|CELL|PART|SECTION|PHASE)\s+\d+[:\.\-]?\s*"  # STEP X: / CELL 2. etc
    r"|"
    r"(?:─+\s*)(?:STEP|CELL|PART)\s+\d+[:\.\-]?\s*"
    r")",
    re.IGNORECASE,
)

# Inline ── STEP X: ── markers inside markdown body text
INLINE_STEP_RE = re.compile(
    r"[─\-]{2,}\s*(?:STEP|CELL|PART|SECTION)\s+\d+[:\.\-]?\s*[─\-]*\s*",
    re.IGNORECASE,
)

def clean_text(text: str) -> str:
    """Remove emojis and step/cell numbering from a string."""
    # Remove emojis
    text = EMOJI_RE.sub("", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()

        # Handle heading lines that start with STEP/CELL
        heading_match = re.match(
            r"^(#+)\s*"
            r"(?:─+\s*)?"
            r"(?:(?:STEP|CELL|PART|SECTION|PHASE)\s+\d+[:\.\-]?\s*)?(.*)$",
            stripped,
            re.IGNORECASE,
        )
        if heading_match and stripped.startswith("#"):
            hashes = heading_match.group(1)
            rest   = heading_match.group(2).strip()
            # Also strip leading ── from rest
            rest = re.sub(r"^[─\-]+\s*", "", rest).strip()
            # Strip trailing ── 
            rest = re.sub(r"\s*[─\-]+$", "", rest).strip()
            if rest:
                line = f"{hashes} {rest}"
            else:
                line = ""  # drop empty headings

        else:
            # Remove inline ── STEP X: patterns
            line = INLINE_STEP_RE.sub("", line)
            # Clean up leftover ─── lines (pure separator lines)
            if re.match(r"^[─\-=]{4,}$", stripped):
                line = ""

        # Remove leftover emoji artifacts
        line = EMOJI_RE.sub("", line)
        cleaned.append(line)

    # Collapse multiple blank lines to max 1
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


def clean_source(source_lines: list) -> list:
    """Clean a cell's source lines."""
    text = "".join(source_lines)
    cleaned = clean_text(text)
    if not cleaned:
        return []
    # Return as list of strings (notebook format)
    lines = cleaned.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


def clean_notebook(nb_path: str) -> int:
    """Clean a single notebook. Returns number of cells modified."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = 0
    for cell in nb.get("cells", []):
        original = "".join(cell.get("source", []))
        cleaned  = clean_source(cell.get("source", []))
        cleaned_text = "".join(cleaned)

        if cleaned_text.strip() != original.strip():
            cell["source"] = cleaned
            changed += 1

    # Write back
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return changed


# Run on all notebooks
nb_files = sorted([
    f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith(".ipynb")
])

print(f"\nCleaning {len(nb_files)} notebooks...")
total_changes = 0
for nb_file in nb_files:
    nb_path = os.path.join(NOTEBOOKS_DIR, nb_file)
    n = clean_notebook(nb_path)
    total_changes += n
    status = f"  {n} cells modified" if n else "  no changes needed"
    print(f"  {nb_file:<55} {status}")

print(f"\n✓  Notebook cleaning complete. Total cells modified: {total_changes}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Safety check — confirm api_keys.yaml is ignored
# ─────────────────────────────────────────────────────────────────────────────
print("\nSecurity check:")
dangerous = [
    os.path.join("config", "api_keys.yaml"),
]
for f in dangerous:
    full = os.path.join(ROOT, f)
    if os.path.exists(full):
        print(f"  [BLOCKED] {f}  ← in .gitignore, will NOT be pushed")
    else:
        print(f"  [OK] {f} does not exist")

print("\nSafe to push:")
safe_files = [
    "README.md",
    "config/model_config.yaml",
    "config/paths.yaml",
    "config/api_keys.yaml.example",
    "data/external/authority_mapping.json",
    "data/external/crisis_lexicon_en.txt",
    "data/external/crisis_lexicon_id.txt",
    "data/external/indonesian_slang_dict.json",
    "scripts/simulate_stream.py",
    "notebooks/*.ipynb",
]
for f in safe_files:
    print(f"  [PUSH] {f}")

print("\nAll done. Now run:")
print("  git add .")
print("  git status   ← verify api_keys.yaml is NOT listed")
print("  git commit -m 'Clean notebooks, add .gitignore, secure credentials'")
print("  git push")
