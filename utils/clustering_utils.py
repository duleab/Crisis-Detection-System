import numpy as np
from collections import Counter

try:
    from sklearn.cluster import DBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from geo_utils import haversine_distance
from temporal_utils import temporal_proximity


# ── Composite Distance Matrix ─────────────────────────────────────────────────
def build_composite_distance_matrix(
    embeddings, timestamps, coordinates,
    weights=None
):
    """
    Build a composite pairwise distance matrix from three components:
      - semantic_dist  (1 - cosine_similarity of SBERT embeddings)
      - temporal_dist  (normalized time difference over 24h window)
      - spatial_dist   (normalized haversine distance over 100km window)

    Args:
        embeddings:    np.ndarray shape (N, D)  — SBERT vectors (L2 normalised)
        timestamps:    list of datetime or None — event timestamps
        coordinates:   list of (lat, lon) or (None, None)
        weights:       dict with keys semantic, temporal, spatial

    Returns:
        np.ndarray shape (N, N) — composite distance matrix (values 0–1)
    """
    if weights is None:
        weights = {'semantic': 0.50, 'temporal': 0.30, 'spatial': 0.20}

    N = len(embeddings)

    # ── Semantic distance ────────────────────────────────────────────────────
    # cosine similarity = dot product of unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embeddings / norms
    cosine_sim = normed @ normed.T
    semantic_dist = np.clip(1.0 - cosine_sim, 0.0, 1.0)

    # ── Temporal distance ─────────────────────────────────────────────────────
    temporal_dist = np.full((N, N), 0.5)  # neutral default if unknown
    for i in range(N):
        for j in range(i + 1, N):
            prox = temporal_proximity(timestamps[i], timestamps[j],
                                      window_hours=24)
            d = 1.0 - prox
            temporal_dist[i, j] = d
            temporal_dist[j, i] = d
    np.fill_diagonal(temporal_dist, 0.0)

    # ── Spatial distance ──────────────────────────────────────────────────────
    MAX_KM = 100.0
    spatial_dist = np.full((N, N), 0.5)  # neutral if geo unknown
    for i in range(N):
        for j in range(i + 1, N):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            if None in (lat1, lon1, lat2, lon2):
                d = 0.5  # unknown
            else:
                km = haversine_distance(lat1, lon1, lat2, lon2)
                d = min(1.0, km / MAX_KM)
            spatial_dist[i, j] = d
            spatial_dist[j, i] = d
    np.fill_diagonal(spatial_dist, 0.0)

    # ── Composite ─────────────────────────────────────────────────────────────
    composite = (
        weights['semantic']  * semantic_dist +
        weights['temporal']  * temporal_dist +
        weights['spatial']   * spatial_dist
    )
    return composite


# ── DBSCAN ────────────────────────────────────────────────────────────────────
def run_dbscan(distance_matrix, eps=0.35, min_samples=2):
    """
    Run DBSCAN on a precomputed distance matrix.
    Returns array of cluster labels (−1 = noise / singleton event).
    """
    if not _SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn not available. Returning all as noise.")
        return np.full(len(distance_matrix), -1)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(distance_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  DBSCAN: {n_clusters} clusters, {n_noise} singleton posts "
          f"(eps={eps}, min_samples={min_samples})")
    return labels


# ── Canonical event extraction ─────────────────────────────────────────────────
def extract_canonical_event(cluster_df):
    """
    Given a DataFrame of posts belonging to one cluster,
    produce a canonical event record.

    Args:
        cluster_df: DataFrame with columns including:
            post_id, predicted_class, confidence_score, credibility_score,
            geo_lat, geo_lon, geo_confidence, timestamp_event_utc,
            text_cleaned, source_platform, has_media, engagement_score

    Returns:
        dict representing the canonical event
    """
    if cluster_df.empty:
        return {}

    # Event type: majority vote weighted by confidence
    event_type = cluster_df['predicted_class'].mode().iloc[0] \
        if not cluster_df['predicted_class'].isna().all() else 'unknown'

    # Location: credibility-weighted centroid
    geo_rows = cluster_df.dropna(subset=['geo_lat', 'geo_lon'])
    if not geo_rows.empty:
        weights = geo_rows.get('geo_confidence', 1.0)
        if hasattr(weights, 'values'):
            weights = weights.fillna(0.5).values
        else:
            weights = np.ones(len(geo_rows))
        weights = np.where(weights == 0, 0.01, weights)
        event_lat = float(np.average(geo_rows['geo_lat'], weights=weights))
        event_lon = float(np.average(geo_rows['geo_lon'], weights=weights))
    else:
        event_lat = None
        event_lon = None

    # Timestamp: median event time
    ts_col = 'timestamp_event_utc' if 'timestamp_event_utc' in cluster_df else 'timestamp_raw'
    timestamps = cluster_df[ts_col].dropna()
    event_timestamp = timestamps.median() if not timestamps.empty else None

    # Representative post: highest credibility score
    cred_col = 'credibility_score' if 'credibility_score' in cluster_df.columns \
        else 'confidence_score'
    rep_row = cluster_df.loc[cluster_df[cred_col].idxmax()]
    representative_text = rep_row.get('text_cleaned',
                                      rep_row.get('raw_text', ''))

    # Severity estimate
    cluster_size = len(cluster_df)
    source_platforms = list(cluster_df['source_platform'].unique()) \
        if 'source_platform' in cluster_df else []
    source_count = len(source_platforms)

    credibility_mean = float(cluster_df[cred_col].mean()) \
        if cred_col in cluster_df else 0.5

    # Severity scoring
    severity = compute_cluster_severity(cluster_size, source_count,
                                        credibility_mean)

    return {
        'event_type': event_type,
        'event_lat': event_lat,
        'event_lon': event_lon,
        'event_timestamp': str(event_timestamp) if event_timestamp else None,
        'event_severity': severity,
        'cluster_size': cluster_size,
        'source_count': source_count,
        'source_platforms': source_platforms,
        'cluster_credibility': round(credibility_mean, 3),
        'representative_text': str(representative_text)[:500],
        'llm_summary': None,  # filled later by llm_utils
    }


def compute_cluster_severity(cluster_size, source_count, credibility):
    """Estimate event severity: low / medium / high / critical."""
    score = (
        min(cluster_size / 50, 1.0) * 0.4 +
        min(source_count / 3,  1.0) * 0.3 +
        credibility * 0.3
    )
    if score >= 0.75:
        return 'critical'
    elif score >= 0.50:
        return 'high'
    elif score >= 0.25:
        return 'medium'
    return 'low'


# ── Cluster evaluation ────────────────────────────────────────────────────────
def compute_cluster_purity(cluster_labels, true_labels):
    """
    Compute cluster purity: fraction of posts that share the majority label
    in their cluster.
    """
    if len(cluster_labels) != len(true_labels):
        return 0.0

    total = 0
    correct = 0
    clusters = set(cluster_labels)

    for cid in clusters:
        if cid == -1:  # noise
            continue
        mask = [i for i, l in enumerate(cluster_labels) if l == cid]
        labels_in_cluster = [true_labels[i] for i in mask]
        majority_count = Counter(labels_in_cluster).most_common(1)[0][1]
        correct += majority_count
        total += len(mask)

    return correct / total if total > 0 else 0.0
