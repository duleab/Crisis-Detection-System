import json

def lookup_authority(lat, lon, event_type, authority_map, gadm_gdf):
    return {"name": "BPBD", "contact": "mock"}

def translate_event_type(event_type_en):
    trans = {"flood": "banjir", "earthquake": "gempa bumi"}
    return trans.get(event_type_en, event_type_en)

def generate_alert_text(event_record, authority, lang="en"):
    return "ALERT: Event detected."

def build_alert_json(event_record, authority, alert_texts):
    return "{}"

def compute_severity(cluster_size, casualty_count, distance_to_pop):
    return "medium"
