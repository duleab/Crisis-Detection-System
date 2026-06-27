import pandas as pd
import time
import os
import shutil

print("🚨 Starting Simulated Streaming Worker 🚨")
print("This script simulates a live data stream by pushing historical posts into a live buffer.")

# Define paths
base_dir = "d:/Project/10academy/Real-Time Crisis Detection"
unified_csv = f"{base_dir}/data/processed/posts_unified.csv"
buffer_dir = f"{base_dir}/data/stream_buffer"

os.makedirs(buffer_dir, exist_ok=True)

# Generate some dummy data if posts_unified doesn't exist
if not os.path.exists(unified_csv):
    print("No unified data found. Creating dummy stream data...")
    dummy_data = pd.DataFrame({
        "post_id": [1, 2, 3],
        "raw_text": ["Banjir parah di Kemang", "Gempa terasa di Bandung", "Tolong ada kebakaran di pasar"],
        "timestamp_raw": ["2024-01-01 10:00:00", "2024-01-01 10:05:00", "2024-01-01 10:10:00"]
    })
    os.makedirs(os.path.dirname(unified_csv), exist_ok=True)
    dummy_data.to_csv(unified_csv, index=False)

df = pd.read_csv(unified_csv)
print(f"Loaded {len(df)} historical posts.")

# Simulate streaming
for index, row in df.iterrows():
    print(f"[{row['timestamp_raw']}] New event received -> Buffer")
    # In a real system, we'd trigger the inference pipeline here
    # For now, we just simulate the delay
    time.sleep(2)

print("Stream simulation complete.")
