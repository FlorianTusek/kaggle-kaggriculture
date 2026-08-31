"""Download replay datasets from Kaggle and explore their structure."""
import os
import json
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")
os.makedirs(DATA_DIR, exist_ok=True)

# Priority datasets to download
DATASETS = [
    "kaggle/kaggriculture-episodes-2026-08-09",   # Latest official episodes
    "kaggle/kaggriculture-episodes-2026-08-08",   # Second latest
    "georgymamarin/kaggriculture-episodes",        # Community curated
]

for ds_ref in DATASETS:
    ds_dir = os.path.join(DATA_DIR, ds_ref.replace("/", "_"))
    os.makedirs(ds_dir, exist_ok=True)
    
    print(f"\n=== Downloading: {ds_ref} ===")
    try:
        # List files first
        files = api.dataset_list_files(ds_ref)
        file_list = files.files if hasattr(files, 'files') else files
        print(f"  Files in dataset: {len(file_list)}")
        for f in file_list[:10]:
            fname = f.name if hasattr(f, 'name') else str(f)
            fsize = f.totalBytes if hasattr(f, 'totalBytes') else '?'
            print(f"    {fname} ({fsize} bytes)")
        
        # Download the dataset
        print(f"  Downloading to {ds_dir}...")
        api.dataset_download_files(ds_ref, path=ds_dir, unzip=True)
        print(f"  Download complete!")
        
        # List what we got
        downloaded = os.listdir(ds_dir)
        print(f"  Downloaded files: {downloaded[:10]}")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n=== DONE ===")
print(f"Replay data directory: {DATA_DIR}")
for item in os.listdir(DATA_DIR):
    full = os.path.join(DATA_DIR, item)
    if os.path.isdir(full):
        count = len(os.listdir(full))
        print(f"  {item}/  ({count} items)")
    else:
        print(f"  {item}  ({os.path.getsize(full)} bytes)")
