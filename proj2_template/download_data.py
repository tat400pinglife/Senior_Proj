import os
import requests
import time

# Matches the path in test_nycyellowcab.py 
useMyDir = True
serverDir = "/tmp"
myDir = "."

dirToUse = myDir if useMyDir else serverDir
output_dir = f"{dirToUse}/tlcdata" 
os.makedirs(output_dir, exist_ok=True)

# NYC TLC S3 Bucket URL
base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2009-{:02d}.parquet"

print(f"--- Downloading 2009 Data to {output_dir} ---")
print("This will take a few minutes (Total Size: ~5.5 GB)")

for month in range(1, 13):
    url = base_url.format(month)
    filename = os.path.join(output_dir, f"yellow_tripdata_2009-{month:02d}.parquet")
    
    if os.path.exists(filename):
        print(f"Month {month:02d}: Already exists. Skipping.")
        continue
        
    print(f"Downloading Month {month:02d}...", end=" ", flush=True)
    try:
        start = time.time()
        # Stream download to avoid high memory usage
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        duration = time.time() - start
        print(f"Done ({duration:.1f}s)")
    except Exception as e:
        print(f"\n[ERROR] Failed to download month {month}: {e}")

print("\n✅ All downloads complete. You can now run test_nycyellowcab.py.")
#vibe coded bs btw