import pandas as pd
import os
import time

# Path to parquet file
parquet_file = "../proj2_template/parquet_Data/yellow_tripdata_2009-01.parquet"
csv_file = "taxi_data.csv"

# Get parquet file size
parquet_size = os.path.getsize(parquet_file)
print(f"Parquet file size: {parquet_size / (1024**2):.2f} MB")

# Read parquet file and time it
start_time = time.time()
df_parquet = pd.read_parquet(parquet_file)
parquet_time = (time.time() - start_time) * 1000
print(f"Time to read parquet: {parquet_time:.2f} ms")

# Save to CSV
df_parquet.to_csv(csv_file, index=False)

# Get CSV file size
csv_size = os.path.getsize(csv_file)
print(f"CSV file size: {csv_size / (1024**2):.2f} MB")

# Read CSV file and time it
start_time = time.time()
df_csv = pd.read_csv(csv_file)
csv_time = (time.time() - start_time) * 1000
print(f"Time to read CSV: {csv_time:.2f} ms")

# Summary
print(f"\nSummary:")
print(f"Parquet: {parquet_size / (1024**2):.2f} MB, {parquet_time:.2f} ms")
print(f"CSV: {csv_size / (1024**2):.2f} MB, {csv_time:.2f} ms")