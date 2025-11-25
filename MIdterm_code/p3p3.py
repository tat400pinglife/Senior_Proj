import pandas as pd
import os
import time

parquet_file = "/tmp/tlcdata/yellow_tripdata_2009-01.parquet"
csv_file = "taxi_data.csv"

parquet_size = os.path.getsize(parquet_file)
print(f"Parquet file size: {parquet_size / (1024**2):.2f} MB")

start_time = time.time()
df_parquet = pd.read_parquet(parquet_file)
parquet_time = (time.time() - start_time) * 1000
print(f"Time to read parquet: {parquet_time:.2f} ms")

df_parquet.to_csv(csv_file, index=False)

csv_size = os.path.getsize(csv_file)
print(f"CSV file size: {csv_size / (1024**2):.2f} MB")

start_time = time.time()
df_csv = pd.read_csv(csv_file)
csv_time = (time.time() - start_time) * 1000
print(f"Time to read CSV: {csv_time:.2f} ms")

print(f"\nSummary:")
print(f"Parquet: {parquet_size / (1024**2):.2f} MB, {parquet_time:.2f} ms")
print(f"CSV: {csv_size / (1024**2):.2f} MB, {csv_time:.2f} ms")