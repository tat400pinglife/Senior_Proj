import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd

filters = [
    ('Start_Lon', '>=', -74.15),
    ('Start_Lon', '<=', -73.70),
    ('Start_Lat', '>=', 40.55),
    ('Start_Lat', '<=', 40.90)
]

dfs = []
for month in range(1, 13):
    file_path = f"Parquet_Data/yellow_tripdata_2009-{month:02d}.parquet"
    table = pq.read_table(file_path, filters=filters)
    dfs.append(table)

taxi = pd.concat([t.to_pandas() for t in dfs], ignore_index=True)
taxi.to_parquet("combined_data.parquet")
