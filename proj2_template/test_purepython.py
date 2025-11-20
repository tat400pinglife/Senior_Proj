import sys
import numpy as np

import pandas as pd
#import cudf
import timeit

#code from: https://github.com/rapidsai/cuspatial/blob/724d170a2105441a3533b5eaf9ee82ddcfc49be0/notebooks/nyc_taxi_years_correlation.ipynb
#data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

#taxi = cudf.read_parquet("yellow_tripdata_2009-01.parquet")
#read from files 01 to 12
def haversine_distance(size, x1, y1, x2, y2, dist):
    R = 6371.0  # Earth radius in kilometers
    for i in range(size):
        lat1 = np.radians(y1[i])
        lon1 = np.radians(x1[i])
        lat2 = np.radians(y2[i])
        lon2 = np.radians(x2[i])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        dist[i] = R * c

query_expr = (
    "Start_Lon >= -74.15 & "
    "Start_Lat >= 40.5774 &"
    "End_Lon <= -73.7004 & "
    "End_Lat <= 40.9176"
)
start_lon = -74.15
start_lat = 40.5774
end_lon = -73.7004
end_lat = 40.9176

all_x1 = []
all_y1 = []
all_x2 = []
all_y2 = []

data_dir = "/tmp/tlcdata"

try:
    with open("./tlcdata/yellow_tripdata_2009-01.parquet", "rb") as f:
        data_dir = "./tlcdata"
        print(f"Reading from {data_dir}")
except (FileNotFoundError, OSError):
    print(f"Reading from {data_dir}")


try:
    with open("Parquet_Data/yellow_tripdata_2009-01.parquet", "rb") as f:
        data_dir = "Parquet_Data"
        print(f"Reading from {data_dir}")
except (FileNotFoundError, OSError):
    print(f"Reading from {data_dir}")

    

for month in range(1, 13):
    month_str = f"{month:02d}"
    file_path = f"{data_dir}/yellow_tripdata_2009-{month_str}.parquet"

    print("Loading:", file_path)
    df = pd.read_parquet(file_path)

    # Filter
    df = df.query(query_expr)
    cur = cur[(cur['Start_Lon'] >= start_lon) & (cur['Start_Lat'] >= start_lat) & (cur['End_Lon'] <= end_lon) & (cur['End_Lat'] <= end_lat)]
    all_x1.append(df["Start_Lon"].to_numpy())
    all_y1.append(df["Start_Lat"].to_numpy())
    all_x2.append(df["End_Lon"].to_numpy())
    all_y2.append(df["End_Lat"].to_numpy())

    del df 

x1 = np.concatenate(all_x1)
y1 = np.concatenate(all_y1)
x2 = np.concatenate(all_x2)
y2 = np.concatenate(all_y2)

del all_x1, all_y1, all_x2, all_y2
size = len(x1)
dist = np.zeros(size, dtype=np.float64)

# Source - https://stackoverflow.com/q
# Posted by gilbert8, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-20, License - CC BY-SA 4.0

start = timeit.timeit()

haversine_distance(size, x1, y1, x2, y2, dist)
end = timeit.timeit()
print(end - start)
print("Kernel finished for", size, "rows.")
print("Distances:", dist)