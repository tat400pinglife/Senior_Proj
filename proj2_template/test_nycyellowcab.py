import sys
import numpy as np
sys.path.append('./build')

import pandas as pd
#import cudf
import haversine_library

#code from: https://github.com/rapidsai/cuspatial/blob/724d170a2105441a3533b5eaf9ee82ddcfc49be0/notebooks/nyc_taxi_years_correlation.ipynb
#data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

#taxi = cudf.read_parquet("yellow_tripdata_2009-01.parquet")
#read from files 01 to 12
query_expr = (
    "Start_Lon >= -74.15 & Start_Lon <= -73.7004 & "
    "Start_Lat >= 40.5774 & Start_Lat <= 40.9176 & "
    "End_Lon >= -74.15 & End_Lon <= -73.7004 & "
    "End_Lat >= 40.5774 & End_Lat <= 40.9176"
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
        print("Reading from ./tlcdata")
except (FileNotFoundError, OSError):
    print(f"Reading from {data_dir}")


for month in range(1, 13):
    month_str = f"{month:02d}"
    file_path = f"{data_dir}/yellow_tripdata_2009-{month_str}.parquet"

    print("Loading:", file_path)
    df = pd.read_parquet(file_path)

    # Filter
    #df = df.query(query_expr)
    df = df[(df['Start_Lon'] >= start_lon) & (df['Start_Lat'] >= start_lat) & (df['End_Lon'] <= end_lon) & (df['End_Lat'] <= end_lat)]
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

haversine_library.haversine_distance(size, x1, y1, x2, y2, dist)

print("Kernel finished for", size, "rows.")
print("Distances:", dist)