print("Running")
import sys
import numpy as np
sys.path.append('./build')

import pandas as pd
#import cudf
import haversine_library

#code from: https://github.com/rapidsai/cuspatial/blob/724d170a2105441a3533b5eaf9ee82ddcfc49be0/notebooks/nyc_taxi_years_correlation.ipynb
#data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

#taxi = cudf.read_parquet("yellow_tripdata_2009-01.parquet")
df = pd.DataFrame()
start_lon = -74.15
start_lat = 40.5774
end_lon = -73.7004
end_lat = 40.9176
for i in range(12):
    filename = f"/tmp/tlcdata/yellow_tripdata_2009-{i+1:02d}.parquet"
    print(f"Reading {filename}...")
    cur = pd.read_parquet(filename)
    print("Filtering...")
    cur = cur[(cur['Start_Lon'] > start_lon) & (cur['Start_Lon'] > start_lat) & (cur['End_Lon'] > end_lat) & (cur['End_Lat'] > end_lat)]
    print("Concating...")
    df = pd.concat([df, cur])




x1=df['Start_Lon'].to_numpy()
y1=df['Start_Lat'].to_numpy()
x2=df['End_Lon'].to_numpy()
y2=df['End_Lat'].to_numpy()
size=len(x1)
dist=np.zeros(size)
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)

print("dist:", dist)