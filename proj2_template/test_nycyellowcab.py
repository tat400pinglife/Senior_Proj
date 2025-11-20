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

query_str = f"""Start_Lon >= {start_lon} &
Start_Lat >= {start_lat} &
End_Lon >= {end_lon} &
End_Lat >= {end_lat}
"""
all_x1 = []
all_y1 = []
all_x2 = []
all_y2 = []

for i in range(12):
    filename = f"/tmp/tlcdata/yellow_tripdata_2009-{i+1:02d}.parquet"
    print(f"Reading {filename}...")
    cur = pd.read_parquet(filename)
    print("Filtering...")
    cur = cur.query(query_str)
    all_x1.append(cur['Start_Lon'].to_numpy())
    all_y1.append(cur['Start_Lat'].to_numpy())
    all_x2.append(cur['End_Lon'].to_numpy())
    all_y2.append(cur['End_Lat'].to_numpy())





x1=np.concatenate(all_x1)
y1=np.concatenate(all_y1)
x2=np.concatenate(all_x2)
y2=np.concatenate(all_y2)

size=len(x1)
dist=np.zeros(size)
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)

print("dist:", dist)