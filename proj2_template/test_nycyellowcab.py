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
# from 01 to 12
for month in range(1,13):
    month_str = str(month).zfill(2)
    file_path = f"/tmp/tlcdata/yellow_tripdata_2009-{month_str}.parquet"
    taxi_month = pd.read_parquet(file_path)
    taxi_month.query('Start_Lon >= -74.15 & Start_Lon <= -73.70 & Start_Lat >= 40.55 & Start_Lat <= 40.90', inplace=True)
    if month == 1:
        taxi = taxi_month
    else:
        taxi = pd.concat([taxi, taxi_month], ignore_index=True)

x1=taxi['Start_Lon'].to_numpy()
y1=taxi['Start_Lat'].to_numpy()
x2=taxi['End_Lon'].to_numpy()
y2=taxi['End_Lat'].to_numpy()
size=len(x1)
dist=np.zeros(size)
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)

print("Distances (km):", dist)