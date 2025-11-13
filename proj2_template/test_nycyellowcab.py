import sys, os, traceback
print("START")

print("sys.path before append:", sys.path[:5])
sys.path.append('./build')
print("sys.path after append:", sys.path[:5])

print("files in ./build:", os.listdir('./build') if os.path.isdir('./build') else "(no build dir)")

try:
    import pandas as pd
    print("pandas imported from:", pd.__file__)
except Exception as e:
    print("pandas import failed:", repr(e))
    traceback.print_exc()

try:
    import pyarrow
    print("pyarrow imported from:", pyarrow.__file__)
except Exception as e:
    print("pyarrow import failed:", repr(e))

try:
    import fastparquet
    print("fastparquet imported from:", fastparquet.__file__)
except Exception as e:
    print("fastparquet import failed:", repr(e))

try:
    import haversine_library
    print("haversine_library imported OK")
except Exception as e:
    print("haversine_library import failed:", repr(e))

print("END")

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