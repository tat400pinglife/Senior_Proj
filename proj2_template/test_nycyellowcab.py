import sys
import numpy as np
sys.path.append('./build')

import pandas as pd
import haversine_library

query_expr = (
    "Start_Lon >= -74.15 & Start_Lon <= -73.7004 & "
    "Start_Lat >= 40.5774 & Start_Lat <= 40.9176 & "
    "End_Lon >= -74.15 & End_Lon <= -73.7004 & "
    "End_Lat >= 40.5774 & End_Lat <= 40.9176"
)

# Collect arrays across all months
all_x1 = []
all_y1 = []
all_x2 = []
all_y2 = []

for month in range(1, 13):
    month_str = f"{month:02d}"
    file_path = f"/tmp/tlcdata/yellow_tripdata_2009-{month_str}.parquet"

    print("Loading:", file_path)
    df = pd.read_parquet(file_path)

    # Filter
    df = df.query(query_expr)

    # Extract the coordinate columns (very small memory)
    all_x1.append(df["Start_Lon"].to_numpy())
    all_y1.append(df["Start_Lat"].to_numpy())
    all_x2.append(df["End_Lon"].to_numpy())
    all_y2.append(df["End_Lat"].to_numpy())

    del df  # free memory early

# ---- CONCATENATE ----
x1 = np.concatenate(all_x1)
y1 = np.concatenate(all_y1)
x2 = np.concatenate(all_x2)
y2 = np.concatenate(all_y2)

del all_x1, all_y1, all_x2, all_y2

# ---- RUN YOUR KERNEL ONCE ----
size = len(x1)
dist = np.zeros(size, dtype=np.float64)

haversine_library.haversine_distance(size, x1, y1, x2, y2, dist)

print("Kernel finished for", size, "rows.")