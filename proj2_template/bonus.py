import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import sys
sys.path.append('./build')

import haversine_library
import time

url = "https://gist.githubusercontent.com/jlewis8756/6b83a54351e91012b9fd541356a347c9/raw/ba14ce4e79cf9f7bc1225add2b214bffc3c26d52/world_cities.csv"
df = pd.read_csv(url)
big_cities = df[df["population"] > 1_000_000].reset_index(drop=True)
# print("Number of cities with population over 1 million:", len(big_cities))
# print(big_cities.head())
start_time = time.time()
coords_rad = np.radians(big_cities[["lat", "lng"]].to_numpy())
dist_matrix_km = haversine_distances(coords_rad) * 6371
haversine_time = (time.time() - start_time) * 1000
print(f"Time to compute Haversine distance using sklearn: {haversine_time:.2f} ms")

x1 = big_cities['lng'].to_numpy(dtype=np.float64) 
y1 = big_cities['lat'].to_numpy(dtype=np.float64)  
x2 = big_cities['lng'].to_numpy(dtype=np.float64)  
y2 = big_cities['lat'].to_numpy(dtype=np.float64)

size = len(x1)
dist = np.zeros(size, dtype=np.float64)

start_time = time.time()
haversine_library.haversine_distance(size, x1, y1, x2, y2, dist)
cuda_time = (time.time() - start_time) * 1000
print(dist)
print(f"Time to compute Haversine distance using CUDA: {cuda_time:.2f} ms")


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
        
start_time = time.time()
haversine_distance(size, x1, y1, x2, y2, dist)
python_time = (time.time() - start_time) * 1000
print(f"Time to compute Haversine distance using pure Python: {python_time:.2f} ms")

