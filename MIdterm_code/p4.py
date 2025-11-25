import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
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
print(f"Time to compute Haversine distance matrix: {haversine_time:.2f} ms")
# eps is in radians: for 1000 km, eps = 1000 / 6371
eps_km = 1000
eps_rad = eps_km / 6371
db = DBSCAN(eps=eps_rad, min_samples= 3, metric='haversine')
labels = db.fit_predict(coords_rad)

big_cities['cluster'] = labels

plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    big_cities['lng'], big_cities['lat'],
    c=big_cities['cluster'], cmap='tab20', s=50
)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering of Cities with Population > 1 Million')
plt.show()

agg = AgglomerativeClustering(
    n_clusters=10, linkage='average', metric='precomputed'
)
labels_agg = agg.fit_predict(dist_matrix_km)
big_cities['cluster_agg'] = labels_agg
plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    big_cities['lng'], big_cities['lat'],
    c=big_cities['cluster_agg'], cmap='tab20', s=50
)
plt.colorbar(scatter, label='Cluster_agg')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Agglomerative Clustering of Cities with Population > 1 Million')
plt.show()