import csv
import random
import numpy as np
from sklearn.cluster import KMeans

# Load the dataset
data = []
with open('j.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        rating = float(row[3])  # IMDB Rating column
        data.append([rating])

# Number of clusters
k = int(input("Enter the number of clusters (k): "))

# Fit k-means model
kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

# Assign each data point to a cluster
final_clusters = kmeans.predict(data)

# Display clustered data
for i in range(k):
    cluster_indices = [j for j in range(len(data)) if final_clusters[j] == i]
    print(f"Cluster {i+1} (Total Movies: {len(cluster_indices)}):")
    for idx in cluster_indices:
        print(f"{idx + 1}. {movies[idx]}")
    print(f"Centroid: {kmeans.cluster_centers_[i][0]}")
    print()

# Detect and display outliers
distances = kmeans.transform(data)
average_distances = np.mean(distances, axis=1)
threshold = 2 * np.mean(average_distances)
outliers = [movies[j] for j in range(len(data)) if average_distances[j] > threshold]
print("Outliers:")
print(outliers)