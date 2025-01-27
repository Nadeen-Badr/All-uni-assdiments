import csv
import random
import numpy as np

# Load the dataset
data = []
movies = []
with open('j.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        rating = float(row[3])  # IMDB Rating column
        data.append(rating)
        movies.append(row[0])  # Movie Name

# Number of clusters
k = int(input("Enter the number of clusters (k): "))

# Randomly initialize centroids
centroids_idx = random.sample(range(len(data)), k)
centroids = [data[i] for i in centroids_idx]

# Maximum number of iterations
max_iter = 100

def euclidean_distance(x, y):
    return np.sqrt((x - y) ** 2)

for _ in range(max_iter):
    # Assign each data point to the nearest centroid
    clusters = []
    for rating in data:
        distances = [euclidean_distance(rating, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    
    # Update centroids
    new_centroids = []
    for i in range(k):
        cluster_ratings = [data[j] for j in range(len(data)) if clusters[j] == i]
        if len(cluster_ratings) > 0:
            new_centroid = sum(cluster_ratings) / len(cluster_ratings)
        else:
            new_centroid = centroids[i]  # Keep the centroid unchanged if no points in the cluster
        new_centroids.append(new_centroid)
    
    # Check for convergence
    if np.all(centroids == new_centroids):
        break
    
    centroids = new_centroids

# Assign each data point to the final clusters
final_clusters = []
for rating in data:
    distances = [euclidean_distance(rating, centroid) for centroid in centroids]
    cluster = np.argmin(distances)
    final_clusters.append(cluster)

# Display clustered data
for i in range(k):
    cluster_indices = [j for j in range(len(data)) if final_clusters[j] == i]
    cluster_movies = [movies[j] for j in cluster_indices]
    print(f"Cluster {i+1} (Total Movies: {len(cluster_movies)}):")
    for movie in cluster_movies:
        print(movie)
    print(f"Centroid: {centroids[i]}")
    print()

# Detect and display outliers
distances = []
for rating in data:
    distances.append([euclidean_distance(rating, centroid) for centroid in centroids])
average_distances = np.mean(distances, axis=1)
threshold = 2 * np.mean(average_distances)
outliers = [movies[j] for j in range(len(data)) if average_distances[j] > threshold]
print("Outliers:")
print(outliers)