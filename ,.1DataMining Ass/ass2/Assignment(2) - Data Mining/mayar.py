import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('j.csv')

# Select only the IMDb ratings column
ratings = data['IMDB Rating'].values.reshape(-1, 1)

# Standardize the data
scaler = StandardScaler()
ratings_scaled = scaler.fit_transform(ratings)

# Get user input for the number of clusters (k)
k = int(input("Enter the number of clusters (k): "))

# Perform k-means clustering
kmeans = KMeans(n_clusters=k, init='random', random_state=42)
kmeans.fit(ratings_scaled)

# Get the cluster labels and assign them to the original dataframe
data['Cluster'] = kmeans.labels_

# Calculate the Euclidean distance of each point from its centroid
distances = kmeans.transform(ratings_scaled)

# Detect outliers
outliers_threshold = 2  # You can adjust this threshold as needed
outliers = distances.max(axis=1) > outliers_threshold

# Print outliers' records
print("Outlier records:")
print(data[outliers])

# Print each cluster
for cluster_num in range(k):
    print(f"\nCluster {cluster_num}:")
    cluster_data = data[data['Cluster'] == cluster_num]
    print(cluster_data[['Movie Name', 'IMDB Rating']])