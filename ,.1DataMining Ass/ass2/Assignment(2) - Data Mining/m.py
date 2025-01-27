import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to perform k-means clustering
def k_means_clustering(data, k):
    # Select the feature for clustering (IMDB Rating)
    ratings = data['IMDB Rating'].values.reshape(-1, 1)

    # Initialize centroids randomly
    centroids = random.sample(list(ratings.flatten()), k)
    centroids = np.array(centroids).reshape(-1, 1)

    # Define a function to calculate Euclidean distance
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Perform k-means clustering
    max_iterations = 100
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        clusters = {}
        for i in range(k):
            clusters[i] = []

        for rating in ratings:
            distances = [euclidean_distance(rating, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(rating)

        # Update centroids
        new_centroids = []
        for cluster_idx, cluster_ratings in clusters.items():
            new_centroid = np.mean(cluster_ratings)
            new_centroids.append(new_centroid)

        # Check for convergence
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids

    # Detect outliers
    outliers = []
    for rating in ratings:
        distances = [euclidean_distance(rating, centroid) for centroid in centroids]
        if min(distances) > 1:  # Threshold for outliers
            outliers.append(rating)

    return clusters, outliers

# Function to run the clustering algorithm
def run_clustering():
    data = select_file()
    if data is None:
        messagebox.showerror("Error", "No file selected.")
        return

    k = int(entry_k.get())
    percentage = float(entry_percentage.get())
    num_records = int(len(data) * (percentage / 100))
    data = data.head(num_records)
    
    clusters, outliers = k_means_clustering(data, k)

    # Display clustering results
    for cluster_idx, cluster_ratings in clusters.items():
        centroid_value = np.mean(cluster_ratings)
        cluster_info = f"Cluster {cluster_idx + 1}: {len(cluster_ratings)} movies, Centroid: {centroid_value:.2f}"
        text_result.insert(tk.END, cluster_info + "\n")
        text_result.insert(tk.END, "Movies in this cluster:\n")
        for movie_idx in cluster_ratings:
            movie_name = data.iloc[movie_idx]['Movie Name']
            movie_rating = data.iloc[movie_idx]['IMDB Rating']
            text_result.insert(tk.END, f"- {movie_name} ({movie_rating})\n")
        text_result.insert(tk.END, "\n")

    # Display outliers
    text_result.insert(tk.END, "Outliers:\n")
    for movie_idx in data.loc[data['IMDB Rating'].isin(outliers)].index:
        movie_name = data.iloc[movie_idx]['Movie Name']
        movie_rating = data.iloc[movie_idx]['IMDB Rating']
        text_result.insert(tk.END, f"- {movie_name} ({movie_rating})\n")

# Function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        return data
    return None

# Create the main window
root = tk.Tk()
root.title("Movie Clustering")

# Label and entry for the number of clusters (K)
label_k = tk.Label(root, text="Enter the number of clusters (K):")
label_k.pack(pady=5)
entry_k = tk.Entry(root)
entry_k.pack(pady=5)

# Label and entry for the percentage of records to read
label_percentage = tk.Label(root, text="Enter the percentage of records to read:")
label_percentage.pack(pady=5)
entry_percentage = tk.Entry(root)
entry_percentage.pack(pady=5)

# Create a file selection button
btn_select_file = tk.Button(root, text="Select File", command=select_file)
btn_select_file.pack(pady=10)

# Create a button to run the clustering algorithm
btn_run = tk.Button(root, text="Run Clustering", command=run_clustering)
btn_run.pack(pady=10)

# Create a text widget to display the clustering results
text_result = tk.Text(root, height=20, width=50)
text_result.pack(pady=10)

# Start the GUI event loop
root.mainloop()