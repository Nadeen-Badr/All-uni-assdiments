import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import random
import numpy as np

def read_data(file_path, percentage):
    data = []
    movies = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header//
        for row in reader:
            rating = float(row[3])  # IMDB Rating column../
            data.append(rating)
            movies.append(row[0] + ' (' + row[3] + ')')  # Movie Name + Rating..//

    # Calculate the number of records to read based on the percentage.//./
    num_records = int(len(data) * percentage / 100)
    data = data[:num_records]
    movies = movies[:num_records]

    return data, movies

def cluster_data(file_path, percentage, k):
    data, movies = read_data(file_path, percentage)
    
    """Detect outliers using the Interquartile Range (IQR) method.
    This method identifies outliers as data points that fall below Q1 - 1.5 * IQR
    or above Q3 + 1.5 * IQR, where Q1 is the first quartile, Q3 is the third quartile,
    and IQR is the interquartile range (Q3 - Q1).
    """
    def detect_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        # Calculate the interquartile range (IQR)//
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Create a list of outliers with their corresponding values/
        outliers = [(movies[i], data[i]) for i in range(len(data)) if data[i] < lower_bound or data[i] > upper_bound]
        return outliers, len(outliers), lower_bound, upper_bound

    outliers, num_outliers, lower_bound, upper_bound = detect_outliers(data)

   # Display outliers information in the GUI/
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Number of outliers: {num_outliers}\n")
    result_text.insert(tk.END, f"Lower bound: {lower_bound}\n")
    result_text.insert(tk.END, f"Upper bound: {upper_bound}\n")
    result_text.insert(tk.END, "Outliers:\n")
    for outlier in outliers:
       result_text.insert(tk.END, f"{outlier[0]} - Rating: {outlier[1]}\n")
    
    #Remove outliers from the data..
    data = [rating for rating in data if rating >= lower_bound and rating <= upper_bound]

    # Randomly select k indices from the data/
    centroids_idx = random.sample(range(len(data)), k)
    # Initialize centroids as the data values at the selected indices/
    centroids = [data[i] for i in centroids_idx]

    # Maximum number of iterations/.
    max_iter = 100

    def euclidean_distance(x, y):
        # Calculate the squared differences between the coordinates..
        return np.sqrt((x - y) ** 2)

    for iteration in range(max_iter):
        # Assign each data point to the nearest centroid//
        clusters = []
        for rating in data:
            # Calculate the Euclidean distance from each point to all centroids//
            distances = [euclidean_distance(rating, centroid) for centroid in centroids]
            # Assign the point to the cluster of the nearest centroid..
            cluster = np.argmin(distances)
            clusters.append(cluster)

        # Update centroids.
        new_centroids = []
        for i in range(k):
            # Get all points assigned to cluster i
            cluster_ratings = [data[j] for j in range(len(data)) if clusters[j] == i]
            if len(cluster_ratings) > 0:
                # Calculate the new centroid as the average of all points in the cluster/
                new_centroid = sum(cluster_ratings) / len(cluster_ratings)
            else:
                new_centroid = centroids[i]  # Keep the centroid unchanged if no points in the cluster///
            new_centroids.append(new_centroid)

        # Check for convergence/
        if np.all(centroids == new_centroids):
            break
        # Update centroids for the next iteration .
        centroids = new_centroids

        # Print clusters at each iteration//
        print(f"Iteration {iteration + 1}:")
        for i in range(k):
            cluster_indices = [j for j in range(len(data)) if clusters[j] == i]
            cluster_movies = [movies[j] for j in cluster_indices]
            print(f"Cluster {i + 1} (Total Movies: {len(cluster_movies)}):")
            for movie in cluster_movies:
                print(movie)
            print(f"Centroid: {centroids[i]}")
            print()

    # Display clustered data..
    result_text.insert(tk.END, "\n\nClustering Results:\n")
    for i in range(k):
        cluster_indices = [j for j in range(len(data)) if clusters[j] == i]
        cluster_movies = [movies[j] for j in cluster_indices]
        result_text.insert(tk.END, f"Cluster {i + 1} (Total Movies: {len(cluster_movies)}):\n")
        for movie in cluster_movies:
            result_text.insert(tk.END, movie + "\n")
        result_text.insert(tk.END, f"Centroid: {centroids[i]}\n\n")

 
def run_clustering():
    file_path = file_path_entry.get()
    try:
        percentage = float(percentage_entry.get())
        k = int(k_entry.get())
        if percentage <= 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100")
        if k <= 0:
            raise ValueError("Number of clusters (k) must be greater than 0")
        cluster_data(file_path, percentage, k)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)

# GUI
root = tk.Tk()
root.title("Movie Clustering")
root.geometry("800x600")

file_path_label = tk.Label(root, text="Select CSV file:")
file_path_label.pack()
file_path_entry = tk.Entry(root, width=50)
file_path_entry.pack()
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack()

percentage_label = tk.Label(root, text="Percentage of data to read:")
percentage_label.pack()
percentage_entry = tk.Entry(root)
percentage_entry.pack()

k_label = tk.Label(root, text="Number of clusters (k):")
k_label.pack()
k_entry = tk.Entry(root)
k_entry.pack()

run_button = tk.Button(root, text="Run Clustering", command=run_clustering)
run_button.pack()

result_label = tk.Label(root, text="Clustering Results:")
result_label.pack()
result_text = tk.Text(root, height=20, width=100)
result_text.pack()

root.mainloop()
