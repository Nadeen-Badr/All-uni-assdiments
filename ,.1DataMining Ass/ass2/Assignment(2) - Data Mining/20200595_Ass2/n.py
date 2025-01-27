import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import random
import numpy as np

class Movie:
    def __init__(self, name, rating):
        self.name = name
        self.rating = rating
        self.centroid = None

def read_data(file_path, percentage):
    movies = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            rating = float(row[3])  # IMDB Rating column
            movie = Movie(row[0], rating)  # Create Movie instance
            movies.append(movie)

    # Calculate the number of records to read based on the percentage
    num_records = int(len(movies) * percentage / 100)
    movies = movies[:num_records]

    return movies

def cluster_data(file_path, percentage, k):
    movies = read_data(file_path, percentage)
    data = [movie.rating for movie in movies]
    
    """Detect outliers using the Interquartile Range (IQR) method.
    This method identifies outliers as data points that fall below Q1 - 1.5 * IQR
    or above Q3 + 1.5 * IQR, where Q1 is the first quartile, Q3 is the third quartile,
    and IQR is the interquartile range (Q3 - Q1).
    """
    def detect_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        # Calculate the interquartile range (IQR)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Create a list of outliers with their corresponding values
        outliers = [(movie.name, movie.rating) for movie in movies if movie.rating < lower_bound or movie.rating > upper_bound]
        return outliers, len(outliers), lower_bound, upper_bound

    outliers, num_outliers, lower_bound, upper_bound = detect_outliers(data)

    # Display outliers information in the GUI
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Number of outliers: {num_outliers}\n")
    result_text.insert(tk.END, f"Lower bound: {lower_bound}\n")
    result_text.insert(tk.END, f"Upper bound: {upper_bound}\n")
    result_text.insert(tk.END, "Outliers:\n")
    for outlier in outliers:
        result_text.insert(tk.END, f"{outlier[0]} - Rating: {outlier[1]}\n")
    
    # Remove outliers from the data
    movies = [movie for movie in movies if movie.rating >= lower_bound and movie.rating <= upper_bound]
    data = [rating for rating in data if rating >= lower_bound and rating <= upper_bound]
    # Randomly select k indices from the data
    centroids_idx = random.sample(range(len(data)), k)
    # Initialize centroids as the data values at the selected indices
    centroids = [data[i] for i in centroids_idx]

    # Maximum number of iterations
    max_iter = 100

    def euclidean_distance(x, y):
        # Calculate the squared differences between the coordinates
        return np.sqrt((x - y) ** 2)

    for iteration in range(max_iter):
        # Assign each data point to the nearest centroid
        for movie in movies:
            # Calculate the Euclidean distance from the movie's rating to all centroids
            distances = [euclidean_distance(movie.rating, centroid) for centroid in centroids]
            # Assign the movie to the cluster of the nearest centroid index
            movie.centroid = np.argmin(distances)

        # Update centroids
        new_centroids = []
        for i in range(k):
            # Get all movies assigned to cluster i
            cluster_movies = [movie for movie in movies if movie.centroid == i]
            if len(cluster_movies) > 0:
                # Calculate the new centroid as the average of all ratings in the cluster
                new_centroid = sum(movie.rating for movie in cluster_movies) / len(cluster_movies)
            else:
                new_centroid = centroids[i]  # Keep the centroid unchanged if no movies in the cluster
            new_centroids.append(new_centroid)
        # Print clusters at each iteration
        print(f"Iteration {iteration + 1}:")
        for i in range(k):
            cluster_movies = [movie.name for movie in movies if movie.centroid == i]
            print(f"Cluster {i + 1} (Total Movies: {len(cluster_movies)}):")
            # for movie in cluster_movies:
            #     print(movie)
            print(f"Centroid: {centroids[i]}")
            print()

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

        # Print clusters at each iteration
        print(f"Iteration {iteration + 1}:")
        for i in range(k):
            cluster_movies = [movie.name for movie in movies if movie.centroid == i]
            print(f"Cluster {i + 1} (Total Movies: {len(cluster_movies)}):")
            # for movie in cluster_movies:
            #     print(movie)
            print(f"Centroid: {centroids[i]}")
            print()

    # Display clustered data
    result_text.insert(tk.END, "\n\nClustering Results:\n")
    for i in range(k):
        cluster_movies = [(movie.name, movie.rating) for movie in movies if movie.centroid == i]
        result_text.insert(tk.END, f"Cluster {i + 1} (Total Movies: {len(cluster_movies)}):\n")
        for movie_name, rating in cluster_movies:
            result_text.insert(tk.END, f"{movie_name} (Rating: {rating})\n")
        result_text.insert(tk.END, f"Centroid: {centroids[i]}\n\n")
    
    print(f"final Iteration :")
    for i in range(k):
        cluster_movies = [movie.name for movie in movies if movie.centroid == i]
        print(f"Cluster {i + 1} (Total Movies: {len(cluster_movies)}):")
        print(f"Centroid: {centroids[i]}")
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