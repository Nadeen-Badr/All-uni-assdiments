import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering App")
        self.root.configure(bg="#ff99cc")  # Set background color
        
        self.file_path = None
        self.data = None
        self.clustered_data = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # File selection
        self.file_label = tk.Label(self.root, text="Select File:", bg="#ff99cc")
        self.file_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.file_button = tk.Button(self.root, text="Browse", command=self.browse_file)
        self.file_button.grid(row=0, column=1, padx=10, pady=5)
        
        # Percentage selection
        self.percent_label = tk.Label(self.root, text="Select Percentage:", bg="#ff99cc")
        self.percent_label.grid(row=1, column=0, padx=10, pady=5)
        
        self.percent_entry = tk.Entry(self.root)
        self.percent_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Clusters selection
        self.cluster_label = tk.Label(self.root, text="Number of Clusters (K):", bg="#ff99cc")
        self.cluster_label.grid(row=2, column=0, padx=10, pady=5)
        
        self.cluster_entry = tk.Entry(self.root)
        self.cluster_entry.grid(row=2, column=1, padx=10, pady=5)
        
        # Analyze button
        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.analyze)
        self.analyze_button.grid(row=3, column=0, columnspan=2, padx=10, pady=5)
        
        # Output Text widget with horizontal scrollbar
        self.output_text = scrolledtext.ScrolledText(self.root, width=98, height=30, bg="white", wrap=tk.NONE)
        self.output_text.grid(row=4, column=0, columnspan=2, padx=10, pady=5)
        
        self.horizontal_scrollbar = tk.Scrollbar(self.root, orient="horizontal", command=self.output_text.xview)
        self.horizontal_scrollbar.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.output_text.config(xscrollcommand=self.horizontal_scrollbar.set)
        
    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
    def analyze(self):
        try:
            percentage = float(self.percent_entry.get())
            k = int(self.cluster_entry.get())
            
            if not self.file_path:
                self.display_error("Error", "Please select a file.")
                return
            
            self.data = pd.read_csv(self.file_path)
            num_records = int(len(self.data) * (percentage / 100))
            sampled_data = self.data.sample(n=num_records, random_state=42)
            
            # Standardize the data
            ratings = sampled_data['IMDB Rating'].values.reshape(-1, 1)
            ratings_mean = np.mean(ratings)
            ratings_std = np.std(ratings)
            ratings_scaled = (ratings - ratings_mean) / ratings_std
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, init='random', random_state=42)
            kmeans.fit(ratings_scaled)
            
            # Get the cluster labels and assign them to the original dataframe
            sampled_data['Cluster'] = kmeans.labels_
            
            # Calculate the Euclidean distance of each point from its centroid
            distances = kmeans.transform(ratings_scaled)
            
            # Detect outliers
            outliers_threshold = 2  # You can adjust this threshold as needed
            outliers = np.max(distances, axis=1) > outliers_threshold
            # Display outlier records and cluster information
            self.display_information("Outlier Records", f"Outlier records:\n\n{sampled_data[outliers]}")
            self.display_information("Cluster Information", self.get_cluster_info(sampled_data, k))
            
            # Plot clusters
            self.plot_clusters(sampled_data, k)
            
        except ValueError:
            self.display_error("Error", "Invalid input. Please enter valid percentage and number of clusters.")

    def display_error(self, title, message):
        messagebox.showerror(title, message)

    def display_information(self, title, message):
        self.output_text.insert(tk.END, f"{title}\n\n{message}\n\n")

    def get_cluster_info(self, data, k):
        cluster_info = ""
        for cluster_num in range(k):
            cluster_info += f"\nCluster {cluster_num}:\n"
            cluster_data = data[data['Cluster'] == cluster_num]
            cluster_info += str(cluster_data[['Movie Name', 'IMDB Rating']]) + "\n"
        return cluster_info

    def plot_clusters(self, data, k):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots()
        for i in range(k):
            cluster_data = data[data['Cluster'] == i]
            ax.scatter(cluster_data['IMDB Rating'], np.zeros_like(cluster_data['IMDB Rating']), color=colors[i], label=f'Cluster {i}')
        ax.set_xlabel('IMDB Rating')
        ax.set_title('K-Means Clustering')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansClusteringApp(root)
    root.mainloop()