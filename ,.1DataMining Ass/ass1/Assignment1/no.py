import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from itertools import combinations

# Function to read data from file
def read_data():
    global transaction_df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Text files", "*.txt")])
    if file_path:
        transaction_df = pd.read_csv(file_path)
        # Optionally, display the number of records in the file
        messagebox.showinfo("File Information", f"Number of records in file: {len(transaction_df)}")

# Function to perform Apriori algorithm
def apriori(trans_data, min_support):
    freq = pd.DataFrame()
    
    df = count_item(trans_data)
   
    while(len(df) != 0):
        
        df = prune(df, min_support)
    
        if len(df) > 1 or (len(df) == 1 and int(df.supp_count >= min_support)):
            freq = df
        
        itemsets = join(df.item_sets)
    
        if(itemsets is None):
            return freq
    
        df = count_itemset(trans_data, itemsets)
    return df

# Function to generate strong association rules
def generate_rules(freq_item_sets, threshold):

    confidences = {}
    for row in freq_item_sets.item_sets:
        for i in range(len(row)):
            for j in range(len(row)):
                 if i != j:
                    tuples = (row[i], row[j])
                    conf = calculate_conf(freq_item_sets[freq_item_sets.item_sets == row].supp_count, count_item(trans_df)[count_item(trans_df).item_sets == row[i]].supp_count)
                    confidences[tuples] = conf

        
    conf_df = pd.DataFrame()
    conf_df['item_set'] = confidences.keys()
    conf_df['confidence'] = confidences.values()

    return conf_df[conf_df.confidence >= threshold]

# Function to handle analysis
def analyze():
    try:
        percentage = int(percentage_entry.get())
        min_support = int(support_entry.get())
        min_confidence = int(confidence_entry.get())
        
        # Sample code to read a percentage of the data
        num_records = len(transaction_df)
        num_records_to_read = int(num_records * (percentage / 100))
        sample_data = transaction_df.head(num_records_to_read)
        
        # Perform Apriori algorithm on sample_data
        frequent_item_sets = apriori(sample_data, min_support)
        
        # Generate strong association rules
        strong_rules = generate_rules(frequent_item_sets, min_confidence)
        
        # Display outputs (frequent item sets and association rules)
        # For simplicity, let's just print them for now
        print("Frequent Item Sets:")
        print(frequent_item_sets)
        
        print("\nStrong Association Rules:")
        print(strong_rules)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main application window
app = tk.Tk()
app.title("Data Mining GUI")

# Create and place widgets
tk.Label(app, text="Select File:").pack()
tk.Button(app, text="Browse", command=read_data).pack()

percentage_label = tk.Label(app, text="Percentage of Data to Read:")
percentage_label.pack()
percentage_entry = tk.Entry(app)
percentage_entry.pack()

support_label = tk.Label(app, text="Minimum Support Count:")
support_label.pack()
support_entry = tk.Entry(app)
support_entry.pack()

confidence_label = tk.Label(app, text="Minimum Confidence (%):")
confidence_label.pack()
confidence_entry = tk.Entry(app)
confidence_entry.pack()

analyze_button = tk.Button(app, text="Analyze", command=analyze)
analyze_button.pack()

# Start the main event loop
app.mainloop()