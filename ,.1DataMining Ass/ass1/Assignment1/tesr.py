import pandas as pd
from tkinter import filedialog, messagebox
import tkinter as tk
from functools import reduce 
import operator

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=(("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("Text files", "*.txt")))
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(0, file_path)

def prune(data, supp):
    df = data[data.supp_count >= supp] 
    return df

def count_itemset(transaction_df, itemsets):
    count_item = {}
    for item_set in itemsets:
        set_A = set(item_set)
        for row in transaction_df:
            set_B = set(row)
            if set_B.intersection(set_A) == set_A: 
                if item_set in count_item.keys():
                    count_item[item_set] += 1
                else:
                    count_item[item_set] = 1
    data = pd.DataFrame()
    data['item_sets'] = count_item.keys()
    data['supp_count'] = count_item.values()
    return data

def count_item(trans_items):
    count_ind_item = {}
    for row in trans_items:
        for i in range(len(row)):
            if row[i] in count_ind_item.keys():
                count_ind_item[row[i]] += 1
            else:
                count_ind_item[row[i]] = 1
    data = pd.DataFrame()
    data['item_sets'] = count_ind_item.keys()
    data['supp_count'] = count_ind_item.values()
    data['item_sets'] = data['item_sets'].astype(str)  # Convert 'item_sets' to strings
    data['supp_count'] = data['supp_count'].astype(int)  # Convert 'supp_count' to integers
    data = data.sort_values('item_sets')
    return data

def join(list_of_items):
    itemsets = []
    i = 1
    for entry in list_of_items:
        proceding_items = list_of_items[i:]
        for item in proceding_items:
            if(type(item) is str):
                if entry != item:
                    tuples = (entry, item)
                    itemsets.append(tuples)
            else:
                if entry[0:-1] == item[0:-1]:
                    tuples = entry+item[1:]
                    itemsets.append(tuples)
        i = i+1
    if(len(itemsets) == 0):
        return None
    return itemsets

def apriori(trans_data, supp=3):
    freq = pd.DataFrame()
    df = count_item(trans_data)
    while(len(df) != 0):
        df = prune(df, supp)
        if len(df) > 1 or (len(df) == 1 and int(df.supp_count >= supp)):
            freq = df
        itemsets = join(df.item_sets)
        if(itemsets is None):
            return freq
        df = count_itemset(trans_data, itemsets)
    return df

def calculate_conf(value1, value2):
    return round(int(value1)/int(value2) * 100, 2)

def strong_rules(freq_item_sets, threshold):

    confidences = {}
    for row in freq_item_sets.item_sets:
        for i in range(len(row)):
            for j in range(len(row)):
                if i != j:
                    itemset_count = freq_item_sets[freq_item_sets.item_sets == row].supp_count
                    item_count = count_item(trans_df)[count_item(trans_df).item_sets == row[i]].supp_count
                    conf = calculate_conf(int(itemset_count.iloc[0]), int(item_count.iloc[0]))
                    confidences[tuples] = conf

    conf_df = pd.DataFrame()
    conf_df['item_set'] = confidences.keys()
    conf_df['confidence'] = confidences.values()

    return conf_df[conf_df.confidence >= threshold]
def interesting_rules(freq_item_sets):
    lifts = {}
    prob_of_items = []
    for row in freq_item_sets.item_sets:
        num_of_items = len(row)
        prob_all = freq_item_sets[freq_item_sets.item_sets == row].supp_count / len(trans_df)
        for i in range(num_of_items):
            prob_of_items.append(count_item(trans_df)[count_item(trans_df).item_sets == row[i]].supp_count / len(trans_df))
        lifts[row] = round(float(prob_all / reduce(operator.mul, (np.array(prob_of_items)), 1)), 2)
        prob_of_items = []
    lifts_df = pd.DataFrame()
    lifts_df['Rules'] = lifts.keys()
    lifts_df['lift'] = lifts.values()
    return lifts_df

def run_association_rule_mining():
    file_path = entry_file_path.get()
    percentage = float(entry_percentage.get())
    min_support = float(entry_min_support.get())
    min_confidence = float(entry_min_confidence.get()) / 100  # Convert percentage to fraction

    # Read the transaction data
    transaction_df = pd.read_csv(file_path)
    transaction_df.index.rename('TID', inplace=True)
    trans_df = transaction_df.values.tolist()

    # Run Apriori algorithm
    freq_item_sets = apriori(trans_df, int(len(transaction_df) * percentage / 100))

    # Get strong association rules
    strong_rules_df = strong_rules(freq_item_sets, min_confidence)

    # Display the results
    results_message = f"File Path: {file_path}\nPercentage: {percentage}\nMin Support: {min_support}\nMin Confidence: {min_confidence}\n\n"
    results_message += "Frequent Item Sets:\n"
    results_message += str(freq_item_sets) + "\n\n"
    results_message += "Strong Association Rules:\n"
    results_message += str(strong_rules_df)

    messagebox.showinfo("Results", results_message)

# Create the main window
root = tk.Tk()
root.title("Association Rule Mining")

# File Path
label_file_path = tk.Label(root, text="File Path:")
label_file_path.grid(row=0, column=0)
entry_file_path = tk.Entry(root, width=50)
entry_file_path.grid(row=0, column=1)
button_browse = tk.Button(root, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2)

# Percentage
label_percentage = tk.Label(root, text="Percentage:")
label_percentage.grid(row=1, column=0)
entry_percentage = tk.Entry(root, width=10)
entry_percentage.grid(row=1, column=1)
label_percentage_percent = tk.Label(root, text="%")
label_percentage_percent.grid(row=1, column=2)

# Min Support Count
label_min_support = tk.Label(root, text="Min Support:")
label_min_support.grid(row=2, column=0)
entry_min_support = tk.Entry(root, width=10)
entry_min_support.grid(row=2, column=1)

# Min Confidence
label_min_confidence = tk.Label(root, text="Min Confidence:")
label_min_confidence.grid(row=3, column=0)
entry_min_confidence = tk.Entry(root, width=10)
entry_min_confidence.grid(row=3, column=1)
label_min_confidence_percent = tk.Label(root, text="%")
label_min_confidence_percent.grid(row=3, column=2)

# Run Button
button_run = tk.Button(root, text="Run", command=run_association_rule_mining)
button_run.grid(row=4, column=0, columnspan=3, pady=10)

root.mainloop()