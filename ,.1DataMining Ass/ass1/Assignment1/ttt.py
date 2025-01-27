import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from itertools import combinations, chain

def read_transactions(file_path, percentage):
    """
    Read transactions from a file (Excel, text, or CSV) and select a percentage of records.
    """
    # Read the file based on the file extension..
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep='\t', header=None)
    else:
        df = pd.read_csv(file_path)
    # Select a percentage of records random..based on usr input
    selected_rows = np.random.choice(df.index, size=int(len(df) * percentage), replace=False)
    df = df.iloc[selected_rows]
    # Group items by transaction number and convert to list
    
    transactions = df.groupby('TransactionNo')['Items'].apply(list).values.tolist()
    """
    TN,ITEM
    3,Jam
    3,Cookies
    4,Muffin
    5,Coffee
    5,Pastry
    ->
    transaction =[ [jam,cookies], [muffin ], [coffee ,pastry] ]
    """
    return transactions

def generate_candidate_itemsets(transactions, k):
    """
    Generate candidate itemsets of size k from the given transactions.
    K is the inti size in apriori
    intailly =1
    """
    candidate_itemsets = []
    #get the unique items
    unique_items = np.unique(list(chain(*transactions)))
     # Generate combinations of items
    for itemset in combinations(unique_items, k):
        candidate_itemsets.append(frozenset(itemset))
    return candidate_itemsets

def support_count(itemset, transactions):
    """
    Calculate the support count of an itemset in the transactions.
    Args:
    - itemset (frozenset): The itemset for which the support count is calculated.
    - transactions (list): A list of transactions, where each transaction is a list of items.

    Returns:
    - int: The support count of the itemset in the transactions
    """
    count = 0
    for transaction in transactions:
         # Check if the itemset is a subset of the transaction..
        if itemset.issubset(transaction):
            count += 1
    return count

def generate_frequent_itemsets(transactions, min_support, k):
    """
    Generate frequent itemsets of size k with support >= min_support.
    """
    frequent_itemsets = []
    candidate_itemsets = generate_candidate_itemsets(transactions, k)
    for itemset in candidate_itemsets:
        # Calculate the support count of the itemset in the transactions..
        support = support_count(itemset, transactions)
        
        if support >= min_support: 
            # If it is, add the itemset and its support count to the frequent_itemsets list..
            frequent_itemsets.append((itemset, support))
    return frequent_itemsets

def apriori(transactions, min_support, min_confidence):
    """
    Apriori algorithm to find all frequent itemsets with support >= min_support.
    """
    k = 1
    frequent_itemsets = []
    # Generate frequent itemsets of increasing size until no more can be found..
    while True:
        current_frequent_itemsets = generate_frequent_itemsets(transactions, min_support, k)
        # If no frequent itemsets of size k were found, exit the loop...
        if not current_frequent_itemsets:
            break
        # Add the current frequent itemsets to the list of all frequent itemsets..
        frequent_itemsets.extend(current_frequent_itemsets)
        k += 1

    association_rules = []
    for itemset, support in frequent_itemsets:
        # Only consider itemsets with more than one item
        if len(itemset) > 1:
             # Generate all possible combinations of antecedent and consequent..
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    # Calculate the confidence of the rule
                    confidence = support_count(itemset, transactions) / support_count(antecedent, transactions)
                    # If the confidence is greater than or equal to min_confidence, add the rule to the list
                    if confidence >= min_confidence:
                        association_rules.append((antecedent, consequent, confidence))

    return frequent_itemsets, association_rules

def run_analysis(file_path, min_support, min_confidence, percentage):
    transactions = read_transactions(file_path, percentage)
    frequent_itemsets, association_rules = apriori(transactions, min_support, min_confidence)

    # Output frequent itemsets
    frequent_itemsets_str = "Frequent Itemsets:\n"
    for itemset, support in frequent_itemsets:
        frequent_itemsets_str += f"Itemset: {itemset}, Support: {support}\n"

    # Output association rules
    association_rules_str = "\nAssociation Rules:\n"
    for antecedent, consequent, confidence in association_rules:
        association_rules_str += f"Rule: {antecedent} => {consequent}, Confidence: {confidence}\n"

    return frequent_itemsets_str, association_rules_str
"""
    Open a file dialog to select a file path and update the entry widget with the selected file path.
    """
def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=(("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("CSV files", "*.csv")))
    entry.delete(0, tk.END)
    entry.insert(0, file_path)
"""
    Create a GUI for running the Apriori algorithm with user-specified parameters.
    """
def run_gui():
    def run_analysis_wrapper():
        file_path = entry_file_path.get()
        min_support = int(entry_min_support.get())
        min_confidence = float(entry_min_confidence.get()) / 100
        percentage = float(entry_percentage.get()) / 100

        if not file_path:
            messagebox.showerror("Error", "Please select a file.")
            return

        try:
            frequent_itemsets, association_rules = run_analysis(file_path, min_support, min_confidence, percentage)
            text_output.delete(1.0, tk.END)
            text_output.insert(tk.END, frequent_itemsets + association_rules)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Create the main window
    root = tk.Tk()
    root.title("Apriori Algorithm GUI")

    # Create file selection button and entry
    label_file_path = tk.Label(root, text="Select File:")
    label_file_path.grid(row=0, column=0)
    entry_file_path = tk.Entry(root, width=50)
    entry_file_path.grid(row=0, column=1)
    button_browse = tk.Button(root, text="Browse", command=lambda: browse_file(entry_file_path))
    button_browse.grid(row=0, column=2)

    # Create min support input
    label_min_support = tk.Label(root, text="Min Support Count:")
    label_min_support.grid(row=1, column=0)
    entry_min_support = tk.Entry(root)
    entry_min_support.grid(row=1, column=1)

    # Create min confidence input
    label_min_confidence = tk.Label(root, text="Min Confidence(%) :")
    label_min_confidence.grid(row=2, column=0)
    entry_min_confidence = tk.Entry(root)
    entry_min_confidence.grid(row=2, column=1)

    # Create percentage input
    label_percentage = tk.Label(root, text="Percentage of Data to Read (%):")
    label_percentage.grid(row=3, column=0)
    entry_percentage = tk.Entry(root)
    entry_percentage.grid(row=3, column=1)

    # Create run button
    button_run = tk.Button(root, text="Run Analysis", command=run_analysis_wrapper)
    button_run.grid(row=4, column=0, columnspan=3)

    # Create output text area
    text_output = tk.Text(root, height=20, width=80)
    text_output.grid(row=5, column=0, columnspan=3)

    root.mainloop()

run_gui()