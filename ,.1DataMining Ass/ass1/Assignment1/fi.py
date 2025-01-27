import tkinter as tk
from tkinter import filedialog, messagebox

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=(("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("Text files", "*.txt")))
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(0, file_path)

def run_association_rule_mining():
    file_path = entry_file_path.get()
    percentage = float(entry_percentage.get())
    min_support = float(entry_min_support.get())
    min_confidence = float(entry_min_confidence.get()) / 100  # Convert percentage to fraction

    # Add your association rule mining code here
 
    messagebox.showinfo("Results", f"File Path: {file_path}\nPercentage: {percentage}\nMin Support: {min_support}\nMin Confidence: {min_confidence}")

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