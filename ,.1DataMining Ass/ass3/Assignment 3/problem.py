
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np


# Step 1: Read the dataset
def read_dataset(file_path, percentage):
    dataset = pd.read_csv(file_path)

    # Calculate the number of records to use based on the percentage provided by the user
    total_records = len(dataset)
    records_to_use = int((percentage / 100) * total_records)

    # Extract the specified percentage of records from the dataset
    dataset = dataset.head(records_to_use)

    return dataset

# Step 2: Split dataset into training and testing sets
def split_dataset(dataset, test_size):
    train_data = dataset.sample(frac=1-test_size, random_state=42)
    test_data = dataset.drop(train_data.index)
    return train_data, test_data

# Step 3: Preprocess the data
def preprocess_data(training_data, testing_data):
    # Separate features and labels
    X_train = training_data.drop(columns=['diabetes']).values
    y_train = training_data['diabetes'].astype(int).values
    X_test = testing_data.drop(columns=['diabetes']).values
    y_test = testing_data['diabetes'].astype(int).values

    # Encode categorical variables
    X_train, X_test = encode_categorical_variables(X_train, X_test)

    # Handle missing values
    X_train, X_test = handle_missing_values(X_train, X_test)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Encode categorical variables
def encode_categorical_variables(X_train, X_test):
    gender_encoder = LabelEncoder()
    smoking_encoder = LabelEncoder()

    # Encode 'gender' column
    X_train[:, 0] = gender_encoder.fit_transform(X_train[:, 0])
    X_test[:, 0] = gender_encoder.transform(X_test[:, 0])

    # Encode 'smoking_history' column
    X_train[:, 4] = smoking_encoder.fit_transform(X_train[:, 4])
    X_test[:, 4] = smoking_encoder.transform(X_test[:, 4])

    return X_train, X_test

# Handle missing values
def handle_missing_values(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Calculate class probabilities
        for cls in self.classes:
            self.class_probabilities[cls] = np.sum(y == cls) / n_samples

        # Calculate feature probabilities
        for cls in self.classes:
            self.feature_probabilities[cls] = {}
            for feature_idx in range(n_features):
                feature_values = X[y == cls, feature_idx]
                self.feature_probabilities[cls][feature_idx] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values)
                }

    def _calculate_probability(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _calculate_class_probability(self, x, cls):
        class_probability = self.class_probabilities[cls]
        for feature_idx, value in enumerate(x):
            mean = self.feature_probabilities[cls][feature_idx]['mean']
            std = self.feature_probabilities[cls][feature_idx]['std']
            class_probability *= self._calculate_probability(value, mean, std)
        return class_probability

    def predict(self, X):
        predictions = []
        for x in X:
            class_probabilities = {
                cls: self._calculate_class_probability(x, cls)
                for cls in self.classes
            }
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)
        return np.array(predictions)


# Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small value to avoid log(0)
        return entropy

    def _information_gain(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        n_total = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        if n_left == 0 or n_right == 0:
            return 0
        entropy_parent = self._entropy(y)
        entropy_left = self._entropy(y[left_mask])
        entropy_right = self._entropy(y[right_mask])
        information_gain = entropy_parent - (n_left / n_total) * entropy_left - (n_right / n_total) * entropy_right
        return information_gain

    def _find_best_split(self, X, y):
        best_information_gain = -1
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                information_gain = self._information_gain(X, y, feature_idx, threshold)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        if len(unique_classes) == 1 or n_samples < 2 or depth == self.max_depth:
            return (None, None, None, None, unique_classes[0])  # Return the class label of the majority class
        best_feature_idx, best_threshold = self._find_best_split(X, y)
        if best_feature_idx is None:
            return (None, None, None, None, unique_classes[0])  # Return the class label of the majority class
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return (best_feature_idx, best_threshold, left_subtree, right_subtree, None)

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_instance(self, x, tree):
        if isinstance(tree, int):
            return tree
        feature_idx, threshold, left_subtree, right_subtree, class_label = tree
        if class_label is not None:
            return class_label
        if x[feature_idx] <= threshold:
            return self._predict_instance(x, left_subtree)
        else:
            return self._predict_instance(x, right_subtree)

    def predict(self, X):
        return np.array([self._predict_instance(x, self.tree) for x in X])

# Main function to orchestrate the process
def main():
    root = tk.Tk()
    root.title("Diabetes Classifier")
    root.geometry("400x450")

    # Function to handle file browse button click
    def browse_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        file_path_var.set(file_path)

    # File path entry
    file_path_var = tk.StringVar()
    file_path_label = tk.Label(root, text="File Path:")
    file_path_label.pack()
    file_path_entry = tk.Entry(root, textvariable=file_path_var)
    file_path_entry.pack()


# Browse file button
    browse_button = tk.Button(root, text="Browse", command=browse_file)
    browse_button.pack()

    # Percentage entry
    percentage_var = tk.DoubleVar()
    percentage_label = tk.Label(root, text="Percentage to Use:")
    percentage_label.pack()
    percentage_entry = tk.Entry(root, textvariable=percentage_var)
    percentage_entry.pack()

    # Output text area
    output_text = tk.Text(root, height=10, width=50)
    output_text.pack()

    # Run button
    run_button = tk.Button(root, text="Run Classifier", command=lambda: run_classifier(file_path_var.get(), percentage_var.get(), output_text))
    run_button.pack()

    root.mainloop()

def run_classifier(file_path, percentage, output_text):
    try:
        percentage = float(percentage)
        dataset= read_dataset(file_path, percentage)

        # User inputs
        test_size = 0.25  # Percentage of data to use for testing

        # Step 2: Divide the dataset into training and testing sets
        training_data, testing_data = split_dataset(dataset, test_size)

        # Step 3: Preprocess the data
        X_train, X_test, y_train, y_test = preprocess_data(training_data, testing_data)

        # Print the shapes of the resulting datasets
        output_text.insert(tk.END, f"Training set size: {len(X_train)}, {len(y_train)}\n")
        output_text.insert(tk.END, f"Testing set size: {len(X_test)}, {len(y_test)}\n")

        # Step 4: Train and evaluate classifiers
        bayesian_classifier = NaiveBayesClassifier()
        bayesian_classifier.fit(X_train, y_train)
        bayesian_predictions = bayesian_classifier.predict(X_test)
        bayesian_accuracy = accuracy_score(y_test, bayesian_predictions)
        output_text.insert(tk.END, f"Accuracy of Bayesian classifier: {bayesian_accuracy}\n")

        decision_tree_classifier = DecisionTreeClassifier(max_depth=5)  # Example depth, you can adjust it
        decision_tree_classifier.fit(X_train, y_train)
        decision_tree_predictions = decision_tree_classifier.predict(X_test)
        decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
        output_text.insert(tk.END, f"Accuracy of Decision Tree classifier: {decision_tree_accuracy}\n")

        # Output each row in test along with its actual and predicted labels
        output_text.insert(tk.END, "\nTest Data Predictions:\n")
        for i, (idx, row) in enumerate(testing_data.iterrows()):
            output_text.insert(tk.END, f"Test Row {i+1}\n")
            output_text.insert(tk.END, f"Actual Label: {row['diabetes']}\n")

            
            output_text.insert(tk.END, f"Predicted Label (Bayesian): {bayesian_predictions[i]}\n")
            output_text.insert(tk.END, f"Predicted Label (Decision Tree): {decision_tree_predictions[i]}\n\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()
