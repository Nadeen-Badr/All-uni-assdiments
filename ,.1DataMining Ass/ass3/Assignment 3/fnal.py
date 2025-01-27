import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_prob = {}
        self.feature_prob = {}

    def fit(self, X, y):
        self.classes = y.unique() # Get unique classes from the target variable
        for cls in self.classes:
            cls_data = X[y == cls] # Get data corresponding to the current class
            # Calculate class probability: P(class) = count(class_data) / total_samples
            self.class_prob[cls] = len(cls_data) / len(X)
            self.feature_prob[cls] = {}# Initialize nested dictionary for feature probabilities of the current class
            for feature in X.columns:
                self.feature_prob[cls][feature] = {}# Initialize dictionary for current feature
                # Calculate probability for each unique value of the feature
                for value in X[feature].unique():# Calculate feature probability:
                   #P(feature=value|class) = (count(class_data with feature=value) + 1) / (count(class_data) + count(unique_values))
                   #Laplace smoothing rule 
                    self.feature_prob[cls][feature][value] = (len(cls_data[cls_data[feature] == value]) + 1) / (len(cls_data) + len(X[feature].unique()))
        print("Naive Bayes Classifier - Class Probabilities:")
        print(self.class_prob)
        print("\nNaive Bayes Classifier - Feature Probabilities:")
        print(self.feature_prob)

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            max_prob = -1
            pred_class = None
            # Calculate the probability for each class and find the class with maximum probability
            for cls in self.classes:
                prob = self.class_prob[cls]
                for feature, value in row.items():
                    if value in self.feature_prob[cls][feature]:
                         # Multiply feature probabilities for each feature value in the row
                        prob *= self.feature_prob[cls][feature][value]
                if prob > max_prob:
                    max_prob = prob
                    pred_class = cls
            predictions.append(pred_class)
        return predictions

class DecisionTree:
    def __init__(self, max_depth=3):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        self.tree = self.build_tree(X, y, depth)
        

    def predict(self, X):
        return [self.predict_instance(row, self.tree) for _, row in X.iterrows()]

    def predict_instance(self, instance, tree):
        '''Recursively traverses the decision tree to predict the label for a single instance instance.
If tree is a dictionary, it retrieves the attribute and its corresponding subtree and continues traversal.
If tree is a leaf node (a class label), it returns the label.'''
        if isinstance(tree, dict):
            attr, subtree = next(iter(tree.items()))
            attr_val = instance[attr]
            if attr_val in subtree:
                return self.predict_instance(instance, subtree[attr_val])
            
        else:
            return tree

    def build_tree(self, X, y, depth):
        """If all labels in y are the same, it returns that label.
        This means that all samples in this branch of the tree belong to the same class
        so no further split is needed."""
        if len(y.unique()) == 1:
            return y.iloc[0]
        """If the maximum depth (self.max_depth) is reached,
        it returns the most common label in y.
        This prevents the tree from growing too deep and overfitting."""
        if self.max_depth is not None and depth >= self.max_depth:
            return y.value_counts().idxmax()
        if len(X.columns) == 0:
            return y.value_counts().idxmax()
        best_feature, best_split = self.find_best_split(X, y)
        """If no valid split is found (best_split is None)
         it returns the most common label in y."""
        if best_split is None:
            return y.value_counts().idxmax()
        subtree = {best_feature: {}}
        for value, data in best_split:
            data = data.drop(best_feature, axis=1)
            subtree[best_feature][value] = self.build_tree(data, y.loc[data.index], depth + 1)
        return subtree

    def find_best_split(self, X, y):
        best_feature = None
        best_info_gain = -1
        print(f"finding best split.....")
        for feature in X.columns:
           
            info_gain = self.calculate_info_gain(X[feature], y)
            print(f"Feature - {feature}, Information Gain - {info_gain}")
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
       
        print(f"Best Feature - {best_feature}, Information Gain - {best_info_gain}")
        return best_feature, X.groupby(best_feature)

    def calculate_info_gain(self, feature, y):
        feature_entropy = self.calculate_entropy(feature)
        total_entropy = self.calculate_entropy(y)
        split_entropy = 0
        for value in feature.unique():
            subset = y[feature == value]
            split_entropy += len(subset) / len(y) * self.calculate_entropy(subset)
        return total_entropy - split_entropy

    def calculate_entropy(self, data):
        entropy = 0
        total = len(data)
        for _, count in data.value_counts().items():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV File Reader")

        tk.Label(self.root, text="Select CSV File:").pack()
        self.file_path_entry = tk.Entry(self.root, width=50)
        self.file_path_entry.pack()
        tk.Button(self.root, text="Browse", command=self.browse_file).pack()

        tk.Label(self.root, text="Percentage of Data to Read:").pack()
        self.data_percentage_entry = tk.Entry(self.root)
        self.data_percentage_entry.pack()

        tk.Label(self.root, text="Training Set Percentage:").pack()
        self.training_percentage_entry = tk.Entry(self.root)
        self.training_percentage_entry.pack()

        tk.Label(self.root, text="Test Set Percentage:").pack()
        self.test_percentage_entry = tk.Entry(self.root)
        self.test_percentage_entry.pack()

        tk.Button(self.root, text="Submit", command=self.process_data).pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.file_path_entry.insert(tk.END, file_path)

    def process_data(self):
        file_path = self.file_path_entry.get()
        data_percentage = float(self.data_percentage_entry.get())
        training_percentage = float(self.training_percentage_entry.get())
        test_percentage = float(self.test_percentage_entry.get())

        df = pd.read_csv(file_path)
        # Calculate the number of rows based on the percentage
        num_rows = int(len(df) * data_percentage / 100)
        df = df.sample(n=num_rows)

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Iterate over each categorical column and encode it
        for col in ['gender', 'smoking_history']:
            df[col] = label_encoder.fit_transform(df[col])

        # Convert the numerical columns into categorical ones
        df['age_cat'] = pd.qcut(df['age'], q=5, labels=False)
        df['bmi_cat'] = pd.qcut(df['bmi'], q=5, labels=False)
        df['HbA1c_cat'] = pd.qcut(df['HbA1c_level'], q=3, labels=False)
        df['glucose_cat'] = pd.qcut(df['blood_glucose_level'], q=5, labels=False)

        # Drop the original numerical columns
        df.drop(['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'], axis=1, inplace=True)

        X = df.drop('diabetes', axis=1)
        y = df['diabetes']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42)

        nb_classifier = NaiveBayes()
        nb_classifier.fit(X_train, y_train)
        nb_predictions = nb_classifier.predict(X_test)
        nb_correct = sum(y == pred for y, pred in zip(y_test, nb_predictions))
        nb_accuracy = nb_correct / len(y_test)

        # Training and Evaluating Decision Tree Classifier
        dt_classifier = DecisionTree()
        dt_classifier.fit(X_train, y_train)
        dt_predictions = dt_classifier.predict(X_test)
        dt_correct = sum(y == pred for y, pred in zip(y_test, dt_predictions))
        dt_accuracy = dt_correct / len(y_test)

        result_text = f'         Naive Bayes Classifier Accuracy: {nb_accuracy}\n'
        result_text += f'        Decision Tree Classifier Accuracy: {dt_accuracy}\n'

        if nb_accuracy > dt_accuracy:
            result_text += '     Naive Bayes Classifier performs better.'
        else:
            result_text += '     Decision Tree Classifier performs better.'

        result_text += "\nNaive Bayes and Decision Tree Classifier predictions (First 5):\n"
        count = 0
        nb_pred_tuples = list(zip(X_test.index, nb_predictions))
        dt_pred_tuples = list(zip(X_test.index, dt_predictions))

        for nb_tpl, dt_tpl in zip(nb_pred_tuples, dt_pred_tuples):
            nb_idx, nb_prediction = nb_tpl
            dt_idx, dt_prediction = dt_tpl

            nb_features = X_test.loc[nb_idx]
            dt_features = X_test.loc[dt_idx]

            result_text += f"Index: {nb_idx}, \n NB Prediction: {nb_prediction}, \n NB Features: {nb_features.to_dict()}, \n DT Prediction: {dt_prediction}, \n DT Features: {dt_features.to_dict()}\n"
            print (f"Index: {nb_idx}, \n NB Prediction: {nb_prediction}, \n NB Features: {nb_features.to_dict()}, \n DT Prediction: {dt_prediction}, \n DT Features: {dt_features.to_dict()}\n")
            count += 1
            if count >= 5:
                break

        result_window = tk.Toplevel(self.root)
        result_window.title("Results")
        result_label = tk.Label(result_window, text=result_text, justify=tk.LEFT)
        result_label.pack()

# Create the main window
root = tk.Tk()
app = App(root)
root.mainloop()
