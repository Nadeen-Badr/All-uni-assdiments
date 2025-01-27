import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('c.csv')

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

# Split the data into features (X) and target variable (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NaiveBayes:
    def __init__(self):
        self.class_prob = {}
        self.feature_prob = {}

    def fit(self, X, y):
        self.classes = y.unique()
        for cls in self.classes:
            cls_data = X[y == cls]
            self.class_prob[cls] = len(cls_data) / len(X)
            self.feature_prob[cls] = {}
            for feature in X.columns:
                self.feature_prob[cls][feature] = {}
                for value in X[feature].unique():
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
            for cls in self.classes:
                prob = self.class_prob[cls]
                for feature, value in row.items():
                    if value in self.feature_prob[cls][feature]:
                        prob *= self.feature_prob[cls][feature][value]
                if prob > max_prob:
                    max_prob = prob
                    pred_class = cls
            predictions.append(pred_class)
        return predictions

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        print("\nDecision Tree Classifier - Tree Structure:")
        print(self.tree)

    def predict(self, X):
        return [self.predict_instance(row, self.tree) for _, row in X.iterrows()]

    def predict_instance(self, instance, tree):
        if isinstance(tree, dict):
            attr, subtree = next(iter(tree.items()))
            return self.predict_instance(instance, subtree)
        else:
            return tree

    def build_tree(self, X, y):
        if len(y.unique()) == 1:
            return y.iloc[0]
        if len(X.columns) == 0:
            return y.value_counts().idxmax()
        best_feature, best_split = self.find_best_split(X, y)
        if best_split is None:
            return y.value_counts().idxmax()
        print(f"Best Split: {best_split}")
        subtree = {best_feature: {}}
        for value, data in best_split.groupby(best_feature):
            data = data.drop(best_feature, axis=1)
            subtree[best_feature][value] = self.build_tree(data, y.loc[data.index])
        return subtree

    def find_best_split(self, X, y):
        best_feature = None
        best_info_gain = -1
        best_split = None
        for feature in X.columns:
            info_gain, split = self.calculate_info_gain(X[feature], y)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_split = split
        print(f"Best Feature: {best_feature}, Best Info Gain: {best_info_gain}")
        print(f"Split: {best_split}")
        return best_feature, best_split

    def calculate_info_gain(self, feature, y):
        feature_entropy = self.calculate_entropy(feature)
        total_entropy = self.calculate_entropy(y)
        split_entropy = 0
        split = None
        for value in feature.unique():
            subset = y[feature == value]
            split_entropy += len(subset) / len(y) * self.calculate_entropy(subset)
        return total_entropy - split_entropy, split

    def calculate_entropy(self, data):
        entropy = 0
        total = len(data)
        for _, count in data.value_counts().items():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy

# Training and Evaluating Naive Bayes Classifier
nb_classifier = NaiveBayes()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Training and Evaluating Decision Tree Classifier
dt_classifier = DecisionTree()
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f'Naive Bayes Classifier Accuracy: {nb_accuracy}')
print(f'Decision Tree Classifier Accuracy: {dt_accuracy}')

# Compare the results of the two classifiers
if nb_accuracy > dt_accuracy:
    print('Naive Bayes Classifier performs better.')
else:
    print('Decision Tree Classifier performs better.')

# Printing tuples with predictions and feature values
nb_pred_tuples = list(zip(X_test.index, nb_predictions))
print("\nNaive Bayes Classifier predictions:")
for tpl in nb_pred_tuples:
    idx = tpl[0]
    prediction = tpl[1]
    features = X_test.loc[idx]
    print(f"Index: {idx}, Prediction: {prediction}, Features: {features.to_dict()}")

dt_pred_tuples = list(zip(X_test.index, dt_predictions))
print("\nDecision Tree Classifier predictions:")
for tpl in dt_pred_tuples:
    idx = tpl[0]
    prediction = tpl[1]
    features = X_test.loc[idx]
    print(f"Index: {idx}, Prediction: {prediction}, Features: {features.to_dict()}")