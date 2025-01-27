import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
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

# Naive Bayes Classifier
nb_classifier = CategoricalNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
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
    # Printing tuples with predictions
# Printing tuples with predictions and feature values
nb_pred_tuples = list(zip(X_test.index, nb_predictions))
print("Naive Bayes Classifier predictions:")
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
