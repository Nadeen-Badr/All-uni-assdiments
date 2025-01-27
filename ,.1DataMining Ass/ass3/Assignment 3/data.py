import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('c.csv')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each categorical column and encode it
for col in ['gender', 'smoking_history']:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head())
# Convert the numerical columns into categorical ones
df['age_cat'] = pd.qcut(df['age'], q=5, labels=False)
df['bmi_cat'] = pd.qcut(df['bmi'], q=5, labels=False)
df['HbA1c_cat'] = pd.qcut(df['HbA1c_level'], q=3, labels=False)
df['glucose_cat'] = pd.qcut(df['blood_glucose_level'], q=5, labels=False)

# Drop the original numerical columns

# Drop the original numerical columns
df.drop(['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'], axis=1, inplace=True)

print(df.head())