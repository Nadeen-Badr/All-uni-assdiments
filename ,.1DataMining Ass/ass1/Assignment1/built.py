import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Read the CSV file into a DataFrame
df = pd.read_csv('bb.csv')

# Preprocess the data to get a list of transactions
transactions = df.groupby('TransactionNo')['Items'].apply(list).values.tolist()

# Convert the list of transactions into a transaction encoding format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Calculate minimum support count
min_support_count = 4 # Adjust as needed

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=min_support_count/len(df_encoded), use_colnames=True)

# Generate association rules
min_confidence = 0.5  # Adjust as needed
total_transactions = len(df_encoded)
frequent_itemsets['support_count'] = frequent_itemsets['support'] * total_transactions
# Output frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Output association rules
print("\nAssociation Rules:")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Output association rules with confidence
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])