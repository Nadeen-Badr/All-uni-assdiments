import csv
import math

# Load the CSV file into a list of dictionaries
data = []
with open('c.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Split the data into training and testing sets (for example, 80% training and 20% testing)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Implement the Bayesian classifier
def calculate_probability(data, attribute, value, target_class):
    """
    Calculate the conditional probability P(attribute=value | diabetes=target_class) from the data.
    
    Args:
    - data: List of dictionaries representing the dataset.
    - attribute: The attribute for which to calculate the probability.
    - value: The specific value of the attribute.
    - target_class: The target class ('0' or '1').

    Returns:
    - The conditional probability P(attribute=value | diabetes=target_class).
    """
    count = 0
    total = 0
    for row in data:
        """
        row[attribute] would be one of the attributes in your dataset,
        such as 'gender', 'age', 'hypertension', etc.
        value would be the specific value you're interested in for
        the attribute (e.g., 'Female' for 'gender', 80.0 for 'age', etc.).
        row['diabetes'] would be the 'diabetes' attribute in your dataset,
        which indicates whether the individual has diabetes (0 for no, 1 for yes).
        target_class would be the specific target class you're interested in
        for the 'diabetes' attribute (e.g., 0 for no diabetes, 1 for diabetes).
        So, the conditional statement is checking if the attribute value 
        (row[attribute]) is equal to a specific value (value) and if the
        'diabetes' attribute (row['diabetes']) is equal to a specific target class 
        (target_class) for each row in your dataset.
        """
        if row[attribute] == value and row['diabetes'] == target_class:
            count += 1
        if row['diabetes'] == target_class:
            total += 1
    return count / total if total > 0 else 0

def bayesian_classifier(train_data, test_data):
    """
    Implement the Bayesian classifier to classify test data instances.

    Args:
    - train_data: The training dataset.
    - test_data: The testing dataset.

    Returns:
    - A list of predicted class labels for the test instances.
    """
    predictions = []
    for test_row in test_data:
        probabilities = {}
        for target_class in ['0', '1']:  # Assuming '0' means no diabetes and '1' means diabetes
            probability = 1
            for attribute in test_row.keys():
                if attribute != 'diabetes':
                    value = test_row[attribute]
                    probability *= calculate_probability(train_data, attribute, value, target_class)
            probabilities[target_class] = probability
        prediction = max(probabilities, key=probabilities.get)
        predictions.append(prediction)
    return predictions

# Implement the decision tree classifier
def calculate_entropy(data):
    """
    Calculate the entropy of a dataset.

    Args:
    - data: List of dictionaries representing the dataset.

    Returns:
    - The entropy value.
    """
    total = len(data)
    counts = {}
    for row in data:
        if row['diabetes'] in counts:
            counts[row['diabetes']] += 1
        else:
            counts[row['diabetes']] = 1
    entropy = 0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy

def split_data(data, attribute, value):
    """
    Split the dataset based on a specific attribute value.

    Args:
    - data: List of dictionaries representing the dataset.
    - attribute: The attribute to split on.
    - value: The value of the attribute to split on.

    Returns:
    - A subset of the dataset.
    """
    subset = []
    for row in data:
        if row[attribute] == value:
            subset.append(row)
    return subset

def decision_tree_classifier(train_data, test_data):
    """
    Implement the decision tree classifier to classify test data instances.

    Args:
    - train_data: The training dataset.
    - test_data: The testing dataset.

    Returns:
    - A list of predicted class labels for the test instances.
    """
    def build_tree(data):
        if len(data) == 0:
            return None
        if len(set(row['diabetes'] for row in data)) == 1:
            return data[0]['diabetes']
        best_attribute = None
        best_gain = -1
        entropy = calculate_entropy(data)
        for attribute in data[0].keys():
            if attribute != 'diabetes':
                values = set(row[attribute] for row in data)
                for value in values:
                    subset = split_data(data, attribute, value)
                    info_gain = entropy
                    info_gain -= len(subset) / len(data) * calculate_entropy(subset)
                    if info_gain > best_gain:
                        best_gain = info_gain
                        best_attribute = attribute
                        best_value = value
        if best_gain == 0:
            return max(set(row['diabetes'] for row in data), key=[row['diabetes'] for row in data].count)
        subtree = {}
        for value in set(row[best_attribute] for row in data):
            subset = split_data(data, best_attribute, value)
            subtree[value] = build_tree(subset)
        return {best_attribute: subtree}

    def predict(tree, test_row):
        if isinstance(tree, str):
            return tree
        attribute = next(iter(tree))
        subtree = tree[attribute]
        attribute_value = test_row.get(attribute)
        if attribute_value not in subtree:
            return max(set(row['diabetes'] for row in data), key=[row['diabetes'] for row in data].count)
        return predict(subtree[attribute_value], test_row)

    tree = build_tree(train_data)
    predictions = []
    for test_row in test_data:
        predictions.append(predict(tree, test_row))
    return predictions

# Apply the classifiers on the testing set
bayes_predictions = bayesian_classifier(train_data, test_data)
decision_tree_predictions = decision_tree_classifier(train_data, test_data)

# Calculate the accuracy of the classifiers
def calculate_accuracy(predictions, actual):
    """
    Calculate the accuracy of the classifiers.

    Args:
    - predictions: A list of predicted class labels.
    - actual: The actual class labels.

    Returns:
    - The accuracy value.
    """
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]['diabetes']:
            correct += 1
    return correct / len(predictions)

bayes_accuracy = calculate_accuracy(bayes_predictions, test_data)
decision_tree_accuracy = calculate_accuracy(decision_tree_predictions, test_data)

# Print the accuracies of the classifiers
print(f'Bayesian Classifier Accuracy: {bayes_accuracy}')
print(f'Decision Tree Classifier Accuracy: {decision_tree_accuracy}')

# Compare the results of the two classifiers
if bayes_accuracy > decision_tree_accuracy:
    print('Bayesian Classifier performs better.')
else:
    print('Decision Tree Classifier performs better.')