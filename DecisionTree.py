import sys
import pandas as pd
import numpy as np

class Node:
    def __init__(self, split_feature=None, edges=None, class_label=None, information_gain=None, entropy=None, class_counters=None):
        self.split_feature = split_feature
        self.edges = edges if edges else {}
        self.class_label = class_label
        self.information_gain = information_gain
        self.entropy = entropy
        self.class_counters = class_counters

# Function to calculate entropy
def calculate_entropy(data):
    # Calculate class probabilities
    class_probabilities = data['class'].value_counts(normalize=True)
    
    # Calculate entropy
    entropy = -sum(class_probabilities * np.log2(class_probabilities))
    
    return entropy

# Function to calculate information gain
def calculate_information_gain(data, feature):
    # Calculate total entropy
    total_entropy = calculate_entropy(data)
    
    # Calculate weighted entropy for each unique value of the feature
    weighted_entropy = 0
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    
    # Calculate information gain
    information_gain = total_entropy - weighted_entropy
    
    return information_gain

# Function to build the decision tree
def build_decision_tree(data, features):
    # Base case: if all instances belong to the same class, return a leaf node
    if len(data['class'].unique()) == 1:
        class_counters = data['class'].value_counts().to_dict()
        return Node(class_label=data['class'].iloc[0], class_counters=class_counters)
    
    # Base case: if there are no more features to split on, return a leaf node with the majority class
    if len(features) == 0:
        majority_class = data['class'].value_counts().idxmax()
        class_counters = data['class'].value_counts().to_dict()
        return Node(class_label=majority_class, class_counters=class_counters)
    
    # Find the best feature to split on based on information gain
    best_feature = max(features, key=lambda feature: calculate_information_gain(data, feature))
    information_gain = calculate_information_gain(data, best_feature)
    entropy = calculate_entropy(data)
    
    # Check if information gain is less than the threshold
    if information_gain < 0.00001:
        majority_class = data['class'].value_counts().idxmax()
        class_counters = data['class'].value_counts().to_dict()
        return Node(class_label=majority_class, class_counters=class_counters)
    
    # Create a split node with the best feature
    node = Node(split_feature=best_feature, information_gain=information_gain, entropy=entropy)
    
    # Recursively build the decision tree for each unique value of the best feature
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subset_features = features.copy()
        subset_features.remove(best_feature)
        node.edges[value] = build_decision_tree(subset, subset_features)
    
    return node

# Function to classify a new instance using the decision tree
def classify_instance(instance, tree):
    if tree.class_label is not None:
        return tree.class_label
    
    split_feature = tree.split_feature
    value = instance[split_feature]
    
    if value not in tree.edges:
        return None
    
    subtree = tree.edges[value]
    return classify_instance(instance, subtree)

# Function to write decision tree to file
def write_tree_to_file(node, output_file, indent=''):
    if node.class_label is not None:
        output_file.write(f"{indent}leaf: {node.class_counters}\n")
    else:
        output_file.write(f"{indent}{node.split_feature} (IG: {node.information_gain}, Entropy: {node.entropy})\n")
        for value, subtree in node.edges.items():
            output_file.write(f"{indent}-- {node.split_feature} == {value} --\n")
            write_tree_to_file(subtree, output_file, indent + "    ")

# Main function
def main():
    # Read command line arguments
    train_file = sys.argv[1]
    output_file = sys.argv[2]

    if len(sys.argv) != 3:
        raise ValueError("Usage: python DecisionTree.py <training data file> <output file>")
    
    # Read training data
    data = pd.read_csv(train_file)
    
    # Get list of features
    features = list(data.columns[:-1]) # All columns except the last one
    
    # Build the decision tree
    decision_tree = build_decision_tree(data, features)

     # Calculate accuracy on training data
    correct_predictions = 0
    for _, instance in data.iterrows():
        predicted_class = classify_instance(instance, decision_tree)
        if predicted_class == instance['class']:
            correct_predictions += 1
    accuracy = correct_predictions / len(data)
    
    # Print accuracy
    print(f"Accuracy: {accuracy}")
    
    # Write decision tree to output file
    with open(output_file, 'w') as f:
        write_tree_to_file(decision_tree, f)

if __name__ == "__main__":
    main()
