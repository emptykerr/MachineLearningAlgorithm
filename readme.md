# kNN Classifier

This program implements a k-nearest neighbors (kNN) classifier for classification tasks. 
It reads training and test data from CSV files, scales the features using min-max feature scaling,
performs kNN classification, and outputs the results to a CSV file.

## Prerequisites

- Python 3.x
- NumPy (pip install numpy)
- Pandas (pip install pandas)

# Usage
## Run the program with the following command: 
python knn.py <training data file> <test data file> <output file> <k>

Replace <training data file>, <test data file>, <output file>, and <k> with the appropriate values:

    <training data file>: Path to the CSV file containing the training data.
    <test data file>: Path to the CSV file containing the test data.
    <output file>: Path to the output CSV file where the predictions will be stored.
    <k>: Value of k for the kNN algorithm. Must be a positive integer greater than 0.

Example:
python knn.py train_data.csv test_data.csv output.csv 3



# Decision Tree Classifier

This script implements a decision tree classifier using entropy as 
the impurity measure and information gain as the criteria for splitting.
It builds the decision tree using the provided training data and outputs 
the tree structure to a file.

## Prerequisites

- Python 3.x
- NumPy (pip install numpy)
- Pandas (pip install pandas)

# Usage
## Run the program with the following command: 
python DecisionTree.py <training data file>  <output file> 

Replace <training data file>, <output file> with the appropriate values:

    <training data file>: Path to the CSV file containing the training data.
    <output file>: Path to the output CSV file where the predictions will be stored.

Example:
python DecisionTree.py train_data.csv output.csv 

The decision tree algorithm uses entropy and information gain to construct the tree. It stops growing the tree when the information gain from a split is less than 0.00001.

Resources:
https://www.youtube.com/watch?v=NxEHSAfFlK8
https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
https://realpython.com/knn-python/
https://www.w3schools.com/python/python_ml_knn.asp
