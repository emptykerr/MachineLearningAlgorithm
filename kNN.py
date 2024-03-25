import sys
import numpy as np
import pandas as pd

# Helper function to take the list of nearest neighbours and returns the predicted class
def predict(nearest_neighbours):
    classes = {}
    for nn in nearest_neighbours: # iterating through the nearest neighbours
        class_label = nn[1] # getting the class label of the nearest neighbour
        if class_label in classes: # if the class label is in the classes dictionary
            classes[class_label] = classes[class_label] + 1 # increment the occurrences of the class label
        else:
            classes[class_label] = 1 # if the class label is not in the classes dictionary, then set the occurrences to 1

    # Sorts a dict of {class: occurrences} by highest occurrences to lowest and returning the first element
    # (the class with the highest occurrences)
    max_occurrences = 0
    predicted_class = None
    for class_label, occurrences in classes.items(): # iterating through the dictionary
        if occurrences > max_occurrences: # if the occurrences of the class is greater than the max_occurrences
            max_occurrences = occurrences   # then set the max_occurrences to the occurrences of the class
            predicted_class = class_label   # set the predicted class to the class_label
    return predicted_class  

# Returns sqrt(sum((x_i-y_i)^2))
def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2)) 


# KNN algorithm
def knn(test_X, test_y, train_X, train_y, k):
    predictions = []
    for i, a in enumerate(test_X): # iterating through the test set
        test_class = test_y[i] # getting the class of the test set
        distances = [(euclidean(a, b), train_class) for b, train_class in zip(train_X, train_y)] # calculating the distance between the test set and the training set
        distances.sort() # sorting the distances
        nearest_neighbours = distances[:k] # getting the k nearest neighbours
        predicted_class = predict(nearest_neighbours) # predicting the class of the test set
        dist_values = [d[0] for d in nearest_neighbours] # Extract distances
        predictions.append((test_class, predicted_class, *dist_values)) # appending the index, predicted class and the actual class to the predictions list
    return predictions

# Min-max feature scaling
def normalise(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val) 

# Reading data using Pandas
def read_data(data):
    df = pd.read_csv(data)
    X = df.iloc[:, :-1].values  # selecting all rows and all columns except the last one
    y = df.iloc[:, -1].values
    return X, y

def write_output(predictions, output):
    df_output = pd.DataFrame(predictions, columns=["Original_Class", "Predicted_Class", *["Distance" + str(i+1) for i in range(k)]])
    df_output.to_csv(output, index=False)

def main():
    if len(sys.argv) != 5:
        raise ValueError("Usage: python kNN.py <training data file> <test data file> <output file> <k>")
    
    # Extract command-line arguments
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    k = int(sys.argv[4])
        
    # Validate k
    if k < 1:
        raise ValueError("k must be greater than 0 for kNN algorithm")
        

    # Read data from files and split into X and y values for training and testing data 
    training_X, training_y = read_data(train_file)
    test_X, test_y = read_data(test_file)

    # Min-max feature scaling on the training and testing data using the training data's max and min values 
    train_max, train_min = np.max(training_X, axis=0), np.min(training_X, axis=0)
    training_X = normalise(training_X, train_max, train_min)
    test_X = normalise(test_X, train_max, train_min)

    # Run kNN algorithm on the test data
    test_predictions = knn(test_X, test_y, training_X, training_y, k)
    test_acc = sum(1 for actual, pred, *_ in test_predictions if pred == actual) / len(test_y)
    print(f"Testing Prediction Accuracy: {test_acc * 100:.2f}%")

    # Write the output to a file
    write_output(test_predictions, output_file)
   
if __name__ == "__main__":
    main()