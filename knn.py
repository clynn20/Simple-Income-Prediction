import numpy as np
import time
import statistics

# find and return the index of the k examples nearest to the query point. 
# here, nearest is defined as having the lowest Euclidean distance. 
# example_set is a n x d matrix of examples where each row corresponds to a single d-dimensional example
# query is a 1 x d vector representing a single example
# k is the number of neighbors to return
# idx_of_nearest is a k x 1 list of indices for the nearest k neighbors of the query point
def get_nearest_neighbors(example_set, query, k):
    
    difference = example_set - query
    # calculate the euclidean distance between every points in sample set and query point 
    many_norms = np.linalg.norm(difference, axis=1)
    # sort the norm array and get index
    # k x 1 vector contains norm of each point in sample to query point 
    # [norm0, norm1, norm2, ...]
    idx_of_sort_norms = np.argsort(many_norms, kind='heapsort')
    # return the first k nearest neighbors index
    # [idx0, idx1, idx2, ...]
    idx_of_nearest = idx_of_sort_norms[0:k]
    return idx_of_nearest


# run a knn classifier on the query point
# examples_x is a n x d matrix of examples where each row corresponds to a single d-dimensional example
# examples_y is a n x 1 vector of example class labels
# query is a 1 x d vector representing a single example
# k is the number of neighbors to return
# predicted_label is either 0 or 1 corresponding to the predicted class of query based on the neighbors
def knn_classify_point(examples_x, examples_y, query, k):
    # get the k_nearest neighbors
    k_nearest = get_nearest_neighbors(examples_x, query, k)
    # array to store the income label
    k_labels = np.array([])
    for i in range(len(k_nearest)):
        # add the corresponding income label to k_labels array
        k_labels = np.append(k_labels, examples_y[k_nearest[i],0])
    # get the value that appears the most in k_labels aka return the majority class of these neighbors as the prediction
    predicted_label = statistics.mode(k_labels)
    return predicted_label

# run k-fold cross validation on our training data
# train_x is a n x d matrix of examples where each row corresponds to a single d-dimensional example
# train_y is a n x 1 vector of examples class labels
# avg_val_acc is the average validation accuracy across the folds
# var_val_acc is the variance of validation accuracy across the folds
def cross_validation(train_x, train_y, num_folds=4, k=1):
    #split the train data into 4 folds
    x_folds = np.split(train_x, num_folds)
    y_folds = np.split(train_y, num_folds)
    accuracy_set = np.array([])
    
    for i in range(num_folds):
        #pick the test set
        test_x_set = x_folds[i]
        test_y_set = y_folds[i]
        
        # leave out the test set and combine the rest of folds into one train set
        train_x_set = np.array([])
        train_y_set = np.array([])
        train_x_set = np.vstack([fold for j, fold in enumerate(x_folds) if j!=i])
        train_y_set = np.vstack([fold for j, fold in enumerate(y_folds) if j!=i])
        
        k = min(k,len(train_y_set))
        
        #predict labels
        predict_labels = predict(train_x_set, train_y_set, test_x_set, k)
        
        #calculate the accuracy
        accuracy = compute_accuracy(test_y_set, predict_labels)
        accuracy_set = np.append(accuracy_set, accuracy)
        
    avg_val_acc = np.mean(accuracy_set)
    var_val_acc = np.var(accuracy_set)
    return avg_val_acc, var_val_acc



# run a knn classifier on every query in a matrix of queries
# examples_x is a n x d matrix of examples where each row corresponds to a single d-dimensional example
# examples_y is a n x 1 vector of example class labels
# queries_x is a m x d matrix representing a set of queries
# k is the number of neighbors to return
# predicted_y is a m x 1 vector of predicted class labels
def predict(examples_x, examples_y, queries_x, k):
    # for each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_x, examples_y, query, k) for query in queries_x]
    return np.array(predicted_y, dtype=np.int64)[:, np.newaxis]


# compute accuracy
# true_y is a n x 1 vector where each value corresponds to the true label of an example data point
# predicted_y is a n x 1 vector where each value corresponds to the predicted label of an example
# accuracy is the fraction of predicted labels that match the true labels
def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy


# load data from csv file
def load_data():
    # read in data exclude first row(attribute name) and first column (id)
    # train_data is 2d array with 8000 rows and 86 columns
    train_data = np.genfromtxt('train.csv', delimiter=',')[1:,1:]
    # train_x is 2d array with 8000 rows and 85 columns, exclude the income column
    train_x = train_data[:,:-1]  
    train_y = train_data[:, -1] 
    # train_y is 2d array with 8000 rows and 1 column(income), aka n x 1 column vector
    train_y = train_y[:, np.newaxis]
    # test_x is 2d array with 2000 rows and 85 columns, exclude the id column and didn't contain income column
    test_x = np.genfromtxt('test_pub.csv', delimiter=',')[1:,1:]
    return train_x, train_y, test_x
    
    
def main():
    # load training and test data as numpy matrices
    train_x, train_y, test_x = load_data()
    
    # search over possible settings of k 
    print("Performing 4-fold cross validation")
    for k in [1,3,5,7,9,99,999,8000]:
        t0 = time.time()
        # compute train accuracy using whole set
        pred_y = predict(train_x, train_y, train_x, k)
        train_acc = compute_accuracy(train_y, pred_y)
        
        # compute 4-fold cross validation accuracy
        avg_val_acc, var_val_acc = cross_validation(train_x, train_y, 4, k)
        
        t1 = time.time()
        print(f"k = {k:5d} -- train acc = {train_acc*100:.2f} val acc = {avg_val_acc*100:.2f} ({var_val_acc*100:.4f}) exe_time = {t1-t0:.2f}")
    
    # submission to kaggle
    # set the best k value and then run on the test set
    best_k = 30
        
    # make prediction on test set
    pred_test_y = predict(train_x, train_y, test_x, best_k)
        
    # add index and header then save to file
    test_out = np.concatenate((np.expand_dims(np.array(range(2000), dtype=np.int64), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted_csv', test_out, fmt='%s', delimiter=',')
        
    
    
if __name__ == "__main__":
    main()