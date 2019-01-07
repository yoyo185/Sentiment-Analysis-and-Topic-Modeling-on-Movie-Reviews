# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:07:18 2018

@author: xieyy
"""

# Import libraries
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import preprocessing
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import roc_curve, auc  

###############################################################################

#### Main function 
def main():
    #load the train data
    train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3) 
    #load the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3) 
    #call the function
    train_data_features = filter_out(train)
    # Classification method - knn,bayes and svm
    classification(train,test,train_data_features)


###data cleaning and text processing
def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
    
def filter_out(train):   
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list 
    for i in range(0, num_reviews):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        clean_train_reviews.append(review_to_words(train["review"][i]))

    ####bag of words
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 500) 
    
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    
    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()   
    return(train_data_features)

    
    
def classification(train,test,train_data_features):
    # Then make the validation set 20% of the entire
    test_size = 0.20
    # Set seed
    seed = 123
    X_train, X_validate, Y_train, Y_validate = train_test_split(train_data_features[0:len(train)], 
    train.sentiment, test_size=test_size, random_state=seed)
    # Transform df into Min-Max Normalization for later calculating
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(train_data_features[0:len(train)])
    #normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])


    # Select 10 folds
    num_folds = 10
    seed = 123
    scoring = 'accuracy'
    # Add each algorithm and its name to the model array
    models = []
    models.append(('Logistic Regression',LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('svm',svm.SVC()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RandomForest', RandomForestClassifier())) 
    
    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        # Do cross validation
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        # Print the result of these methods
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        

    # Validation test
    ######################################################
    #Make predictions on validation dataset in logistic regression
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    predictions1 = lr.predict(X_validate)
    # Print the accuracy for knn
    print('\n\n********For lr, we use test data to validate it :**********')
    print('The accurary_score of lr is:\n',accuracy_score(Y_validate, predictions1))
    print('The confusion_matrix of lr is:\n',confusion_matrix(Y_validate, predictions1))
    print('The classification of lr is:\n', classification_report(Y_validate, predictions1))

  
    ###summition
    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = [] 

    for i in range(0,num_reviews):
        clean_review = review_to_words( test["review"][i] )
        clean_test_reviews.append( clean_review )
    
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    
    # Use the random forest to make sentiment label predictions
    result = predictions1
    
    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    
    # Use pandas to write the comma-separated output file
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

# Call main function 
if __name__ == "__main__":   
    main()    
    