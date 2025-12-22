# Training the machine learning model on 80% of my data.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import joblib


# Converting the .csv files (categorised) to pandas DataFrames.
test_data = pd.read_csv("data/test_data.csv")
train_data = pd.read_csv("data/train_data.csv")


# Must convert the text input data (the transaction descriptions) from a string to a numerical vector for the model to understand.
# Using TF-IDF vectoriser. Considers the frequency of words (term frequency, TF) and importance of words (inverse document frequency, IDF).
vectoriser = TfidfVectorizer()
X_text_train = vectoriser.fit_transform(train_data["Transaction Description"])
# Converting the test features to vectors using the vocabulary just learnt above.
X_text_test = vectoriser.transform(test_data["Transaction Description"])


# The numerical input data (the amount credited to the account and the type of transaction) is already suitable.
X_num_train = (train_data.drop(columns=["Transaction Description", "Category"])).values
X_num_test = (test_data.drop(columns=["Transaction Description", "Category"])).values


# Combining these numerical and text features into one feature, suitable for inputting into the LogisticRegression model.
# hstack ensures we are still using sparse matrices when combining this sparse matrix and dense vector.
X_train = hstack([X_text_train, X_num_train])
X_test = hstack([X_text_test, X_num_test])


# Creating the target variables, y, for the train and test dataframes.
y_train = train_data["Category"]
y_test = test_data["Category"]


# Choosing logistic regression as our machine learning, classifier model.
# 1000 iterations to allow the model to converge.
clf = LogisticRegression(max_iter=10000)


# Training the model.
clf.fit(X_train, y_train)


# Testing the model by predicting categories for the test transaction descriptions.
y_predicted = clf.predict(X_test)
# Printing the accuracy of the model and the classification report.
print("The accuracy of the model is: {:.2f}%.".format(100*accuracy_score(y_test, y_predicted)))
print(classification_report(y_test, y_predicted))


# Saving the transaction-classifier model for use on real data.
joblib.dump((vectoriser, clf), "transaction_clf.joblib")
