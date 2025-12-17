# Script to plot ROC curves to determine whether our model's category predictions are better than random.
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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


# The numerical input data (the amount credited to the account) is already suitable.
# Re-shaping this array to ensure a suitable matrix is produced from the combination of these numerical and text features.
X_num_train = train_data["Amount"].values.reshape(-1, 1)
X_num_test = test_data["Amount"].values.reshape(-1, 1)


# Combining these numerical and text features into one feature, suitable for inputting into the LogisticRegression model.
# hstack ensures we are still using sparse matrices when combining this sparse matrix and dense vector.
X_train = hstack([X_text_train, X_num_train])
X_test = hstack([X_text_test, X_num_test])


# Creating the target variables, y, for the train and test dataframes.
y_train = train_data["Category"]
y_test = test_data["Category"]


# Importing our transaction classification model.
vectoriser, clf = joblib.load("transaction_clf.joblib")