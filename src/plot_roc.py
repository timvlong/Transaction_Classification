# Script to plot ROC curves to determine whether our model's category predictions are better than random.
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
from scipy.sparse import hstack
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np


# Importing our transaction classification model.
vectoriser, clf = joblib.load("transaction_clf.joblib")


# Converting the .csv file (categorised) to a pandas DataFrame.
test_data = pd.read_csv("data/test_data.csv")


# Converting the test features to vectors using our imported vectoriser.
X_text_test = vectoriser.transform(test_data["Transaction Description"])


# The numerical input data (the amount credited to the account) is already suitable.
X_num_test = (test_data.drop(columns=["Transaction Description", "Category"])).values


# Combining these numerical and text features into one feature, suitable for inputting into the LogisticRegression model.
# hstack ensures we are still using sparse matrices when combining this sparse matrix and dense vector.
X_test = hstack([X_text_test, X_num_test])


# Creating the target variables, y, for the test dataframe.
y_test = test_data["Category"]


# Creating an array containing the category labels.
labels = clf.classes_
# Using binary categorisations to allow for the ROC curve plotting.
# Each row corresponds to a transaction with a 1 in the column of the category it belongs to and a 0 for other categories.
y_test_binary = label_binarize(y_test, classes=labels)


# Calculating the probability of assigning each transaction to each category.
y_probs = clf.predict_proba(X_test)


# Creating the function that will plot the ROC curve for an inputted category.
def plot_roc(category):
    """
    Plots the ROC curve for the inputted category.

    Parameters
    ----------
    category (string): Category name.
    """
    # Raising an error if the inputted category is not recognised.
    assert category in labels, "The inputted category is not one that the transactions are labelled by."
    # Finding the index that the inputted category belongs to.
    idx = np.where(labels == category)[0][0]
    # Extracting the probability of assigning the chosen category to each of the bank transactions.
    y_prob = y_probs[:, idx]
    # Calculating the false and true positive rates.
    fpr, tpr, thresholds = roc_curve(y_test_binary[:, idx], y_prob)
    # Finding the optimal probability threshold for this category, ie the one that maximises tpr - fpr.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    # Finding the optimates rates corresponding to this threshold.
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    # Calculating the area under the ROC curve, AUC, which is a useful metric to determine the model's ability to successfully classify transactions as this category.
    auc = roc_auc_score(y_test_binary[:, idx], y_prob)
    # Plotting the true against false positive rates.
    plt.plot(fpr, tpr, label="Our Classifier")
    # Plotting the y=x line corresponding to if the model just randomly guessed the classification.
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    # Plotting and labelling the optimal threshold.
    plt.plot(optimal_fpr, optimal_tpr, 'o', label="Optimal Threshold: {:.2f}".format(optimal_threshold))
    plt.title("ROC Curve for the '{}' Category. AUC = {:.2f}".format(category, auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


# Asking the user which ROC curve they would like to plot.
while True:
    print("\n Please enter the category of interest. Type exit to quit the program. \n")
    choice = input("")
    if choice.lower() == 'exit':
        break
    else:
        plot_roc(choice.upper())
