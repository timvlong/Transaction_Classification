# This program aims to visually explore our model (rather than the output as in visualise.py) and explain how it works.
# Feature importances will be explored via coefficients of the logistic regression and SHAP values.
# Will also plot ROC curves to determine whether our model's category predictions are better than random.
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
from scipy.sparse import hstack
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
import shap


# Loading our transaction-classifier model.
vectoriser, clf = joblib.load("models/transaction_clf.joblib")


# Converting the .csv file (categorised) to a pandas DataFrame.
test_data = pd.read_csv("data/test_data.csv")
train_data = pd.read_csv("data/train_data.csv")


# Converting the features to vectors using our imported vectoriser.
X_text_test = vectoriser.transform(test_data["Transaction Description"])
X_text_train = vectoriser.transform(train_data["Transaction Description"])


# The numerical input data (the amount credited to the account) is already suitable.
X_num_test = (test_data.drop(columns=["Transaction Description", "Category"])).values
X_num_train = (train_data.drop(columns=["Transaction Description", "Category"])).values


# Combining these numerical and text features into one feature, suitable for inputting into the LogisticRegression model.
# hstack ensures we are still using sparse matrices when combining this sparse matrix and dense vector.
X_train = hstack([X_text_train, X_num_train])
X_test = hstack([X_text_test, X_num_test])


# Creating the target variables, y, for the test dataframe.
y_train = train_data["Category"]
y_test = test_data["Category"]


# Creating an array containing the category labels.
labels = clf.classes_


# Collecting an array of feature names.
num_features = (train_data.drop(columns=["Transaction Description", "Category"])).columns
text_features = vectoriser.get_feature_names_out()
feature_names = np.concatenate([text_features, num_features])


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


# Creating the function that will visualise the feature importances via the feature coefficients in the regression.
def plot_coefs(n_features = 10):
    """
    Plots a bar chart displaying the absolute value of the coefficients of each feature (ie the feature importances) in the logistic regression.
    In other words, the importance of each feature in predicting any category.

    Parameters
    ----------
    n_features (integer): (Optional) Number of significant features we wish to plot. Default of 10.
    """
    assert type(n_features) is int and n_features > 0, "The number of features must be a positive integer."
    # Investigating the importance of each feature in categorising the bank transactions.
    # Since the features are normalised, the magnitude of the coefficients in this logistic regression should be proportional to the feature importance.
    importances = np.abs(clf.coef_[0])
    # We have many features (as text is split into tokens each represented by a binary feature) so will only show the n_features most important.
    indices = np.argsort(importances)[-n_features:]
    plt.barh(feature_names[indices], importances[indices])
    plt.title(f"The {n_features} Most Important Features in Predicting Any Class")
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Creating the function that will visualise the feature importances via their SHAP values in a bar chart.
def plot_shap_vals(category, n_features = 10):
    """
    Plots a bar chart displaying the absolute SHAP value of each feature.
    In other words, the importance of each feature in predicting the inputted category.

    Parameters
    ----------
    category (string): Category name.
    n_features (integer): (Optional) Number of significant features we wish to plot. Default of 10.
    """
    # Raising an error if the inputted category is not recognised.
    assert category in labels, "The inputted category is not one that the transactions are labelled by."
    assert type(n_features) is int and n_features > 0, "The number of features must be a positive integer."
    # Finding the index that the inputted category belongs to.
    idx = np.where(labels == category)[0][0]
    # Using the model-agnostic SHAP Kernel Explainer.
    # Using 10 clusters of the training dataset as a summary for efficiency.
    explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(X_train, 10))
    # Extracting the SHAP values from the explainer.
    shap_values = explainer.shap_values(X_test)
    # Finding the average (over the samples) SHAP value for each feature for each class/category.
    mean_shap = np.abs(shap_values).mean(axis=0)
    # Extracing the SHAP values for each feature for the inputted category.
    cat_mean_shap = mean_shap[:, idx]
    # We have many features (as text is split into tokens each represented by a binary feature) so will only show the n_features most important.
    indices = np.argsort(cat_mean_shap)[-n_features:]
    plt.barh(feature_names[indices], cat_mean_shap[indices])
    plt.title(f"The {n_features} Most Important Features in Predicting the '{category}' Category")
    plt.ylabel("Feature")
    plt.xlabel("SHAP Value (Importance)")
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Creating the function that will visualise feature importance via their SHAP values across all samples.
def plot_beeswarm(category, n_features = 10):
    """
    Plots a beeswarm graph displaying the SHAP value of each feature and each sample in the testset.
    In other words, the importance of each feature in predicting the inputted category.

    Parameters
    ----------
    category (string): Category name.
    n_features (integer): (Optional) Number of significant features we wish to plot. Default of 10.
    """
    # Raising an error if the inputted category is not recognised.
    assert category in labels, "The inputted category is not one that the transactions are labelled by."
    assert type(n_features) is int and n_features > 0, "The number of features must be a positive integer."
    # Finding the index that the inputted category belongs to.
    idx = np.where(labels == category)[0][0]
    # Using the model-agnostic SHAP Kernel Explainer.
    # Using 10 clusters of the training dataset as a summary for efficiency.
    explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(X_train, 10))
    # Extracting the SHAP values from the explainer.
    shap_values = explainer.shap_values(X_test)
    # Selecting the SHAP values for a specific class / category.
    cat_shap = shap_values[:, :, idx]
    # Using built-in shap method to produce a beeswarm plot.
    shap.summary_plot(cat_shap, X_test.toarray(), plot_type="dot", max_display=n_features, feature_names=feature_names, show=False)
    plt.title(f"The {n_features} Most Important Features in Predicting the '{category}' Category")
    plt.tight_layout()
    plt.show()


# Creating the function that will visualise the effect of the 'Amount' feature in predicting the inputted category.
def plot_dependence(category):
    """
    Plots the partial dependence curve for the inputted category.
    This shows the model output (probability of predicting the category) for varying values of the 'Amount' feature.

    Parameters
    ----------
    category (string): Category name.
    """
    # Raising an error if the inputted category is not recognised.
    assert category in labels, "The inputted category is not one that the transactions are labelled by."
    # Finding the index that the inputted category belongs to.
    idx = np.where(labels == category)[0][0]
    # Using the model-agnostic SHAP Kernel Explainer.
    # Using 10 clusters of the training dataset as a summary for efficiency.
    explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(X_train, 10))
    # Extracting the SHAP values from the explainer.
    shap_values = explainer.shap_values(X_test)
    # Using built-in shap method to produce a partial dependence plot of model prediction against the 'Amount' feature.
    # Only showing the 'Amount' feature as this is very explainable and continuous-valued.
    # Using a lambda function to ensure we are predicting probabilities for just one class/category.
    shap.partial_dependence_plot("Amount", lambda X: clf.predict_proba(X)[:, idx], X_test.toarray(), show=False, feature_names=feature_names)
    plt.title(f"Effect of 'Amount' Feature in Predicting the '{category}' Category")
    plt.tight_layout()
    plt.show()


# Allowing the user to decide how they wish to visualise their data.
while True:
    print("\n Please choose from the following visualisations of your model. \n")
    print("0 - Exit Program. \n")
    print("1 - ROC Curve. \n")
    print("2 - Bar Chart of Feature Coefficients. \n")
    print("3 - Bar Chart of SHAP Values. \n")
    print("4 - Beeswarm Plot of SHAP Values. \n")
    print("5 - Partial Dependence Plot of Model Prediction Against 'Amount' Feature. \n")
    choice = input()
    print("\n")
    if choice == '0':
        break
    elif choice == '1':
        print("Please input the category you wish to investigate. \n")
        category = input().upper()
        plot_roc(category)
    elif choice == '2':
        print("Please input the number of features you wish to investigate. \n")
        n_features = int(input())
        plot_coefs(n_features=n_features)
    elif choice == '3':
        print("Please input the category you wish to investigate. \n")
        category = input().upper()
        print("\n")
        print("Please input the number of features you wish to investigate. \n")
        n_features = int(input())
        plot_shap_vals(category, n_features=n_features)
    elif choice == '4':
        print("Please input the category you wish to investigate. \n")
        category = input().upper()
        print("\n")
        print("Please input the number of features you wish to investigate. \n")
        n_features = int(input())
        plot_beeswarm(category, n_features=n_features)
    elif choice == '5':
        print("Please input the category you wish to investigate. \n")
        category = input().upper()
        plot_dependence(category)
    else:
        print("\n Please enter an integer between 0 and 5.")