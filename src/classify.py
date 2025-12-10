# Classifying bank transactions into a number of pre-set categories.
import joblib
import pandas as pd
from scipy.sparse import hstack


# Loading our transaction-classifier model.
vectoriser, clf = joblib.load("transaction_clf.joblib")


# Converting the inputted .csv file (containing the uncategorised data) into a pandas DataFrame.
data = pd.read_csv("data/uncat_data.csv")


# Replacing all NaNs with zeros to allow for an 'amount' to be calculated and associated to each transaction.
data["Credit Amount"] = data["Credit Amount"].fillna(0)
data["Debit Amount"] = data["Debit Amount"].fillna(0)
data["Amount"] = data["Credit Amount"] - data["Debit Amount"]


# Dropping unnecessary fields in the DataFrame.
data = data.drop(columns=["Transaction Type", "Sort Code", "Account Number", "Credit Amount", "Debit Amount"])


# Extracting the features from the inputted .csv file.
X_text = vectoriser.transform(data["Transaction Description"])
X_num = data["Amount"].values.reshape(-1, 1)
# Combining these into one feature.
X = hstack([X_text, X_num])


# Predicting the category of each transaction.
data["Predicted Category"] = clf.predict(X)


# Calculating confidence levels for each prediction.
# probs contains a vector (for each transaction) which contains the model's confidence level in assigning each category to that specific transaction.
probs = clf.predict_proba(X)
# The category with highest confidence level is selected by the model, so taking the max confidence level for each transaction.
data["Confidence"] = probs.max(axis=1)


# Finding the rows of the dataframe with too small a confidence level.
threshold = 0.5
uncertain_data = data[data["Confidence"] <= threshold]
# Iterating through each unique transaction description in the dataframe of uncertain categories.
for desc in uncertain_data["Transaction Description"].unique():
    # Displaying the transaction description to help the user decide.
    print("Transaction description: {}.".format(desc))
    # Asking for the user to correct the predicted category.
    print("Please enter the correct category.")
    category = input("Category:")
    if category != "":
        data.loc[data["Transaction Description"] == desc, "Predicted Category"] = category.upper()
        # The corrected categories are guaranteed to be accurate so correcting the confidence levels to 1.
        data.loc[data["Transaction Description"] == desc, "Confidence"] = 1


# Saving the categorised transaction data as a .csv file.
data.to_csv("data/cat_data.csv", index=False)
