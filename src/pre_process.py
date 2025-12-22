# Cleaning the .csv files, removing unnecessary fields and checking for missing values.
# Splitting the data into training and testing data and outputting them as .csv files.
# Standardising the numerical data.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Converting the .csv file (pre-categorised transaction data) to a pandas DataFrame.
raw_data = pd.read_csv("data/raw_data.csv")
# Removing unnecessary, sensitive columns.
clean_data = raw_data.drop(columns=["Transaction Date", "Sort Code", "Account Number", "Balance"])
# Replacing all NaNs with zeros to allow for standardisation of the numerical data.
clean_data["Credit Amount"] = clean_data["Credit Amount"].fillna(0)
clean_data["Debit Amount"] = clean_data["Debit Amount"].fillna(0)
# Combining the debit and credit fields into one field, amount.
# The sign information (ie credit or debit info) is contained within the transaction type field.
clean_data["Amount"] = clean_data["Credit Amount"] + clean_data["Debit Amount"]
# Dropping the now-redundant fields.
clean_data = clean_data.drop(columns=["Credit Amount", "Debit Amount"])


# Deciding not to impute as the model will be very sensitive to the features, so won't risk guesses.
# Dropping samples with any missing values.
clean_data.dropna()

# The transaction type is a string of very limited options, eg DEB, FPO, FPI, DD.
# Therefore, we shall convert these n options for this feature to n-1 binary features / dummy variables.
types = pd.get_dummies(clean_data["Transaction Type"], drop_first=True, dtype=int)
# Adding these dummies to the main dataframe.
clean_data = pd.concat([clean_data, types], axis=1)
# Saving the possible transaction types.
joblib.dump(types.columns.to_list(), "transaction_types.joblib")
# Removing the now-redundant 'Transaction Type' field.
clean_data = clean_data.drop(columns=["Transaction Type"])


# Splitting this into a dataframe for training the model and for testing the model.
# Using 20% of the data for testing.
# Using stratify to ensure that there are a proportionate number of categories in both dataframes.
train_data, test_data = train_test_split(clean_data, test_size=0.2, stratify=clean_data["Category"])


# Scaling and centering the bank transactions.
# This ensures the 'Amount' feature doesn't influences the model more than the other features which are normalised by the vectoriser.
scaler = StandardScaler()
# Using double brackets, [[]], to output a dataframe as opposed to a 1-dimensional pandas Series.
train_data["Amount"] = scaler.fit_transform(train_data[["Amount"]])
# Transforming the test data according to the standardising model we created based on the training data.
test_data["Amount"] = scaler.transform(test_data[["Amount"]])


# Saving the DataFrames as separate .csv files.
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)


# Saving the scaler to use on new data when classifying.
joblib.dump(scaler, "scaler.joblib")
