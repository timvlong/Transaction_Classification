# Cleaning the .csv files, removing unnecessary fields.
import pandas as pd
from sklearn.model_selection import train_test_split


# Converting the .csv file to a pandas DataFrame.
raw_data = pd.read_csv("raw_data.csv")
# Removing unnecessary, sensitive columns.
clean_data = raw_data.drop(columns=["Transaction Date", "Transaction Type", "Sort Code", "Account Number", "Debit Amount", "Credit Amount", "Balance"])


# Splitting this into a dataframe for training the model and for testing the model.
# Using 20% of the data for testing.
# Using stratify to ensure that there are a proportionate number of categories in both dataframes.
train_data, test_data = train_test_split(clean_data, test_size=0.2, stratify=clean_data["Category"])


# Saving the DataFrames as separate .csv files.
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)