# Automated Bank Transaction Classification

This Python tool exploits supervised machine learning to categorise bank transactions, allowing for spending patterns to be analysed and visually displayed. The user is required to submit a bank statement (in the form of a .csv file) labelled with their desired categories. Upon training the model, the user submits an unlabelled .csv file and receives a labelled .csv file as well as a number of graphical choices to view the data.

## Method

The file 'pre_process.py' converts the labelled .csv file into a **pandas DataFrame** and drops any unnecessary / sensitive fields. A new field, 'Amount', is created to represent the magnitude of the amount credited or debited into the account. This field replaces the separate 'Debit Amount' and 'Credit Amount' fields. The information relating to the sign of the amount is contained within the 'Transaction Type' field. In order for the 'Amount' feature to not influence the model more than others, the numerical data is scaled and centred. The dataframe is divided into training and testing data. 80% of our transactions are used to train the model and 20% for testing it.

The file 'train.py' converts the labelled train and test .csv files into pandas Dataframes first. The **TF-IDF vectoriser** from **sklearn** is used to translate the category names into numerical values that can be understood by the model. This vectoriser priotises high frequency words within a transaction description and low frequency words across all transaction descriptions (to eliminate common words). A **logistic regression model** from **sklearn** is employed to predict categories for each bank transaction based on their associated description and amount. The model is saved using **joblib**.

The file 'classify.py' uses the saved transaction-classification model to categorise a new .csv file inputted by the user. The confidence level is calculated for each prediction. Should the model's confidence dip below the threshold of 0.5, the user is asked to input the correct category for that transaction.

The file 'visualise.py' provides the user with a range of options to visualise their categorised transaction data. This includes a bar chart of each category, a pie chart displaying the outgoing transactions and a pie chart displaying the incoming transactions. Furthermore, a summary of these plots may be outputted as a pdf.

The file 'explain.py' investigates the machine learning model created and aims to explain / provide a deeper understanding into it's predictions. The user is able to plot a **ROC curve** to determine the model's effectiveness at identifying a certain category in comparison to a random guess. Furthemore, the importance of each feature is explored via the coefficients of the features in the logistic regression and the **SHAP values** of each feature. These are shown within a bar chart and a **beeswarm plot**. Finally, the user also has an option for a **partial dependence plot** displaying the model output against the 'Amount' feature.


## User Guide

The program was designed for LLoyds Bank .csv files. Therefore, the code must be edited to account for varying field names. This program should be universal as all statements should contain a description, an associated value and a transaction type.

As previously stated, the user must first categorise some past transaction data to train and test the model. A simple way to achieve this is by manipulating the .csv files in Excel. The program currently assumes there will be a category labelled 'IGNORE' for distracting data you don't want visualised. The categories to be displayed can be easily edited in the 'visualise.py' code. 2090 transactions were used to train and test my model leading to a prediction accuracy of 88.52%. Of course this will vastly depend on the predictability of your data and the specificity of your chosen categories. The user should ensure there are a sufficient number of transactions belonging to each category. Ultimately, the amount of inputted transaction data is up to the user to decide, dependent on their desired accuracy.

The program is designed to have a folder, labelled 'data', containing all the .csv files to read and analyse. As this is sensitive information, it is not included within the Github repository.

Finally, all necessary libraries and their versions are contained within the 'requirements.txt' file.


## Example Output

![Example Output](examples/Spending_Summary_2025-10-01_2025-11-28.png)
