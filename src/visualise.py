# Visualising my spending patterns from an inputted, categorised .csv file.
import matplotlib.pyplot as plt
import pandas as pd


# Converting the .csv file to a pandas DataFrame.
data = pd.read_csv("data/cat_data.csv")
# Visualising the spending as a bar chart for each category of transactions.
cat_sum = data.groupby("Predicted Category")["Amount"].sum().sort_values()
# Removing un-interesting categories specific to my data.
cat_sum = cat_sum.drop(["FAMILY", "IGNORE", "SFE"])


# Defining functions corresponding to different visualisations of the data.


def plot_bar(cat_sum):
    """
    Plots bar chart containing the net change in bank balance due to each transaction category.
    
    Parameters
    ----------
    cat_sum (pandas Series): Amount credited to bank account for each predicted category.
    """
    cat_sum.plot(kind='bar')
    plt.title("Categorised Bank Transactions")
    plt.xlabel("Category")
    plt.ylabel("Money Credited into the Account (£)")
    plt.show()


def plot_out_pie(cat_sum):
    """
    Plots pie chart displaying the distribution of the outgoing payments across each transaction category.
    
    Parameters
    ----------
    cat_sum (pandas Series): Amount credited to bank account for each predicted category.
    """
    # Taking the magnitude of the outgoing payments.
    out_cat_sum = cat_sum[cat_sum < 0] * (-1)
    # Visualising the outgoing payments as a pie chart for each transaction category.
    out_cat_sum.plot(kind='pie')
    plt.title("Categorised Outgoing Transactions")
    plt.show()


def plot_in_pie(cat_sum):
    """
    Plots pie chart displaying the distribution of the incoming payments across each transaction category.
    
    Parameters
    ----------
    cat_sum (pandas Series): Amount credited to bank account for each predicted category.
    """
    # Splitting the dataframe into outgoing and incoming transactions.
    in_cat_sum = cat_sum[cat_sum > 0]
    # Visualising the incoming payments as a pie chart for each transaction category.
    in_cat_sum.plot(kind='pie')
    plt.title("Categorised Incoming Payments")
    plt.show()


# Allowing the user to decide how they wish to visualise their data.
while True:
    print("\n Please choose from the following visualisations of your data. \n")
    print("0 - Exit Program. \n")
    print("1 - Bar Chart. \n")
    print("2 - Pie Chart of Outgoing Payments. \n")
    print("3 - Pie Chart of Incoming Payments. \n")
    choice = input()
    if choice == '0':
        break
    elif choice == '1':
        plot_bar(cat_sum)
    elif choice == '2':
        plot_out_pie(cat_sum)
    elif choice == '3':
        plot_in_pie(cat_sum)
    else:
        print("\n Please enter an integer between 0 and 3.")
