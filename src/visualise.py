# Visualising my spending patterns from an inputted, categorised .csv file.
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec


# Converting the .csv file to a pandas DataFrame.
data = pd.read_csv("data/cat_data.csv")


# Converting the transaction date to proper datetime. This will allow for correct start and end dates.
data["Transaction Date"] = pd.to_datetime(data["Transaction Date"], dayfirst=True)
# Only keeping the dates, not times.
data["Transaction Date"] = data["Transaction Date"].dt.date
# Calculating the date range associated to the bank statement.
start = data["Transaction Date"].min()
end = data["Transaction Date"].max()


# Grouping the data by their category and summing the net amount associated to each category.
cat_sum = data.groupby("Predicted Category")["Amount"].sum().sort_values()
# Removing un-interesting categories specific to my data.
cat_sum = cat_sum.drop(["FAMILY", "IGNORE", "SFE"])


# Defining functions corresponding to different visualisations of the data.


def plot_bar(cat_sum, start, end):
    """
    Plots bar chart containing the net change in bank balance due to each transaction category.
    
    Parameters
    ----------
    cat_sum (pandas Series): Amount credited to bank account for each predicted category.
    start (date): Earliest date encountered in bank statement.
    end (date): Latest date encountered in bank statement.
    """
    # Plotting a bar chart to display the transaction categories.
    cat_sum.plot(kind='bar')
    # Creating a semi-transparent grid to allow for easy reading.
    plt.grid(linestyle='--', alpha=0.5)
    plt.title("Categorised Bank Transactions from {} - {}.".format(start, end), weight='bold')
    plt.ylabel("Money Credited into the Account (£)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


def plot_out_pie(cat_sum, start, end):
    """
    Plots pie chart displaying the distribution of the outgoing payments across each transaction category.
    
    Parameters
    ----------
    cat_sum (pandas Series): Amount credited to bank account for each predicted category.
    start (date): Earliest date encountered in bank statement.
    end (date): Latest date encountered in bank statement.
    """
    # Taking the magnitude of the outgoing payments.
    out_cat_sum = cat_sum[cat_sum < 0] * (-1)
    # Spacing the pie chart slices away from eachother slightly.
    explode = [0.02] * len(out_cat_sum)
    # Visualising the outgoing payments as a pie chart for each transaction category.
    out_cat_sum.plot(kind='pie', autopct='%1.0f%%', explode=explode)
    plt.title("Categorised Outgoing Transactions from {} - {}.".format(start, end), weight='bold')
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_in_pie(cat_sum, start, end):
    """
    Plots pie chart displaying the distribution of the incoming payments across each transaction category.
    
    Parameters
    ----------
    cat_sum (pandas Series): Amount credited to bank account for each predicted category.
    start (date): Earliest date encountered in bank statement.
    end (date): Latest date encountered in bank statement.
    """
    # Splitting the dataframe into outgoing and incoming transactions.
    in_cat_sum = cat_sum[cat_sum > 0]
    # Spacing the pie chart slices away from eachother slightly.
    explode = [0.02] * len(in_cat_sum)
    # Visualising the incoming payments as a pie chart for each transaction category.
    in_cat_sum.plot(kind='pie', autopct='%1.0f%%', explode=explode)
    plt.title("Categorised Incoming Payments from {} - {}.".format(start, end), weight='bold')
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def download_summary(cat_sum, start, end):
    """
    Downloads a pdf summary of spending containing each graphical representation of the spending throughout the given time period.

    Parameters
    ----------
    cat_sum (pandas Series): Amount credited to bank account for each predicted category.
    start (date): Earliest date encountered in bank statement.
    end (date): Latest date encountered in bank statement.
    """
    # Splitting the dataframe into outgoing and incoming transactions.
    in_cat_sum = cat_sum[cat_sum > 0]
    # Taking the magnitude of the outgoing payments.
    out_cat_sum = cat_sum[cat_sum < 0] * (-1)
    # Plotting the bar chart and the pie charts on the same figure.
    # Formatting the figure to fit nicely on one sheet of a4 paper and saving as a pdf.
    a4 = (8.27, 11.69)
    fig = plt.figure(figsize=a4)
    # Using GridSpec to control subplot sizes. 
    axes = fig.add_gridspec(4, 1, height_ratios=[3, 0.5, 3, 3], hspace=0.5)
    # Plotting the bar chart.
    ax0 = fig.add_subplot(axes[0])
    cat_sum.plot(kind='bar', ax=ax0)
    ax0.set_title("Categorised Bank Transactions", weight='bold')
    ax0.set_ylabel("Money Credited into the Account (£)")
    ax0.set_xlabel("")
    # Creating a semi-transparent grid to allow for easy reading.
    ax0.grid(linestyle='--', alpha=0.5)
    # Spacing the pie chart slices away from eachother slightly.
    out_explode = [0.02] * len(out_cat_sum)
    # Visualising the outgoing payments as a pie chart for each transaction category.
    ax2 = fig.add_subplot(axes[2])
    out_cat_sum.plot(kind='pie', ax=ax2, autopct='%1.0f%%', explode=out_explode)
    ax2.set_title("Categorised Outgoing Transactions", weight='bold')
    ax2.set_ylabel("")
    # Spacing the pie chart slices away from eachother slightly.
    in_explode = [0.02] * len(in_cat_sum)
    # Visualising the incoming payments as a pie chart for each transaction category.
    ax3 = fig.add_subplot(axes[3])
    in_cat_sum.plot(kind='pie', ax=ax3, autopct='%1.0f%%', explode=in_explode)
    ax3.set_title("Categorised Incoming Payments", weight='bold')
    ax3.set_ylabel("")
    fig.suptitle("Summary of Spending from {} - {}.".format(start, end), weight='bold', fontsize=15)
    fig.tight_layout()
    fig.savefig("Spending_Summary_{}_{}.pdf".format(start, end))


# Allowing the user to decide how they wish to visualise their data.
while True:
    print("\n Please choose from the following visualisations of your data. \n")
    print("0 - Exit Program. \n")
    print("1 - Bar Chart. \n")
    print("2 - Pie Chart of Outgoing Payments. \n")
    print("3 - Pie Chart of Incoming Payments. \n")
    print("4 - Download a pdf Summary. \n")
    choice = input()
    if choice == '0':
        break
    elif choice == '1':
        plot_bar(cat_sum, start, end)
    elif choice == '2':
        plot_out_pie(cat_sum, start, end)
    elif choice == '3':
        plot_in_pie(cat_sum, start, end)
    elif choice == '4':
        download_summary(cat_sum, start, end)
    else:
        print("\n Please enter an integer between 0 and 4.")
