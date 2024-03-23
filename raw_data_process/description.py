import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def foo(values):
    try:
        return round(100 * values / sum(values), 3)
    except ZeroDivisionError:
        return 0

def EDA():
    print('Loading the data and Analysis its structure')
    # Load the file with relative path
    df = pd.read_excel( 'data/CUSTOMER_FEEDBACK.xlsx',sheet_name= 'Sheet')

    features_dtypes = df.dtypes
    rows, columns = df.shape

    # Missing values
    missing_values_col = df.isnull().sum()
    missing_values_sort = missing_values_col.sort_values(ascending=False)

    features_names = missing_values_sort.index.values
    missing_values = missing_values_sort.values

    print('=' * 100)
    print('===> This data frame contains {} rows and {} columns'.format(rows, columns))
    print('=' * 100)

    print("{:13}{:13}{:30}{:15}".format('Feature Name'.upper(),
                                        'Data Format'.upper(),
                                        'Number of Missing Values By Columns'.upper(),
                                        'The first few samples'.upper()))

    for features_names, features_dtypes, missing_values in zip(features_names, features_dtypes[features_names],
                                                               missing_values_sort):
        print('{:15} {:14} {:20}'.format(features_names, str(features_dtypes), str(missing_values)), end=" ")

        for i in range(5):
            print(df[features_names].iloc[i], end=",")

        print("=" * 50)

    return df

def visulaisation(dataframe):
    print('Data Exploration')
    dataframe_updated = data_process(dataframe, datetimecol='SURVEY_TIME')
    # Distribution of scores
    # distribution_plt(dataframe_updated, 'LEVEL_ID', 'Scores', 'scores',' ')

    # Pie chart of Trade_Zone
    #pie_chart(dataframe_updated, 'TRADE_ZONE', 'LEVEL_ID', 'TRADE_ZONE Distribution')
    # Line chart by survey time (hrs, months)

    print('Visualisation')
    return dataframe_updated

def pie_chart(dataframe, col,target,title):
    plt.figure(figsize=(10,5), dpi = 100)
    labels = dataframe[col].unique()
    colors = sns.color_palette('pastel')[0:len(labels)]
    target_df = dataframe.groupby([col])[target].agg(['count']).reset_index()

    plt.pie(target_df['count'],labels = labels,
            autopct='%1.2f%%', startangle=45, colors=colors,
            labeldistance=0.75, pctdistance=0.4)
    plt.title(title, fontsize = 20)
    plt.axis('off')
    plt.legend()
    plt.show()

def distribution_plt(dataframe,column_name,title,xlabel,ylabel):
    # Distribution of LEVEL_ID
    sns.distplot(dataframe[column_name], color = 'red')
    plt.title(title, fontsize = 30)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel)
    plt.axvline(np.median(dataframe[column_name]), 0, linestyle='--', linewidth=1.5, color='b')
    plt.show()

def data_process(dataframe,datetimecol):
    # Covert date_time col Into multiple new columns (Year, month, day, hour)
    dataframe[datetimecol] = pd.to_datetime(dataframe[datetimecol])
    dataframe['Year'] = dataframe[datetimecol].dt.year
    dataframe['month'] = dataframe[datetimecol].dt.month
    dataframe['day'] = dataframe[datetimecol].dt.day
    dataframe['hour'] = dataframe[datetimecol].dt.hour

    return dataframe


def word2vector():
    return None