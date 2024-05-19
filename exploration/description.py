import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def EDA(df):
    # Explorating its dtypes, shapes and missing value by each column
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

def visulaisation(df):
    # Chinese world se
    matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
    matplotlib.rcParams['axes.labelsize'] = '15'

    # TRADE_ZONE DISTRIBUTION
    file_path ='output/visualisation/TRADE ZONE DISTRIBUTION.png'
    pie_chart(df,'TRADE_ZONE', 'STORE_CODE', 'TRADE ZONE Distribution',file_path)

    # LEVEL_ID Distribution
    file_path ='output/visualisation/LEVEL_ID Barchart.png'
    pie_chart(df,'LEVEL_ID', 'STORE_CODE', 'LEVEL ID Distribution',file_path)

    file_path = 'output/visualisation/Review Length Barchart.png'
    bar_chart(df, file_path)

    return df

def bar_chart(df, file_path):
    review_length_bar = df.groupby(['content_len']).count().reset_index()
    reviews_len = review_length_bar[['content_len','STORE_CODE']]
    reviews_len.columns = ['content_len','frequency']
    fig = px.bar(reviews_len, x="content_len", y="frequency",title="customer review length")
    fig.write_image(file_path)
    # fig.show()

def pie_chart(dataframe, col,target,title,file_path):
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
    plt.savefig(file_path)
    plt.show()



def data_process(dataframe,datetimecol,wordcol):
    # Covert date_time col Into multiple new columns (Year, month, day, hour)
    dataframe[datetimecol] = pd.to_datetime(dataframe[datetimecol])
    dataframe['Year'] = dataframe[datetimecol].dt.year
    dataframe['month'] = dataframe[datetimecol].dt.month
    dataframe['day'] = dataframe[datetimecol].dt.day
    dataframe['hour'] = dataframe[datetimecol].dt.hour

    dataframe['content_len'] = dataframe[wordcol].str.len()

    return dataframe

