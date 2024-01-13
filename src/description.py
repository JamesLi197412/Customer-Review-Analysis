# Acquire basic information about each dataset
def missing_columns(df):
    missing_values = df.isnull().sum()
    missing_values = missing_values.sort_values(ascending=False)

    return missing_values


def df_info_(dataframe):
    """
        Information about the DataFrame
        # Columns Data Type
        # Data Frame shape
        # Columns Name
        # Columns Description
    """
    features_dtypes = dataframe.dtypes
    rows, columns = dataframe.shape

    missing_col = missing_columns(dataframe)
    features_names = missing_col.index.values
    missing_values = missing_col.values

    print('=' * 50)
    print('===> This data frame contains {} rows and {} columns'.format(rows, columns))
    print('=' * 50)

    print("{:13}{:13}{:30}{:15}".format('Feature Name'.upper(),
                                        'Data Format'.upper(),
                                        'Number of Missing Values'.upper(),
                                        'The first few samples'.upper()))

    for features_names, features_dtypes, missing_values in zip(features_names, features_dtypes[features_names],
                                                               missing_values):
        print('{:15} {:14} {:20}'.format(features_names, str(features_dtypes), str(missing_values) + '-' +
                                         str(round(100 * missing_values / sum(missing_col), 3)) + ' %'), end=" ")

        for i in range(5):
            print(dataframe[features_names].iloc[i], end=",")

        print("=" * 50)

def EDA():
    print('I am thinking')