import pandas as pd

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
        print('{:15} {:14} {:20}'.format(features_names, str(features_dtypes), str(missing_values) + '-' +
                                         str(round(100 * missing_values / sum(missing_values), 3)) + ' %'), end=" ")

        for i in range(5):
            print(df[features_names].iloc[i], end=",")

        print("=" * 50)

    return df

def visulaisation(df):
    print('Visualisation')