from exploration.description import *
from analysis import *
import string
import warnings
# Sentiment analysis

def warn(*args, **kwargs):
    pass

if __name__ == '__main__':
    # input your data with relative path
    warnings.warn = warn()
    customer_feedback = pd.read_excel('data/CUSTOMER_FEEDBACK.xlsx', sheet_name='Sheet')

    # Data  Visualisatoin
    # customer_feedback = EDA(customer_feedback)

    # Column Adjustment
    customer_feedback = data_process(customer_feedback,'SURVEY_TIME','CONTENT_TX')
    visulaisation(customer_feedback)

    #customer_feedback = EDA(customer_feedback)


    #review_analysis(customer_feedback, 'CONTENT_TX')






