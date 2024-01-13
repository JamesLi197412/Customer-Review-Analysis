
import numpy as np
import pandas as pd
from description import df_info_
import openpyxl
import os
import time


def data_load():
    df = pd.read_excel( 'data/CUSTOMER_FEEDBACK.xlsx',sheet_name= 'Sheet')
    print(f'Loading the dataset')
    return df



if __name__ == '__main__':
    customer_feedback = data_load()
    print(df_info_(customer_feedback))
    #start_time = time.time()




