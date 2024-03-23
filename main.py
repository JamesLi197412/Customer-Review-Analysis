
import numpy as np
import pandas as pd
import os
import time

from models import model
from raw_data_process.description import *


if __name__ == '__main__':
    review_df = EDA()
    visulaisation(review_df)

    #print(EDA())
    #start_time = time.time()




