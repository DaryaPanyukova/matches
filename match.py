import pandas as pd
import numpy as np
import datetime
import time
from sklearn.metrics import mean_absolute_error
import random
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv(r"C:\Hahaton\season-1920_csv.csv")
