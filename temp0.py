
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.data import Dataset
from sklearn import metrics
import math
from matplotlib import cm
from matplotlib import pyplot as plt
from IPython import display




def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  
  housing_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
  print(housing_data.describe)
  housing_data.isna().sum()
 