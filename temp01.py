#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:03:54 2019

@author: bmw
"""

#def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from matplotlib import cm
#from IPython import display

def my_input_fn(features, targets, batch_size, shuffle, num_epochs):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def trainer(training_steps, periods):

    housing_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
    my_feature = "total_rooms"
    my_label = "median_house_value"
    #periods = 10
    steps_per_period = training_steps / periods    
    feature_columns = [tf.feature_column.numeric_column(my_feature)]
    train_stats = housing_data.describe()
    train_stats = train_stats.transpose()
    normed_data = norm(housing_data,train_stats)
    print(normed_data.describe)
    targets = normed_data[my_label]
     # Create input functions.
    training_input_fn = lambda:my_input_fn(normed_data,targets,batch_size=400, num_epochs=None,shuffle=True)
    prediction_input_fn = lambda: my_input_fn(normed_data,targets,batch_size=400, num_epochs=1,shuffle=False)
    # define optimzer
    my_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.9,beta2=0.999,epsilon=.1,use_locking=False)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
          feature_columns=feature_columns,
          optimizer=my_optimizer
      )

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = normed_data.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

# Train the model, starting from the prior state.

    for period in range (0, periods):
        print("period now is ", period)
      
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        
    #train_metrics =  linear_regressor.evaluate(input_fn=training_input_fn)  
    #print("training data ", train_metrics)
    # predict from the model
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)

    # Format predictions as a NumPy array, so we can calculate error metrics.
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Print Mean Squared Error and Root Mean Squared Error.
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    min_house_value = normed_data["median_house_value"].min()
    max_house_value = normed_data["median_house_value"].max()   
    min_max_difference = max_house_value - min_house_value
    print("normed_data min: %0.3f" % min_house_value)
    print("normed_data max: %0.3f" % max_house_value)
    print("Min. Median House Value: %0.3f" % min_house_value)
    print("Max. Median House Value: %0.3f" % max_house_value)
    print("Difference between Min. and Max.: %0.3f" % min_max_difference)
    print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

#sample = normed_data.sample(n=300)
#x_0 = sample["total_rooms"].min()
#x_1 = sample["total_rooms"].max()

#weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
#bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

#y_0 = weight * x_0 + bias 
#y_1 = weight * x_1 + bias

#plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
#plt.ylabel("median_house_value")
#plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
#plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
#plt.show()

    y_extents = np.array([0, sample[my_label].max()])
        
    weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    
    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
    print("Model training finished.")

      # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.show()
#    plt.plot(root_mean_squared_errors)

trainer(1000,10)