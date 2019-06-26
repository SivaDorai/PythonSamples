from keras.datasets import boston_housing
import pandas as pd
import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix, boxplot
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import keras.backend as K
from keras.callbacks import History

#load the data
(x_train,y_train), (x_test, y_test) = boston_housing.load_data('boston.npz',0.2, 113)

# remove unwanted columns from further processing
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(x_train, columns=column_names)
#df.pop('CRIM')
#df.drop(columns=['CRIM','ZN','INDUS','CHAS','NOX','AGE','DIS','RAD','TAX'],axis=1,inplace=True)
x_train = np.delete(x_train,[0,1,2,3,4,5,6,7,8,9,10,11],axis=1)#axis 1 deletes columns and 0 deletes rows
x_test = np.delete(x_test,[0,1,2,3,4,5,6,7,8,9,10,11],axis=1)#axis 1 deletes columns and 0 deletes rows


# program works fine for LSTAT - col 12.
#normalize the input data - no need to normalize the output col
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

def range_with_ignore(start, stop, ignore):
    return np.concatenate([
        np.arange(start, ignore),
        np.arange(ignore + 1, stop)
    ])

history = History()
model = Sequential()
model.add(Dense(8, input_dim=1, activation="relu"))
#model.add(Dense(2, input_dim=1,activation="relu"))
model.add(Dense(1,input_dim=1,activation='linear'))

#model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae','accuracy'])
model.compile(optimizer=optimizers.SGD(lr=0.01), loss='mse', metrics=['mse'])
#print(model.metrics[0])

weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
lr = K.eval(model.optimizer.lr)
print('Linear regression model is initialized with weights w: %.2f, b: %.2f , lr: %.2f' % (w_init, b_init,lr)) 


# This builds the model for the first time:
#history = model.fit(x_train, y_train, validation_split=0.25,batch_size=32, epochs=500,shuffle=True)
model.fit(x_train, y_train, validation_split=0.3,batch_size=32, epochs=30,shuffle=True,callbacks=[history])
loss = model.evaluate(x_test,y_test,batch_size=None)
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
lr_final = K.eval(model.optimizer.lr)
print('Linear regression model is trained to have weight w: %.2f, b: %.2f, lr: %.2f' % (w_final, b_final,lr_final))
print(history.history['val_loss'])

#Predict the output and plot the prediction vs actual target
predict=model.predict(x_test)
plt.plot(x_test, predict, 'b',x_test , y_test, 'k.')
plt.show()
epoch_arr = range_with_ignore(1,31,0)
plt.plot(epoch_arr,history.history['val_mean_squared_error'],'r')
plt.show()

#Per capita crime rate.
#The proportion of residential land zoned for lots over 25,000 square feet.
#The proportion of non-retail business acres per town.
#Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
#Nitric oxides concentration (parts per 10 million).
#The average number of rooms per dwelling.
#The proportion of owner-occupied units built before 1940.
#Weighted distances to five Boston employment centers.
#Index of accessibility to radial highways.
#Full-value property-tax rate per $10,000.
#Pupil-teacher ratio by town.
#1000 * (Bk – 0.63) ** 2 where Bk is the proportion of Black people by town.
#Percentage lower status of the population.