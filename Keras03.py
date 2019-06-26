#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:27:26 2019

@author: bmw
"""

#model = Sequential()
#model.add(Dense(units=1,input_dim=13, activation='linear'))
#model.compile(optimizer=SGD(lr=0.0001,clipvalue=0.5), loss='mse', metrics=['accuracy'])
#weights = model.layers[0].get_weights()
#w_init = weights[0][0][0]
#b_init = weights[1][0]
#print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) 
#
#
## This builds the model for the first time:
#history = model.fit(x_train, y_train, validation_split=0.25,batch_size=32, epochs=100,shuffle=True)
#loss = model.evaluate(x_test,y_test,batch_size=32)
#weights = model.layers[0].get_weights()
#w_final = weights[0][0][0]
#b_final = weights[1][0]
#print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))
#
#predict=model.predict(x_test)

#plt.plot(x_test, y_test)
#plt.show()

from keras.datasets import boston_housing
import pandas as pd

(x_train,y_train), (x_test, y_test) = boston_housing.load_data('boston.npz',0.2, 113)

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(x_train, columns=column_names)
df.head()

