# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:19:49 2021

@author: Qalbe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Poly_dataSet.csv')

X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)
rfr.fit(X, y)


# Visualising results

x_grid=np.arange(min(X),max(X), 0.1)
x_grid =x_grid.reshape(len(x_grid),1)

plt.scatter(X, y, color = 'red')
plt.plot(x_grid, rfr.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (DT Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Prdict by Polynominal Regression
rfr.predict(np.reshape(1,(1,1)))