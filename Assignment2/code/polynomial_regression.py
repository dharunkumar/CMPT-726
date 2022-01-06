#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt


def raise_to_powers(x_values, features, order):
    data = np.concatenate((features, np.power(x_values, order)), axis=1)
    return data


def find_polynomial_features(x, degree):
    features = x
    for i in range(2,degree+1):
        features = raise_to_powers(x, features, i)

    return np.hstack([np.ones((x.shape[0], 1)), features])


def least_squares(x, y):
    w = np.linalg.pinv(x).dot(y)
    return w


def avg_loss(x, y, w):
  y_hat = x.dot(w)
  loss = np.sqrt(np.mean(np.power((y - y_hat), 2)))
  return loss


(countries, features, values) = a2.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a2.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_losses = {}
test_losses = {}
for i in range(1, 8+1):
    augumented_x_train = find_polynomial_features(x_train, i)
    w_train = least_squares(augumented_x_train, t_train)
    train_loss = avg_loss(augumented_x_train, t_train, w_train)
    train_losses[i] = train_loss
    
    augumented_x_test = find_polynomial_features(x_test, i)
    test_loss = avg_loss(augumented_x_test, t_test, w_train)
    test_losses[i] = test_loss
    
print("train_losses")
print(train_losses)
print("test_losses")
print(test_losses)







# Produce a plot of results.
plt.plot(test_losses.keys(), test_losses.values())
plt.plot(train_losses.keys(), train_losses.values())
plt.ylabel('RMS')
plt.legend(['Test error', 'Training error'])
plt.title('Fit with polynomials, with normalization')
plt.xlabel('Polynomial degree')
plt.show()
