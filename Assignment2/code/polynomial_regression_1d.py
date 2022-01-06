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
targets = values[:, 1]
x = values[:, 7:15]

train_losses = {}
test_losses = {}

N_TRAIN = 100
for i in range(x.shape[1]):
    x_train = x[0:N_TRAIN, i]
    print(x_train.shape)
    x_test = x[N_TRAIN:, i]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    augumented_x_train = find_polynomial_features(x_train, 3)
    w_train = least_squares(augumented_x_train, t_train)
    train_loss = avg_loss(augumented_x_train, t_train, w_train)
    train_losses[i] = train_loss

    augumented_x_test = find_polynomial_features(x_test, 3)
    test_loss = avg_loss(augumented_x_test, t_test, w_train)
    test_losses[i] = test_loss

print("train_losses")
print(train_losses)
print("test_losses")
print(test_losses)

X = np.arange(x.shape[1])
plt.subplots(figsize=(20,8))
bar_size = 0.25
plt.bar(X , train_losses.values(), color = 'b', width = bar_size)
plt.bar(X + bar_size, test_losses.values(), color = 'g', width = bar_size)
plt.ylabel('RMS')
plt.xlabel('Features')
plt.title('Fit with features from 8-15 (Total population - Low birthweight), no regularization')
plt.legend(labels=['Train loss', 'Test loss'])
xlabel = features[7:15]
label_width = [(bar_size + i) for i in range(len(xlabel))]
plt.xticks(label_width, xlabel, rotation=75)



######## plots of the fits for degree 3 polynomials for features 11 (GNI), 12 (Life expectancy), 13 (literacy). ########

x = values[:, 10:13]
t_train = targets[0:N_TRAIN]
for i in range(x.shape[1]):
    x_train = x[0:N_TRAIN, i]

    augumented_x_train = find_polynomial_features(x_train, 3)
    w_train = least_squares(augumented_x_train, t_train)
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500).reshape(-1, 1)
    augumented_x_ev = find_polynomial_features(x_ev, 3)
    y_ev = augumented_x_ev.dot(w_train)

    plt.subplots(figsize=(20, 8))
    plt.plot(x_ev, y_ev, 'r.-')
    plt.plot(x_train, t_train, 'bo')
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()