import assignment2 as a2
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def raise_to_powers(x_values, features, order):
    data = np.concatenate((features, np.power(x_values, order)), axis=1)
    return data


def find_polynomial_features(x, degree):
    features = x
    for i in range(2,degree+1):
        features = raise_to_powers(x, features, i)

    return np.hstack([np.ones((x.shape[0], 1)), features])


def least_squares(x, y, lambda_value):
    xTx = x.T.dot(x)
    xTx_with_lambda = xTx + lambda_value * np.identity((xTx.shape[1]))
    xTx_inv = np.linalg.pinv(xTx_with_lambda)
    w = xTx_inv.dot(x.T.dot(y))
    print(w.shape)
    return w


def avg_loss(x, y, w, lambda_value):
  y_hat = x.dot(w) 
  loss = np.mean(np.power((y - y_hat), 2)) + lambda_value * w.transpose().dot(w)
  return loss.flatten()


(countries, features, values) = a2.load_unicef_data()
targets = values[:, 1]
x = values[:, 7:]
x = a2.normalize_data(x)

lambda_values = [0, .01, .1, 1, 10, 100, 1000, 10000, 100000]

train_losses = {}
validation_losses = {}
loss = {}
N_TRAIN = 100

for lambda_value in lambda_values:
    for i in range(0, 10):
        x_train_before = x[(i - 1) * 10:i * 10, :]
        x_validation_set = x[i * 10:(i + 1) * 10, :]
        x_train_after = x[(i + 1) * 10:N_TRAIN, :]

        t_train_before = targets[(i - 1) * 10:i * 10, :]
        t_validation_set = targets[i * 10:(i + 1) * 10, :]
        t_train_after = targets[(i + 1) * 10:N_TRAIN, :]

        # calculating x train and t train values to handle before and after validation
        # data properly
        if (x_train_before.shape[0] != 0):
            x_train = np.concatenate((x_train_before, x_train_after), axis=0)
        else:
            x_train = x_train_after

        if (x_train_after.shape[0] != 0):
            x_train = np.concatenate((x_train_before, x_train_after), axis=0)
        else:
            x_train = x_train_before

        if (t_train_before.shape[0] != 0):
            t_train = np.concatenate((t_train_before, t_train_after), axis=0)
        else:
            t_train = t_train_after

        if (t_train_after.shape[0] != 0):
            t_train = np.concatenate((t_train_before, t_train_after), axis=0)
        else:
            t_train = t_train_before

        augumented_x_train = find_polynomial_features(x_train, 2)
        w_train = least_squares(augumented_x_train, t_train, lambda_value)
        train_loss = avg_loss(augumented_x_train, t_train, w_train, lambda_value)
        train_losses[i] = train_loss

        augumented_x_validate = find_polynomial_features(x_validation_set, 2)
        validation_loss = avg_loss(augumented_x_validate, t_validation_set, w_train, lambda_value)
        validation_losses[i] = validation_loss

    total = 0
    for d in validation_losses.values():
        total += d[(0, 0)]
    loss[lambda_value] = total / 10

print("loss")
print(loss)

# Produce a plot of results.
plt.plot(loss.keys(), loss.values())
plt.semilogx(base=10)
plt.ylabel('RMS')
plt.legend(['Test error'])
plt.title('Fit with polynomials, with regularization')
plt.xlabel('Polynomial degree')
plt.show()