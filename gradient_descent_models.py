
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Boston.csv')
epochs = 1000
learning_rate = 0.01

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

X = train_df[['Distance to Employment Centres', 'ValueProperty/tax rate']].values
y = train_df['median home price'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

def dot_product(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def cost_function(X, y, w, b):
    cost = 0
    n = len(X)
    for i in range(n):
        y_pred = dot_product(w, X[i]) + b
        cost += (y_pred - y[i]) ** 2
    return cost / n

def batch_gd(X, y, weights, bias, learning_rate):
    for epoch in range(epochs):
        dw = [0.0] * len(weights)
        db = 0.0

        for i in range(len(X)):
            y_pred = dot_product(weights, X[i]) + bias
            error = y_pred - y[i]

            for j in range(len(weights)):
                dw[j] += error * X[i][j]
            db += error

        for j in range(len(weights)):
            weights[j] -= (learning_rate * dw[j]) / len(X)
        bias -= (learning_rate * db) / len(X)

        if epoch % 100 == 0:
            cost = cost_function(X, y, weights, bias)
            print(f"Epoch {epoch}, Cost: {cost:.4f}")

def stochastic_gd(X, y, weights, bias, learning_rate):
    for epoch in range(epochs):
        for i in range(len(X)):
            y_pred = dot_product(weights, X[i]) + bias
            error = y_pred - y[i]

            for j in range(len(weights)):
                weights[j] -= learning_rate * error * X[i][j]
            bias -= learning_rate * error

        if epoch % 100 == 0:
            cost = cost_function(X, y, weights, bias)
            print(f"Epoch {epoch}, Cost: {cost:.4f}")

def mini_batch_gd(X, y, weights, bias, learning_rate, batch_size=32):
    num_samples = len(X)

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            X_batch = X[start:end]
            y_batch = y[start:end]

            dw = [0.0] * len(weights)
            db = 0.0

            for i in range(len(X_batch)):
                y_pred = dot_product(weights, X_batch[i]) + bias
                error = y_pred - y_batch[i]

                for j in range(len(weights)):
                    dw[j] += error * X_batch[i][j]
                db += error

            for j in range(len(weights)):
                weights[j] -= (learning_rate * dw[j]) / len(X_batch)
            bias -= (learning_rate * db) / len(X_batch)

        if epoch % 100 == 0:
            cost = cost_function(X, y, weights, bias)
            print(f"Epoch {epoch}, Cost: {cost:.4f}")

weights = [0.0] * X.shape[1]
bias = 0.0

batch_gd(X, y, weights, bias, learning_rate)
# OR
# stochastic_gd(X, y, weights, bias, learning_rate)
# OR
# mini_batch_gd(X, y, weights, bias, learning_rate, batch_size=32)
