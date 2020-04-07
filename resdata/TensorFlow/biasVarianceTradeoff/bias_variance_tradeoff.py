#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

sns.set(style='whitegrid', palette='muted', font_scale=0.75)

rcParams['figure.figsize'] = 10, 6

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = np.linspace(-3, 5, 100)

np.random.shuffle(X)
X = np.sort(X[:30])
noise = [(-3 + np.random.random() * 5) for i in range(len(X))]
y = X * X + noise

X = np.expand_dims(X, axis=1)

x_plot = np.linspace(-3, 5, 100)
y_test = np.expand_dims(x_plot, axis=1)

colors = ['tomato', 'royalblue', 'goldenrod']
lw = 3

fit = ["Underfit", "Good Fit", "Overfit"]
for count, degree in enumerate([1, 2, 15]):
    plt.xlim([-3, 5])
    plt.ylim([-10, 30])
    plt.scatter(X, y, color='darkolivegreen', s=32, marker='o', label="training examples")
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_pred = model.predict(y_test)
    plt.plot(x_plot, y_pred, color=colors[count], linewidth=lw,
             label=f'prediction (degree {degree})')

    plt.legend(loc='upper right', prop={'size': 24})
    plt.title(fit[count])
    plt.show()

df = pd.read_csv('heart.csv')

df.columns

df.head()

df.shape

sns.countplot(df.target);

df.isnull().values.any()

sns.heatmap(df.corr().round(decimals=2), annot=True, linewidths=.2);

top_cor = abs(df.corr()).sort_values('target', ascending=False).index.values[1:]

top_cor

sns.pairplot(df[np.append(top_cor[:7], ["target"])]);

sns.pairplot(df[np.append(top_cor[7:], ["target"])]);

# # Underfitting
# 
# ## Non informative feature

from sklearn.model_selection import train_test_split

X = df[['trestbps']]
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

X_train.shape


def build_classifier(train_data):
    model = keras.Sequential([
        keras.layers.Dense(units=32, activation='relu', input_shape=[train_data.shape[1]]),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dense(units=1),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=['accuracy']
    )

    return model


clf = build_classifier(X_train)

BATCH_SIZE = 32

clf_history = clf.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=100,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1
)


def plot_accuracy(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],
             label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],
             label='Val Accuracy')
    plt.ylim((0, 1))
    plt.legend()
    plt.show()


plot_accuracy(clf_history)

# ### Fix with informative features

X = pd.get_dummies(df[['oldpeak', 'cp']], columns=["cp"])
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

clf = build_classifier(X_train)

clf_history = clf.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=100,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1
)

plot_accuracy(clf_history)

# ## Underpowered model

sns.scatterplot(df.age, df.thalach);
plt.title("Age vs Maximum Heart Rate");

from sklearn.preprocessing import MinMaxScaler

s = MinMaxScaler()

X = s.fit_transform(df[['age']])
y = s.fit_transform(df[['thalach']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

X_train.shape

lin_reg = keras.Sequential([
    keras.layers.Dense(1, activation='linear', input_shape=[X_train.shape[1]]),
])

lin_reg.compile(
    loss="mse",
    optimizer="adam",
    metrics=['mse']
)

reg_history = lin_reg.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=500,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1
)


def plot_loss(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val Loss')
    plt.legend()
    plt.show()


plot_loss(reg_history);

reg_history.history['val_mse'][-1]

plt.scatter(X_train, y_train, color='black')
plt.plot(X_train, lin_reg.predict(X_train), color='blue', linewidth=3);

# ### Fix
# 
# Use more powerful model:

lin_reg = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='linear'),
])

lin_reg.compile(
    loss="mse",
    optimizer="adam",
    metrics=['mse']
)

reg_history = lin_reg.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=200,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1
)

plot_loss(reg_history);

reg_history.history['val_mse'][-1]

plt.scatter(X_train, y_train, color='black')
plt.plot(X_train, lin_reg.predict(X_train), color='blue', linestyle='None', marker='x', markersize=12);

# # Overfitting
# 
# ## Many features with little training examples

X = df[['oldpeak', 'age', 'exang', 'ca', 'thalach']]
X = pd.get_dummies(X, columns=['exang', 'ca', 'thalach'])
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


def build_classifier():
    model = keras.Sequential([
        keras.layers.Dense(units=16, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(units=1, activation='sigmoid'),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=['accuracy']
    )

    return model


clf = build_classifier()

clf_history = clf.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=500,
    validation_split=0.95,
    batch_size=BATCH_SIZE,
    verbose=1
)

plot_accuracy(clf_history)

# ### The fix
# 
# Add more data

clf = build_classifier()

clf_history = clf.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=500,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1
)

plot_accuracy(clf_history)


# ## Too complex model

def build_classifier():
    model = keras.Sequential([
        keras.layers.Dense(units=128, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dense(units=8, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid'),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=['accuracy']
    )

    return model


clf = build_classifier()

clf_history = clf.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=200,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1
)

plot_accuracy(clf_history)

# ### The fix #1
# 
# Early Stopping

clf = build_classifier()

early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25)

clf_history = clf.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=200,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[early_stop]
)

plot_accuracy(clf_history)

# ### The Fix #2
# 
# Use regularization

model = keras.Sequential([
    keras.layers.Dense(units=128, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid'),
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

clf_history = model.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=200,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    verbose=1
)

plot_accuracy(clf_history)
