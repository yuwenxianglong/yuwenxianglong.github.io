# -*- coding: utf-8 -*-
"""
@Project : elasticPres
@Author  : Xu-Shan Zhao
@Filename: elasticPresTF202005250903.py
@IDE     : PyCharm
@Time1   : 2020-05-25 09:03:42
@Time2   : 2020/5/25 9:03
@Month1  : 5月
@Month2  : 五月
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('elastic_tensor.csv', index_col=0)
# df = df.sort_values(by='K_VRH', ascending=True)
y = df['K_VRH'].values
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy",
            "poisson_ratio", "structure", "composition", "composition_oxid",
            'G_Reuss', 'G_Voigt', 'K_Reuss', 'K_Voigt', 'compliance_tensor',
            'elastic_tensor', 'elastic_tensor_original']
X = df.drop(excluded, axis=1)
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_formula = X_train['formula']
X_train = X_train.drop('formula', axis=1)
X_test_formula = X_test['formula']
X_test = X_test.drop('formula', axis=1)

try:
    model = tf.keras.models.load_model('elasticPres.h5')
except OSError:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=60, input_dim=len(X_train.columns), activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ]
    )

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse',
              metrics=['mae', 'mse'])

history = model.fit(X_train, y_train, epochs=500,
                    batch_size=128, validation_split=0.2)
model.save('elasticPres.h5')

hist = pd.DataFrame(history.history, index=history.epoch)

import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist['loss'],
                         mode='lines+markers',
                         name='Loss'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['val_loss'],
                         mode='lines+markers',
                         name='Val_Loss'))
fig.update_layout(
    title='Loss functions',
    xaxis_title='Epochs',
    yaxis_title='Loss Function',
    font=dict(
        family='Times New Roman',
        size=18,
        color='Maroon'
    ),
    legend=dict(
        x=0.25,
        y=0.9,
        bordercolor='yellow',
        borderwidth=2
    )
)
# fig.show()
offline.plot(fig, filename='Loss.html', auto_open=True)

fig1 = go.Figure()
m = model.predict(X_train)
fig1.add_trace(
    go.Scatter(
        x=y_train,
        y=m.reshape(len(m)),
        mode='markers',
        name='Bulk Prediction of Training Data',
        hovertext=X_train_formula,
        hoverinfo='x+y+text'
    )
)
fig1.add_trace(go.Scatter(
    x=y_train, y=y_train,
    mode='lines',
    name='Y=X'
))
fig1.update_layout(
    title='Bulk Prediction of Training Data',
    xaxis_title='DFT (MP) Bulk Modulus',
    yaxis_title='Predicted Bulk Modulus',
    font=dict(
        family='Times New Roman',
        size=20,
        color='#8B0A50'
    ),
    height=750,
    width=750,
    legend=dict(
        x=0.1,
        y=0.9,
        bordercolor='yellow',
        borderwidth=2
    )
)
offline.plot(fig1, filename='BulkPredictionOfTrainingData.html',
             auto_open=True)

fig2 = go.Figure()
m = model.predict(X_test)
fig2.add_trace(
    go.Scatter(
        x=y_test,
        y=m.reshape(len(m)),
        mode='markers',
        name='Bulk Prediction of Training Data',
        hovertext=X_test_formula,
        hoverinfo='x+y+text'
    )
)
fig2.add_trace(go.Scatter(
    x=y_test, y=y_test,
    mode='lines',
    name='Y=X'
))
fig2.update_layout(
    title='Bulk Prediction of Test Data',
    xaxis_title='DFT (MP) Bulk Modulus',
    yaxis_title='Predicted Bulk Modulus',
    font=dict(
        family='Times New Roman',
        size=20,
        color='#8B658B'
    ),
    height=750,
    width=750,
    legend=dict(
        x=0.1,
        y=0.9,
        bordercolor='yellow',
        borderwidth=2,
        bgcolor='Cyan'
    )
)
offline.plot(fig2, filename='BulkPredictionOfTestData.html',
             auto_open=True)
# hist['loss'].plot(legend=True, style='-')
# hist['val_loss'].plot(legend=True, style=":")
# plt.show()
#
# plt.plot(y_train, model.predict(X_train), 'ro')
# plt.plot(y_train, y_train, 'k-')
# plt.axis('equal')
# plt.show()
# plt.plot(y_test, model.predict(X_test), 'ro')
# plt.plot(y_test, y_test, 'k-')
# plt.axis('equal')
# plt.show()
