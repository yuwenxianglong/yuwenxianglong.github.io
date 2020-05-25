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
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula",
            "poisson_ratio", "structure", "composition", "composition_oxid",
            'G_Reuss', 'G_Voigt', 'K_Reuss', 'K_Voigt', 'compliance_tensor',
            'elastic_tensor', 'elastic_tensor_original']
X = df.drop(excluded, axis=1)
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

try:
    model = tf.keras.models.load_model('elasticPres.h5')
except:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=60, input_dim=len(X_train.columns), activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ]
    )

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mse',
              metrics=['mae', 'mse'])

history = model.fit(X_train, y_train, epochs=1000,
                    batch_size=128, validation_split=0.2)
model.save('elasticPres.h5')

hist = pd.DataFrame(history.history, index=history.epoch)
hist['loss'].plot(legend=True, style='-')
hist['val_loss'].plot(legend=True, style=":")
plt.show()

plt.plot(y_train, model.predict(X_train), 'ro')
plt.plot(y_train, y_train, 'k-')
plt.axis('equal')
plt.show()
plt.plot(y_test, model.predict(X_test), 'ro')
plt.plot(y_test, y_test, 'k-')
plt.axis('equal')
plt.show()
