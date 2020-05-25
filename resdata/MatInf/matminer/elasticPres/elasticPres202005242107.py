# -*- coding: utf-8 -*-
"""
@Project : elasticPres
@Author  : Xu-Shan Zhao
@Filename: elasticPres202005242107.py
@IDE     : PyCharm
@Time1   : 2020-05-24 21:07:40
@Time2   : 2020/5/24 9:07 下午
@Month1  : 5月
@Month2  : 五月
"""

import pymongo
import pymatgen
import pandas as pd
import numpy as np

client = pymongo.MongoClient(host='localhost',
                             port=27017)
collection = client['MIEDB_3rdParty']['elastic_tensor_2015']
df = pd.DataFrame(collection.find())
for i in range(len(df)):
    df['structure'][i] = pymatgen.Structure.from_dict(df['structure'][i])
    df['elastic_tensor'][i] = np.array(df['elastic_tensor'][i]['data'])

"""
['_id', 'material_id', 'formula', 'nsites', 'space_group', 'volume',
       'structure', 'elastic_anisotropy', 'G_Reuss', 'G_VRH', 'G_Voigt',
       'K_Reuss', 'K_VRH', 'K_Voigt', 'poisson_ratio', 'compliance_tensor',
       'elastic_tensor', 'elastic_tensor_original', 'cif', 'kpoint_density',
       'poscar']
"""
unwanted_columns = ['_id', 'material_id', 'nsites', 'volume',
                    'cif', 'kpoint_density', 'poscar']
df = df.drop(unwanted_columns, axis=1)

from matminer.featurizers.conversions import StrToComposition

sc_feat = StrToComposition()
df = sc_feat.featurize_dataframe(df, col_id='formula')

from matminer.featurizers.composition import ElementProperty

ep_feat = ElementProperty.from_preset(preset_name='magpie')
df = ep_feat.featurize_dataframe(df, col_id='composition')

from matminer.featurizers.conversions import CompositionToOxidComposition

co_feat = CompositionToOxidComposition()
df = co_feat.featurize_dataframe(df, col_id='composition')

from matminer.featurizers.composition import OxidationStates

os_feat = OxidationStates()
df = os_feat.featurize_dataframe(df, col_id='composition_oxid')

from matminer.featurizers.structure import DensityFeatures

df_feat = DensityFeatures()
df = df_feat.featurize_dataframe(df, col_id='structure')

"""
formula, structure, elastic_anisotropy, G_Reuss, G_VRH, G_Voigt, K_Reuss, K_VRH, K_Voigt,
poisson_ratio, compliance_tensor, elastic_tensor, elastic_tensor_original, composition
"""

y = df['K_VRH'].values
excluded = ['formula', 'structure', 'elastic_anisotropy', 'G_Reuss', 'G_VRH', 'G_Voigt',
            'K_Reuss', 'K_VRH', 'K_Voigt', 'poisson_ratio', 'compliance_tensor',
            'elastic_tensor', 'elastic_tensor_original', 'composition', 'composition_oxid']
X = df.drop(excluded, axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(np.sqrt(mean_squared_error(y_true=y_test, y_pred=lr.predict(X_test))))

crossvalidation = KFold(n_splits=10, shuffle=False, random_state=1)
scores = cross_val_score(lr, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation,
                         n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X_train, y_train, scoring='r2', cv=crossvalidation, n_jobs=-1)
print(rmse_scores)
print(r2_scores)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=88, random_state=1)
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf.predict(X_train))))
print(rf.score(X_test, y_test))
print(np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf.predict(X_test))))

import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            units=128,
            input_dim=len(X_train.columns)
        ),
        tf.keras.layers.Dense(1)
     ]
)
model.compile(optimizer='adam',
              loss='mse',
              )
history = model.fit(X.values, y, epochs=500, validation_data=(
    X_test.values,
    y_test
))
