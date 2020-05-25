# -*- coding: utf-8 -*-
"""
@Project : elasticPres
@Author  : Xu-Shan Zhao
@Filename: elasticPres202005231636.py
@IDE     : PyCharm
@Time1   : 2020-05-23 16:36:49
@Time2   : 2020/5/23 4:36 下午
@Month1  : 5月
@Month2  : 五月
"""

import pymongo
import pandas as pd

client = pymongo.MongoClient(host='localhost',
                             port=27017)
collection = client['MIEDB_3rdParty']['elastic_tensor_2015']
df = pd.DataFrame(collection.find())

unwanted_columns = ["volume", "nsites", "compliance_tensor", "elastic_tensor",
                    "elastic_tensor_original", "K_Voigt", "G_Voigt", "K_Reuss", "G_Reuss"]
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

import pymatgen

for i in range(len(df)):
    df['structure'][i] = pymatgen.Structure.from_dict(df['structure'][i])

from matminer.featurizers.structure import DensityFeatures

df_feat = DensityFeatures()
df = df_feat.featurize_dataframe(df, col_id='structure')

y = df['K_VRH'].values
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula", "material_id",
            "poisson_ratio", "structure", "composition", "composition_oxid",
            "_id", "cif", "poscar", "kpoint_density"]
X = df.drop(excluded, axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(X, y)
print(lr.score(X, y), np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))

from sklearn.model_selection import KFold, cross_val_score

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error',
                         cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2',
                            cv=crossvalidation,
                            n_jobs=-1)

from matminer.figrecipes.plot import PlotlyFig
from sklearn.model_selection import cross_val_predict

pf = PlotlyFig(
    x_title='DFT (MP) bulk modulus (GPa)',
    y_title='Predicted bulk modulus (GPa)',
    title='Linear Regression',
    mode='offline',
    filename='lr_regression.html'
)

pf.xy(
    xy_pairs=[
        (y, cross_val_predict(lr, X, y, cv=crossvalidation)),
        ([0, 400], [0, 400])
    ],
    labels=df['formula'],
    modes=['markers', 'lines'],
    lines=[{}, {'color': 'black', 'dash': 'dash'}],
    showlegends=False
)

from sklearn.ensemble import RandomForestRegressor


def test_Nestimators(xtrain=X, ytrain=y, xtest=X, ytest=y):
    import matplotlib.pyplot as plt
    n_etm = []
    scs = []
    scs_test = []
    mses = []
    mses_test = []
    for i in range(100, 1001, 50):
        rf = RandomForestRegressor(n_estimators=i, random_state=1, n_jobs=-1)
        rf.fit(xtrain, ytrain)
        n_etm.append(i)
        scs.append(rf.score(xtrain, ytrain))
        mses.append(np.sqrt(mean_squared_error(y_true=ytrain,
                                               y_pred=rf.predict(xtrain))))
        scs_test.append(rf.score(xtest, ytest))
        mses_test.append(np.sqrt(mean_squared_error(y_true=ytest,
                                                    y_pred=rf.predict(xtest))))
    plt.plot(n_etm, scs, 'r-')
    plt.plot(n_etm, scs_test, 'ro')
    plt.show()
    plt.plot(n_etm, mses, 'k-')
    plt.plot(n_etm, mses_test, 'k+')
    plt.show()


rf = RandomForestRegressor(n_estimators=50, random_state=1)
rf.fit(X, y)
print(rf.score(X, y))
print(np.sqrt(mean_squared_error(y_true=y,
                                 y_pred=rf.predict(X))))
r2_scores = cross_val_score(
    rf, X, y, scoring='r2', cv=crossvalidation,
    n_jobs=-1
)
scores = cross_val_score(
    rf, X, y, scoring='neg_mean_squared_error',
    cv=crossvalidation, n_jobs=-1
)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
print(r2_scores)
print(rmse_scores)

pf_rf = PlotlyFig(
    x_title='DFT (MP) bulk modulus (GPa)',
    y_title='Random forest bulk modulus (GPa)',
    title='Random forest regression',
    mode='offline',
    filename='rf_regression.html'
)
pf_rf.xy(
    [(y, cross_val_predict(rf, X, y, cv=crossvalidation)),
     ([0, 400], [0, 400])],
    labels=df['formula'],
    modes=['markers', 'lines'],
    lines=[{}, {'color': 'black', 'dash': 'dash'}],
    showlegends=False
)

from sklearn.model_selection import train_test_split

X['formula'] = df['formula']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
train_formula = X_train['formula']
X_train = X_train.drop('formula', axis=1)
test_formula = X_test['formula']
X_test = X_test.drop('formula', axis=1)
rf_reg = RandomForestRegressor(
    n_estimators=50,
    random_state=1
)
rf_reg.fit(X_train, y_train)
print(rf_reg.score(X_train, y_train))
print(np.sqrt(mean_squared_error(
    y_true=y_train,
    y_pred=rf_reg.predict(X_train)
)))
print('------')
print(rf_reg.score(X_test, y_test))
print(np.sqrt(mean_squared_error(
    y_true=y_test,
    y_pred=rf_reg.predict(X_test)
)))

from matminer.figrecipes.plot import PlotlyFig

pf_rf = PlotlyFig(
    x_title="Bulk modulus prediction residual (GPa)",
    y_title='Probability',
    title='Random forest regression residuals',
    mode='offline',
    filename='rf_regression_residuals.html'
)
hist_plot = pf_rf.histogram(
    data=[y_train - rf_reg.predict(X_train),
          y_test - rf_reg.predict(X_test)],
    histnorm='probability', colors=['blue', 'red'],
    return_plot=True
)
hist_plot['data'][0]['name'] = 'train'
hist_plot['data'][1]['name'] = 'test'
pf_rf.create_plot(hist_plot)

importances = rf.feature_importances_
included = X.columns.values
indices = np.argsort(importances)[::-1]
pf = PlotlyFig(
    y_title='Importance (%)',
    title='Feature by importances',
    mode='offline',
    fontsize=20,
    ticksize=15
)
pf.bar(x=included[indices][0:10],
       y=importances[indices][0:10])