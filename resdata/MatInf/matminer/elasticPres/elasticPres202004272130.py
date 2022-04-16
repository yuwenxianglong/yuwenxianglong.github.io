from matminer.datasets.convenience_loaders import load_elastic_tensor

df = load_elastic_tensor()
print(df.columns)
"""
Index(['material_id', 'formula', 'nsites', 'space_group', 'volume',
       'structure', 'elastic_anisotropy', 'G_Reuss', 'G_VRH', 'G_Voigt',
       'K_Reuss', 'K_VRH', 'K_Voigt', 'poisson_ratio', 'compliance_tensor',
       'elastic_tensor', 'elastic_tensor_original'],
      dtype='object')
"""
unwanted_columns = ["volume", "nsites", "compliance_tensor", "elastic_tensor",
                    "elastic_tensor_original", "K_Voigt", "G_Voigt", "K_Reuss", "G_Reuss"]
df = df.drop(unwanted_columns, axis=1)

from matminer.featurizers.conversions import StrToComposition

df = StrToComposition().featurize_dataframe(df, 'formula')

from matminer.featurizers.composition import ElementProperty

ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id='composition')

from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition import OxidationStates

df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

os_feat = OxidationStates()
df = os_feat.featurize_dataframe(df, "composition_oxid")

from matminer.featurizers.structure import DensityFeatures

df_feat = DensityFeatures()
df = df_feat.featurize_dataframe(df, col_id='structure')

y = df['K_VRH'].values
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula", "material_id",
            "poisson_ratio", "structure", "composition", "composition_oxid"]
X = df.drop(excluded, axis=1)
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(X, y)
print(lr.score(X, y))
print(np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))

from sklearn.model_selection import KFold, cross_val_score

crossvalidation = KFold(n_splits=10, shuffle=False,
                        random_state=1)
scores = cross_val_score(lr, X, y,
                         scoring='neg_mean_squared_error',
                         cv=crossvalidation,
                         n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2',
                            cv=crossvalidation,
                            n_jobs=1)
print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))

from matminer.figrecipes.plot import PlotlyFig
from sklearn.model_selection import cross_val_predict

pf = PlotlyFig(x_title='DFT (MP) bulk modulus (GPa)',
               y_title='Predicted bulk modulus (GPa)',
               title='Linear regression',
               # mode='notebook',
               # mode='online',
               mode='offline',
               filename="lr_regression.html")

pf.xy(xy_pairs=[(y, cross_val_predict(lr, X, y, cv=crossvalidation)), ([0, 400], [0, 400])],
      labels=df['formula'],
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}],
      showlegends=False
      )

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50,
                           random_state=1)
rf.fit(X, y)
print(rf.score(X, y))
print(np.sqrt(mean_squared_error(y_true=y,
                                 y_pred=rf.predict(X))))

r2_scores = cross_val_score(rf, X, y, scoring='r2',
                            cv=crossvalidation,
                            n_jobs=-1)
scores = cross_val_score(rf, X, y,
                         scoring='neg_mean_squared_error',
                         cv=crossvalidation,
                         n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))

pf = PlotlyFig(x_title='DFT (MP) bulk modulus (GPa)',
               y_title='Predicted bulk modulus (GPa)',
               title='Random Forest Regression',
               # mode='notebook',
               # mode='online',
               mode='offline',
               filename="rf_regression.html")

pf.xy(xy_pairs=[(y, cross_val_predict(rf, X, y, cv=crossvalidation)), ([0, 400], [0, 400])],
      labels=df['formula'],
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}],
      showlegends=False
      )

from sklearn.model_selection import train_test_split

X['formula'] = df['formula']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

train_formula = X_train['formula']
X_train = X_train.drop('formula', axis=1)
test_formula = X_test['formula']
X_test = X_test.drop('formula', axis=1)

rf_reg = RandomForestRegressor(n_estimators=50,
                               random_state=1)
rf_reg.fit(X_train, y_train)
print(rf_reg.score(X_train, y_train))
print(np.sqrt(mean_squared_error(y_true=y_train,
                                 y_pred=rf_reg.predict(X_train))))
print(rf_reg.score(X_test, y_test))
print(np.sqrt(mean_squared_error(y_true=y_test,
                                 y_pred=rf_reg.predict(X_test))))

from matminer.figrecipes.plot import PlotlyFig
pf_rf = PlotlyFig(x_title='Bulk modulus prediction residual (GPa)',
                  y_title='Probability',
                  title='Random forest regression residuals',
                  mode="offline",
                  filename="rf_regression_residuals.html")

hist_plot = pf_rf.histogram(data=[y_train-rf_reg.predict(X_train),
                                  y_test-rf_reg.predict(X_test)],
                            histnorm='probability', colors=['blue', 'red'],
                            return_plot=True
                           )
hist_plot["data"][0]['name'] = 'train'
hist_plot["data"][1]['name'] = 'test'
pf_rf.create_plot(hist_plot)

importances = rf.feature_importances_
# included = np.asarray(included)
included = X.columns.values
indices = np.argsort(importances)[::-1]

pf = PlotlyFig(y_title='Importance (%)',
               title='Feature by importances',
               mode='offline',
               filename='Feature_by_importances',
               fontsize=20,
               ticksize=15)

pf.bar(x=included[indices][0:10], y=importances[indices][0:10])







import pymongo

client = pymongo.MongoClient(host='localhost', port=27017)
collection = client['matminer_format']['featurizers']

import numpy
import pymatgen

import hashlib

nums = len(df)
for i in range(nums):
    doc = {}
    for j in df.columns:
        if type(df[j][i]) == numpy.int64:
            doc.update({j: int(df[j][i])})
        elif type(df[j][i]) == pymatgen.core.structure.Structure:
            doc.update({j: str(df[j][i])})
        elif type(df[j][i]) == pymatgen.core.composition.Composition:
            doc.update({j: str(df[j][i])})
        else:
            doc.update({j: df[j][i]})

    # with open('test.json', 'a+') as f:
    #     json.dump(doc, f)
    #     f.write('\n')
    #     f.close()

    hashvalue = hashlib.sha256(str(doc).encode('utf-8')).hexdigest()
    doc.update(hashvalue=hashvalue)

    count = collection.count_documents({'hashvalue': hashvalue})
    if count == 0:
        collection.insert_one(doc)
    else:
        pass
        # print('Same data is exist in DB.')
