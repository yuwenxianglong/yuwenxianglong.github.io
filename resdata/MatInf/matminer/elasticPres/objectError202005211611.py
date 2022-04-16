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
print(df.columns)

import pymongo

client = pymongo.MongoClient(host='localhost', port=27017)
collection = client['matminer_format']['featurizers']

import numpy
import json
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
        # elif type(df[j][i]) == pymatgen.core.composition.Composition:
        #     doc.update({j: str(df[j][i])})
        else:
            doc.update({j: df[j][i]})
        # doc.update({j: df[j][i]})

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
        print('Same data is exist in DB.')
