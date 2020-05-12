# -*- coding: utf-8 -*-
"""
@Project : plotDosBS
@Author  : Xu-Shan Zhao
@Filename: bsToMongoDB202005121111.py
@IDE     : PyCharm
@Time1   : 2020-05-12 11:11:38
@Time2   : 2020/5/12 11:11
@Month1  : 5月
@Month2  : 五月
"""
from pymatgen.io.vasp import BSVasprun
import pymongo
import hashlib

bsvasprun = BSVasprun('AlEuO3_Perovskite_BS/vasprun.xml', parse_projected_eigen=True)
bs = bsvasprun.get_band_structure(kpoints_filename='AlEuO3_Perovskite_BS/KPOINTS', line_mode=True)
bs = bs.as_dict()

hashvalue = hashlib.sha256(str(bs).encode('utf-8')).hexdigest()
print(hashvalue)
bs.update(hashvalue=hashvalue)

client = pymongo.MongoClient(host='localhost', port=27017)
db = client['pymatgenFormatDBs']
collection = db['band_structure']

count = collection.count_documents({"hashvalue": hashvalue})

if count == 0:
    collection.insert_one(bs)
else:
    print('Same data is exist in Database.')
