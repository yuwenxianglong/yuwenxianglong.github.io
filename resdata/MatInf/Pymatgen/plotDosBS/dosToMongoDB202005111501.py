# -*- coding: utf-8 -*-
"""
@Project : plotDosBS
@Author  : Xu-Shan Zhao
@Filename: dosToMongoDB202005111501.py
@IDE     : PyCharm
@Time1   : 2020-05-11 15:01:40
@Time2   : 2020/5/11 15:01
@Month1  : 5月
@Month2  : 五月
"""

import pymongo
from pymatgen.io.vasp import Vasprun
import hashlib

dosvasprun = Vasprun('./AlEuO3_Perovskite_DOS/vasprun.xml')
complete_dos = dosvasprun.complete_dos.as_dict()

hashvalue = hashlib.sha256(str(complete_dos).encode('utf-8')).hexdigest()
print(hashvalue)
complete_dos.update(hashvalue=hashvalue)

client = pymongo.MongoClient(host='localhost',
                             port=27017)
db = client['pymatgenFormatDBs']
collection = db['complete_dos']
count = collection.count_documents({"hashvalue":hashvalue})

if count == 0:
    collection.insert_one(complete_dos)
else:
    print("Same data is exist in DB.")
