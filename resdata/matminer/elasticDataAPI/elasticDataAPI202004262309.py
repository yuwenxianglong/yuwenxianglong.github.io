# -*- coding: utf-8 -*-
"""
@Project : elasticDataAPI
@Author  : Xu-Shan Zhao
@Filename: elasticDataAPI202004262309.py
@IDE     : PyCharm
@Time1   : 2020-04-26 23:09:50
@Time2   : 2020/4/26 23:09
@Month1  : 4月
@Month2  : 四月
"""

import pymongo
import pandas as pd

client = pymongo.MongoClient(host='localhost',
                             port=27017)
colleciton = client.MIEDB_3rdParty.elastic_tensor_2015
res = colleciton.find()
df = pd.DataFrame(res)

print(df['volume'])

with open('POSCAR', 'w') as f:
    f.write(df['poscar'][0])
