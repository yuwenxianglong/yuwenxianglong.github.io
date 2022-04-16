# -*- coding: utf-8 -*-
"""
@Project : elasticDataAPI
@Author  : Xu-Shan Zhao
@Filename: elasticDataAPI202004262110.py
@IDE     : PyCharm
@Time1   : 2020-04-26 21:10:40
@Time2   : 2020/4/26 21:10
@Month1  : 4月
@Month2  : 四月
"""

import pymongo
import pandas as pd

client = pymongo.MongoClient(host='localhost',
                             port=27017)
colleciton = client.MIEDB_3rdParty.elastic_tensor_2015_origin
res = colleciton.find()
df = pd.DataFrame(res[0]['data'], columns=res[0]['columns'])

with open('elastic_tensor_2015.json', 'w') as f:
    for i in range(1181):
        f.write(df.iloc[i].to_json())
