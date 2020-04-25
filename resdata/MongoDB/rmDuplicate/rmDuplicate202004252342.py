# -*- coding: utf-8 -*-
"""
@Project : rmDuplicate
@Author  : Xu-Shan Zhao
@Filename: rmDuplicate202004252342.py
@IDE     : PyCharm
@Time1   : 2020-04-25 23:42:06
@Time2   : 2020/4/25 11:42 下午
@Month1  : 4月
@Month2  : 四月
"""

import pymongo

client = pymongo.MongoClient(host='localhost',
                             port=27017)
db = client.MIEDB_3rdParty
collection = db.oqmd

scrollID = collection.find({}, {'_id': 1, 'mdf.scroll_id': 1})

# print(list(scrollID))

scrid = []
iid = []
for i in scrollID:
    ii = i['mdf']['scroll_id']
    iii = i['_id']
    print(ii)
    print(iii)
    scrid.append(ii)
    iid.append(iii)

import pandas as pd

df = pd.DataFrame(scrid, columns=['scroll_id'],
                  index=iid)

print(df.head(2))

a = df.duplicated()
print(a)
print('\n\n\n')

print('Duplicate _id:')
for i in range(len(a)):
    if a[i] == True:
        print(df.index[i])
