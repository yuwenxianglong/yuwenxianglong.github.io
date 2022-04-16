# -*- coding: utf-8 -*-
"""
@Project : formationEPres
@Author  : Xu-Shan Zhao
@Filename: mdfOqmdRetrieval202005280940.py
@IDE     : PyCharm
@Time1   : 2020-05-28 09:40:27
@Time2   : 2020/5/28 9:40
@Month1  : 5月
@Month2  : 五月
"""

from mdf_forge import Forge

mdf = Forge()

dataset_name = 'oqmd'
# ro = mdf.match_source_names(dataset_name)
# ro = ro.search(limit=-1)
ro = mdf.aggregate_sources(dataset_name)

import pymongo

client = pymongo.MongoClient(host='localhost', port=27017)
collection = client['MDF_datasets']['oqmd']
# collection.insert_many(ro)
for i in range(len(ro)):
    try:
        collection.insert_one(ro[i])
    except:
        print(i)
        pass