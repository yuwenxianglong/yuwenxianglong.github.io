# -*- coding: utf-8 -*-
"""
@Project : pymongo_basis
@Author  : Xu-Shan Zhao
@Filename: pymongoBasis202004241048.py
@IDE     : PyCharm
@Time1   : 2020-04-24 10:48:50
@Time2   : 2020/4/24 10:48
@Month1  : 4月
@Month2  : 四月
"""

import pymongo

client = pymongo.MongoClient(host='localhost',
                             port=27017)
# client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client.MPData
# db = client['MPData']

collection = db.BandStrucs
# collection = db['BandStrucs']

result = collection.find_one()
print(result)
print(type(result))
print(result.keys())
print(result['_id'], '\n', result['@module'], '\n', result['@class'])

bsgap0 = collection.find({"band_gap.energy": 0})
print(bsgap0[0])
print(type(bsgap0))
print(type(bsgap0[0]))

index = bsgap0[0]['_id']
print(index)

bsgap_energy = bsgap0[0]['band_gap']['energy']
print(bsgap_energy)
