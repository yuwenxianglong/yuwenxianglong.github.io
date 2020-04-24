---
title: MongoDB之PyMongo基本使用
author: 赵旭山
tags: MongoDB
typora-root-url: ..
---



#### 1. 连接MongoDB数据库

```python
client = pymongo.MongoClient(host='localhost',
                             port=27017)
# client = pymongo.MongoClient('mongodb://localhost:27017/')
```

#### 2. 指定所用数据库和集合

```python
db = client.MPData
# db = client['MPData']

collection = db.BandStrucs
# collection = db['BandStrucs']
```

#### 3. 查询数据

##### 3.1 Pymongo的`find_one`

`find_one`函数返回`dict`型。

```python
result = collection.find_one()
print(result)  # {'_id': ObjectId('5ea102d4ec120c439b2bc201') ...
print(type(result))  # <class 'dict'>
print(result.keys())  # dict_keys(['_id', '@module', '@class', 'kpoints', ...
print(result['_id'], '\n', result['@module'], '\n', result['@class'])
# 5ea102d4ec120c439b2bc201 pymatgen.electronic_structure.bandstructure BandStructureSymmLine
```

##### 3.2 其他查询

`find`返回

```python
bsgap0 = collection.find({"band_gap.energy": 0})
print(bsgap0[0])  # {'_id': ObjectId('5ea104a1c78151121660237f'), ...
print(type(bsgap0))  # <class 'pymongo.cursor.Cursor'>
print(type(bsgap0[0]))  # <class 'dict'>

index = bsgap0[0]['_id']  # 5ea104a1c78151121660237f
print(index)  # 0.0
```











#### References：

* [Python操作MongoDB看这一篇就够了](https://cloud.tencent.com/developer/article/1151814)