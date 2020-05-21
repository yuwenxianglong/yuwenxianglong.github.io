---
title: Pymongo之解决“cannot encode object”错误
author: 赵旭山
tags: MongoDB
typora-root-url: ..
---



Pands转换的`dict`型数据无法插入MongoDB数据库，提示如下错误：

```python
bson.errors.InvalidDocument: cannot encode object: 124, of type: <class 'numpy.int64'>
```

或类似如下一些自定义的数据类型：

```python
bson.errors.InvalidDocument: cannot encode object: Structure Summary
```




```python
bson.errors.InvalidDocument: documents must have only string keys, key was Element Nb
```

**主要原因是：**

Pandas库在读取数值时，返回的不是整型或者浮点数，而是类似numpy.int64类型的一个对象，MongoDB是无法对一个对象进行编码存储的，所以需要对读取到的结果进行强制类型转换。

**处理如下：**

```python
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
```



#### Reference：

* [解决bson.errors.InvalidDocument: Cannot encode object:错误的一种方法](https://blog.csdn.net/leaderwsh/article/details/80771178)
* [一个numpy.float32类型数据存入mongodb引发的异常](https://www.cnblogs.com/earthhouge/p/10165720.html)

