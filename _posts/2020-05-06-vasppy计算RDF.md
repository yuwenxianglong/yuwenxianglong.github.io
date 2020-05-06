---
title: Vasppy计算RDF(Radial Distribution Functions)
author: 赵旭山
tags: 随笔
typora-root-url: ..
---

对于非晶态、无序结构，通常使用径向分布函数（RDF）代表不同粒子的分布状态。指的是给定某个粒子的坐标，其他粒子在空间的分布几率。

径向分布函数可以用来研究物质的有序性。对于晶体，由于其有序的结构，径向分布函数有长程的峰，而对于非晶体（amorphous）物质，则径向分布函数一般只有短程 的峰。

本文主要参考《[Calculating radial distribution functions](https://vasppy.readthedocs.io/en/latest/examples/rdfs.html)》，结合使用Vasppy和Pymatgen计算RDF。流程如下：

![](/assets/images/pymatgenVasppyCalcRDF202005062335.jpg)

#### 1. 通过Pymatgen生成结构

代码如下：

```python
from pymatgen import Structure, Lattice

a = 5.6402
lattice = Lattice.from_parameters(a, a, a, 90.0, 90.0, 90.0)
print(lattice)
structure = Structure.from_spacegroup(sg='Fm-3m', lattice=lattice,
                                      species=['Na', 'Cl'],
                                      coords=[[0, 0, 0], [0.5, 0, 0]])
print(structure)
```

输出如下：

```python
# print(lattice)
5.640200 0.000000 0.000000
0.000000 5.640200 0.000000
0.000000 0.000000 5.640200
```

```python
# print(structure)
Full Formula (Na4 Cl4)
Reduced Formula: NaCl
abc   :   5.640200   5.640200   5.640200
angles:  90.000000  90.000000  90.000000
Sites (8)
  #  SP      a    b    c
---  ----  ---  ---  ---
  0  Na    0    0    0
  1  Na    0    0.5  0.5
  2  Na    0.5  0    0.5
  3  Na    0.5  0.5  0
  4  Cl    0.5  0    0
  5  Cl    0    0.5  0
  6  Cl    0    0    0.5
  7  Cl    0.5  0.5  0.5
```





#### References:

* [Calculating radial distribution functions](https://vasppy.readthedocs.io/en/latest/examples/rdfs.html)
* [径向分布函数](https://baike.baidu.com/item/%E5%BE%84%E5%90%91%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0/12723225)