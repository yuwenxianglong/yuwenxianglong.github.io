---
title: Vasppy计算RDF(Radial Distribution Functions)
author: 赵旭山
tags: 随笔
typora-root-url: ..
---

对于非晶态、无序态物质，由于不能采用对称性和周期性表征其结构特征，故通常使用径向分布函数（RDF）代表不同粒子的分布状态。指的是给定某个粒子的坐标，其他粒子在空间的分布几率。

径向分布函数可以用来研究物质的有序性。对于晶体，由于其有序的结构，径向分布函数有长程的峰，而对于非晶体（amorphous）物质，则径向分布函数一般只有短程 的峰。

本文主要参考《[Calculating radial distribution functions](https://vasppy.readthedocs.io/en/latest/examples/rdfs.html)》，结合使用Vasppy和Pymatgen计算RDF。流程如下：

![](/assets/images/pymatgenVasppyCalcRDF202005062335.jpg)

#### 1. 获取晶体结构

##### 1.1 通过Pymatgen构建结构

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

##### 1.2 通过Pymatgen接口读入结构

如读入VASP的XDATACAR。

```python
from pymatgen.io.vasp import Xdatcar

xd = Xdatcar('NaCl_800K_MD_XDATCAR')
```

以下简要描述下如何获取的`xd`结构的特征信息。

```python
# excute xd.*? using ipython
xd.__class__
...
xd.__sizeof__
xd.__str__
xd.__subclasshook__
xd.__weakref__
xd.comment
xd.concatenate
xd.get_string
xd.natoms
xd.site_symbols
xd.structures
xd.write_file
```

例如：

```python
print(xd.comment)
# Na108 Cl108
print(xd.get_string())
"""
Na108 Cl108
1.0
16.920600 0.000000 0.000000
0.000000 16.920600 0.000000
0.000000 0.000000 16.920600
Na Cl
108 108
Direct configuration=      1
0.99783880 0.00431895 0.00287784
0.01588219 0.96778688 0.33645602
0.00698080 0.00448861 0.66452168
0.01627229 0.31467975 0.01126151
0.96725581 0.34475972 0.33837438
0.00251352 0.33732767 0.67060135
0.00957808 0.65195219 0.99595384
0.02015174 0.63937025 0.34054283
...
0.83204200 0.83640607 0.84072632
"""
print(xd.site_symbols)
# ['Na', 'Cl']
```

#### 2. 计算径向分布函数（RDF）

Vasppy提供了`vasppy.rdf.RadialDistributionFunction`函数用于计算输入结构的RDF。其获取不同种类原子的方式有以下两种：

##### 2.1 indices方式选择原子

以“通过Pymatgen自行构建的晶体结构”为例，其化学式为“Na4 Cl4”，共计8个原子。

首先，通过如下方式分别找到“Na”、“Cl”原子的标注序号。

```python
indices_Na = [i for i, site in enumerate(structure) if site.species_string is 'Na']
indices_Cl = [i for i, site in enumerate(structure) if site.species_string is 'Cl']
print(indices_Na)  # [0, 1, 2, 3]
print(indices_Cl)  # [4, 5, 6, 7]
```

通过在`RadialDistributionFunction`函数中指定`indices_i`和`indices_j`参数，计算同种原子之间，或不同种原子之间的径向分布函数RDF。

```python
rdf_nana = RadialDistributionFunction(structures=[structure],
                                      indices_i=indices_Na)
rdf_clcl = RadialDistributionFunction(structures=[structure],
                                      indices_i=indices_Cl)
rdf_nacl = RadialDistributionFunction(structures=[structure],
                                      indices_i=indices_Na,
                                      indices_j=indices_Cl)

plt.plot(rdf_nana.r, rdf_nana.rdf, label='Na-Na')
plt.plot(rdf_clcl.r, rdf_clcl.rdf, label='Cl-Cl')
plt.plot(rdf_nacl.r, rdf_nacl.rdf, label='Na-Cl')
plt.legend(loc='best')
plt.show()
```

![](/assets/images/vasppyRDF_1_202005071052.jpeg)

上图中可以看到，“Na-Na”和“Cl-Cl”的第二峰重合了。Vasppy内置了一种`smeared_rdf()`方法，在原始RDF数据的基础上叠加一个Gaussian核函数，从而对峰进行展宽，展宽度由`sigma`参数控制，默认为0.1。

```python
plt.plot(rdf_nana.r, rdf_nana.smeared_rdf(), 'k', label='Na-Na')
plt.plot(rdf_clcl.r, rdf_clcl.smeared_rdf(sigma=0.05), 'b:', label='Cl-Cl')
plt.plot(rdf_nacl.r, rdf_nacl.smeared_rdf(sigma=0.05), 'g--', label='Na-Cl')
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
```

通过展宽方式分离“Na-Na”、“Cl-Cl”的RDF曲线后，绘图如下：

![](/assets/images/vasppyRDF_2_202005071101.jpeg)

##### 2.2 from_species_strings方式选择原子

此方式比上面的方式简单一些。

```python
rdf_nana = RadialDistributionFunction.from_species_strings(structures=[structure],
                                                           species_i='Na')
rdf_clcl = RadialDistributionFunction.from_species_strings(structures=[structure],
                                                           species_i='Cl')
rdf_nacl = RadialDistributionFunction.from_species_strings(structures=[structure],
                                                           species_i='Na',
                                                           species_j='Cl')
plt.plot(rdf_nana.r, rdf_nana.smeared_rdf(), 'k', label='Na-Na')
plt.plot(rdf_clcl.r, rdf_clcl.smeared_rdf(sigma=0.07), 'b:', label='Cl-Cl')
plt.plot(rdf_nacl.r, rdf_nacl.smeared_rdf(sigma=0.07), 'g--', label='Na-Cl')
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
```

![](/assets/images/vasppyRDF_3_202005071113.jpeg)

“1.2 通过Pymatgen接口读入结构”中从XDATACAR中读入的一系列结构通过此方式计算的RDF如下：

![](/assets/images/vasppyRDF_4_202005071325.jpeg)



#### References:

* [Calculating radial distribution functions](https://vasppy.readthedocs.io/en/latest/examples/rdfs.html)
* [径向分布函数](https://baike.baidu.com/item/%E5%BE%84%E5%90%91%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0/12723225)