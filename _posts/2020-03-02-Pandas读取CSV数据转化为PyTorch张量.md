---
title: Pandas读取CSV数据转化为PyTorch张量
author: 赵旭山
tags: Pytorch
---

#### 1. 获取数据

参考：[https://www.cnblogs.com/pinard/p/6016029.html](https://www.cnblogs.com/pinard/p/6016029.html)

UCI大学公开的机器学习数据集：[http://archive.ics.uci.edu/ml/datasets.php](http://archive.ics.uci.edu/ml/datasets.php)。

本文使用的数据介绍：[http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant](http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)

下载地址：[http://archive.ics.uci.edu/ml/machine-learning-databases/00294/](http://archive.ics.uci.edu/ml/machine-learning-databases/00294/)

该数据为发电厂的运行状况数据，共计9568个样本，分为5列，分别为：

* AT：温度；
* V：压力
* AP：湿度
* RH：压强
* PE：输出电力

AT、V、AP、RH为特征值，PE为目标值。根据“[用scikit-learn和pandas学习线性回归](https://www.cnblogs.com/pinard/p/6016029.html)”一文描述，目标值和特征值为线性对应关系，即：

<img src="http://latex.codecogs.com/gif.latex?\ PE = \theta_0 + \theta_1*AT + \theta_2*V + \theta_3*AP + \theta_4*RH" />

<img src="http://latex.codecogs.com/gif.latex?\theta_0" />、<img src="http://latex.codecogs.com/gif.latex?\theta_1" />、<img src="http://latex.codecogs.com/gif.latex?\theta_2" />、<img src="http://latex.codecogs.com/gif.latex?\theta_3" />、<img src="http://latex.codecogs.com/gif.latex?\theta_4" />为需要学习的参数。

#### 2. 读取数据并转化为PyTorch张量

将Excel文件另存为`.csv`格式。读取`csv`文件并转化为PyTorch张量的代码如下：

```python
import pandas as pd
import torch

ccpp = pd.read_csv('Folds5x2_pp.csv')

fts = ccpp.iloc[:, 0:4]
target = ccpp.iloc[:, 4]

xfts = torch.tensor(fts.to_numpy())
ytarget = torch.tensor(target.to_numpy())
```

#### 3. 用到的函数

##### 3.1 pd.read_csv()

参考：[https://blog.csdn.net/sinat_35562946/article/details/81058221][https://blog.csdn.net/sinat_35562946/article/details/81058221]

官方参数：

```python
read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, iterator=False, chunksize=None, compression='infer', thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0, escapechar=None, comment=None, encoding=None, dialect=None, tupleize_cols=None, error_bad_lines=True, warn_bad_lines=True, skipfooter=0, doublequote=True, delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None)
```

主要参数含义如下：

**filepath_or_buffer**

待读取的文件路径，还包括`http`、`ftp`、`s3`等URL链接。

**sep**

指定分隔符，默认分隔符为“`,`”（逗号）。分隔符还可以是：

* `\f`：匹配一个换页；
* `\n`：匹配一个换行符；
* `\r`：匹配一个回车符；
* `\t`：匹配一个制表符；
* `\v`：匹配一个垂直制表符；
* **而`\s+`则表示匹配任意多个上面的字符**

```python
In [3]: pd.read_csv('test.txt') # 此处的\t表示的是TAB，不是键盘打一个“\”和“t”
Out[3]: 
  1\t1.3\t1.4\t2
0  0 as\t3\t4\t5

In [4]: pd.read_csv('test.txt', sep='\s+')                      
Out[4]: 
    1  1.3  1.4  2
0  as    3    4  5
```

**encoding**

指定字符编码，默认为None，通常可以指定为`utf-8`。

##### 3.2 Pandas的loc、iloc和at、iat

参考：[https://www.jianshu.com/p/199a653e9668](https://www.jianshu.com/p/199a653e9668)

###### 3.2.1 loc

通过**标签**选取数据，即通过index和columns的值进行选取。loc方法有两个参数，按顺序控制行列选择。

```python
In [5]: df = pd.DataFrame(np.random.randn(3, 4), columns=list('a
   ...: bcd'), index=list('efg'))                               

In [6]: df                                                      
Out[6]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
f  0.009353 -1.596606  0.085415  0.016214
g -0.229541  0.432984 -0.129122  0.794267

In [7]: df.loc['e'] # 索引某一行                     
Out[7]: 
a   -0.926893
b    0.427773
c    0.553269
d   -0.014907
Name: e, dtype: float64

In [10]: df.loc[['e', 'g']] # 索引多行                            
Out[10]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
g -0.229541  0.432984 -0.129122  0.794267

In [11]: df.loc[:, :'c'] # 索引多列                               
Out[11]: 
          a         b         c
e -0.926893  0.427773  0.553269
f  0.009353 -1.596606  0.085415
g -0.229541  0.432984 -0.129122

# 如果索引的标签不在index或columns范围则会报错，a标签在列中，loc的第一个参数为行索引。
In [12]: df.loc['a']                                            
----------------------------------------------------------------
KeyError                       Traceback (most recent call last)
/usr/local/Caskroom/miniconda/base/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
......
KeyError: 'a'
```

###### 3.2.2 iloc

通过**行号**选取数据，即通过数据所在的行列数（0、1、2、3、...，从0开始）选取数据。iloc方法也有两个参数，按顺序控制行列选取。

```python
In [13]: df                                                     
Out[13]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
f  0.009353 -1.596606  0.085415  0.016214
g -0.229541  0.432984 -0.129122  0.794267

# 选取一行
In [14]: df.iloc[0]                                             
Out[14]: 
a   -0.926893
b    0.427773
c    0.553269
d   -0.014907
Name: e, dtype: float64

# 选取多行
In [15]: df.iloc[0:2]                                           
Out[15]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
f  0.009353 -1.596606  0.085415  0.016214

# 选取一列或多列
In [16]: df.iloc[:, 1:3]                                        
Out[16]: 
          b         c
e  0.427773  0.553269
f -1.596606  0.085415
g  0.432984 -0.129122
```

###### 3.2.3 at/iat

通过标签或行好获取**某个数值**的具体位置。

```python
In [17]: df                                                     
Out[17]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
f  0.009353 -1.596606  0.085415  0.016214
g -0.229541  0.432984 -0.129122  0.794267

# 获取第2行，第3列位置的数据
In [18]: df.iat[1, 2]                                           
Out[18]: 0.08541461975023365

# 获取f行，b列位置的数据
In [19]: df.at['f', 'b']                                        
Out[19]: -1.596605682352801
```

###### 3.2.4 直接索引

```python
In [26]: df                                                     
Out[26]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
f  0.009353 -1.596606  0.085415  0.016214
g -0.229541  0.432984 -0.129122  0.794267

# 选择行
In [27]: df[0:2]                                                
Out[27]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
f  0.009353 -1.596606  0.085415  0.016214

# 选择列
In [28]: df['a']                                                
Out[28]: 
e   -0.926893
f    0.009353
g   -0.229541
Name: a, dtype: float64

# 选择多列
In [29]: df[['a', 'c']]                                         
Out[29]: 
          a         c
e -0.926893  0.553269
f  0.009353  0.085415
g -0.229541 -0.129122
```

继续：

```python
#行号和区间索引只能用于行（预想选取C列的数据，但这里选取除了df的所有数据，区间索引只能用于行，因defg均>c，所以所有行均被选取出来）
In [32]: df['f':]                                               
Out[32]: 
          a         b         c         d
f  0.009353 -1.596606  0.085415  0.016214
g -0.229541  0.432984 -0.129122  0.794267

In [33]: df['b':]                                               
Out[33]: 
          a         b         c         d
e -0.926893  0.427773  0.553269 -0.014907
f  0.009353 -1.596606  0.085415  0.016214
g -0.229541  0.432984 -0.129122  0.794267

# df.选取列
In [34]: df.a                                                   
Out[34]: 
e   -0.926893
f    0.009353
g   -0.229541
Name: a, dtype: float64

# 不能使用df.选取行
In [35]: df.f                                                   
----------------------------------------------------------------
AttributeError                 Traceback (most recent call last)
<ipython-input-35-f3548ccf3b14> in <module>
----> 1 df.f
......
AttributeError: 'DataFrame' object has no attribute 'f'
```

###### 3.2.5 总结

（1） `.loc`、`.iloc`只加第一个参数，如`.loc([1, 2])`、`.iloc([2:3])`，则进行的是行选择；

（2） `.loc`、`.at`，选列时只能是列名，不能是position数字；

（3） `.iloc`、`.iat`，选列时只能是position数字，不能是列名；

（4） `df[]`只能进行行选择，或列选择，不能同时进行行列选择，列选择只能是列名。行号和区间选择只能进行行选择。当index和columns标签存在重复时，通过标签选择会优先返回行数据。`df.`只能进行列选择，不能进行行选择。



#### References

* [用scikit-learn和pandas学习线性回归](https://www.cnblogs.com/pinard/p/6016029.html)
* [pandas.read_csv参数超级详解，好多栗子！](https://blog.csdn.net/sinat_35562946/article/details/81058221)
* [Pandas DataFrame的loc、iloc、ix和at/iat浅析](https://www.jianshu.com/p/199a653e9668)

