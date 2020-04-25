---
title: Materials Data Facility数据接口
author: 赵旭山
tags: 随笔
typora-root-url: ..

---



Materials Data Facility (MDF)作为材料分层设计（Hierarchical Materials Design, CHiMaD）中心的建设内容之一，由美国商务部国家标准与技术研究所提供资助。面向全美乃至全球提供数据汇交、统一管理和共享服务。

目前数据提供开放访问，但在国内访问下载可能会有些困难。

#### 1. 程序接口

##### 1.1 安装Forge

pip安装：`pip install mdf_forge`；

conda安装：`conda install mdf_forge`。

##### 1.2 检索数据

```python
from mdf_forge.forge import Forge

mdf = Forge()  # Define connection

res = mdf.search_by_elements(elements=['Fe', 'H', 'Zr'])  # criteria query by elements

# print formula
for i in res:
    print(i['material']['composition'])

print(len(res))  # number of results, 4
print(res[0])  # view data
```

共计查询到4条含Fe、H和Zr的数据：


```shell
Fe1H5Zr2
Fe2H2O1Zr4
Cs15.2K0.08Na8.72Zr10.92Ti0.76Fe0.16Si72Sn0.16H16O188
K0.05Na3.29Ca0.21Mg0.07Zr0.01Ti3.21Mn0.33Nb0.51Fe0.15Si4H8O22
```

获取数据格式化为json后如下：

```json
{
    "crystal_structure": {
        "cross_reference": {
            "icsd": 86022
        },
        "number_of_atoms": 32,
        "space_group_number": 130,
        "volume": 263.856
    },
    "dft": {
        "converged": true,
        "cutoff_energy": 520.0,
        "exchange_correlation_functional": "PBE"
    },
    "files": [
        {
            "data_type": "ASCII text, with very long lines, with no line terminators",
            "filename": "26069.json",
            "globus": "globus://e38ee745-6d04-11e5-ba46-22000b92c6ec/MDF/mdf_connect/prod/data/oqmd_v13/26069.json",
            "length": 12639,
            "mime_type": "text/plain",
            "sha512": "b40f1cffb60e89e40f34b3ff096fa46bf482e562294cc9a2bcb0151a35f2df350a862267c78af3cf563157881ec448d8a05c236c5d22bdb83f21c93e3fc97c7e",
            "url": "https://e38ee745-6d04-11e5-ba46-22000b92c6ec.e.globus.org/MDF/mdf_connect/prod/data/oqmd_v13/26069.json"
        }
    ],
    "material": {
        "composition": "Fe1H5Zr2",
        "elements": [
            "Fe",
            "H",
            "Zr"
        ]
    },
    "mdf": {
        "ingest_date": "2018-11-09T19:44:43.687681Z",
        "resource_type": "record",
        "scroll_id": 185495,
        "source_id": "oqmd_v13.13",
        "source_name": "oqmd",
        "version": 13
    },
    "oqmd": {
        "band_gap": {
            "units": "eV",
            "value": 0.0
        },
        "magnetic_moment": {
            "units": "bohr/atom"
        },
        "total_energy": {
            "units": "eV/atom",
            "value": -5.6949740003125
        },
        "volume_pa": {
            "units": "angstrom^3/atom",
            "value": 8.2455
        }
    }
}
```

#### 2. 高级检索

根据以上json所示数据结构，可以通过字段定制检索。需注意的是，只能一个一个添加检索元素，否则会报错。

```python
mdf.match_field("material.elements", "H")  # set query using key-value way
mdf.exclude_field("material.elements", "C")  # set exclude criteria
mdf.exclude_field("material.elements", "O")
mdf.match_field("mdf.source_name", "oqmd*")  # set data source

res = mdf.search()  # excute query

# write as json format which can be imported by MongoDB
with open('Hcoms_oqmd.json', 'w') as f:
    for i in res:
        print(i['material']['composition'])
        f.write(json.dumps(i))

print(len(res))
```

本文将检索的数据导出为json文件，方便导入MongoDB作进一步分析。





#### References：

* [Materials Data Facility](https://www.materialsdatafacility.org/)





