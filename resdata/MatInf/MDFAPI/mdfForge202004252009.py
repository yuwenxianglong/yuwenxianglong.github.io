# -*- coding: utf-8 -*-
"""
@Project : matminer
@Author  : Xu-Shan Zhao
@Filename: mdfForge202004252009.py
@IDE     : PyCharm
@Time1   : 2020-04-25 20:09:10
@Time2   : 2020/4/25 8:09 下午
@Month1  : 4月
@Month2  : 四月
"""

from mdf_forge.forge import Forge
import json

mdf = Forge()

# mdf.match_field("material.elements", "Al")
# mdf.match_field("material.elements", "Cu")
# mdf.match_field("material.elements", "Sn")
mdf.match_field("material.elements", "H")
# mdf.match_field("material.elements", "Zr")
# mdf.match_field("material.elements", "Fe")
mdf.exclude_field("material.elements", "C")
mdf.exclude_field("material.elements", "O")
# mdf.exclude_field("material.elements", "N")
# mdf.exclude_field("material.elements", "F")
# mdf.exclude_field("material.elements", "Cl")
mdf.match_field("mdf.source_name", "oqmd*")

res = mdf.search()
with open('Hcoms_oqmd.json', 'w') as f:
    for i in res:
        print(i['material']['composition'])
        f.write(json.dumps(i))
        f.write('\n')

print(len(res))

# res_json = json.dumps(res[0])
res_json = json.dumps(res)
# print(res_json)
