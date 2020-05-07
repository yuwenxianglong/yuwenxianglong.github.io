# -*- coding: utf-8 -*-
"""
@Project : matminer
@Author  : Xu-Shan Zhao
@Filename: mdfsearch202004252216.py
@IDE     : PyCharm
@Time1   : 2020-04-25 22:16:41
@Time2   : 2020/4/25 10:16 下午
@Month1  : 4月
@Month2  : 四月
"""

from mdf_forge.forge import Forge

mdf = Forge()

res = mdf.search_by_elements(elements=['Fe', 'H', 'Zr'])

for i in res:
    print(i['material']['composition'])

print(len(res))
print(res[0])

