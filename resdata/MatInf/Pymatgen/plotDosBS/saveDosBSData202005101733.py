# -*- coding: utf-8 -*-
"""
@Project : plotDosBS
@Author  : Xu-Shan Zhao
@Filename: saveDosBSData202005101733.py
@IDE     : PyCharm
@Time1   : 2020-05-10 17:33:41
@Time2   : 2020/5/10 5:33 下午
@Month1  : 5月
@Month2  : 五月
"""

import csv
import pandas as pd
from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.core import Spin, OrbitalType, Orbital

dosvasprun_path = 'AlEuO3_Perovskite_DOS/vasprun.xml'
bsvasprun_path = 'AlEuO3_Perovskite_BS/vasprun.xml'
bskpoint = 'AlEuO3_Perovskite_BS/KPOINTS'

dosvasprun = Vasprun(dosvasprun_path)
tdos_egy = dosvasprun.tdos.energies
tdos_up = dosvasprun.tdos.densities[Spin.up]
tdos_down = dosvasprun.tdos.densities[Spin.down]
spd_dos = dosvasprun.complete_dos.get_spd_dos()
elem_dos = dosvasprun.complete_dos.get_element_dos()

df = pd.DataFrame(tdos_egy, columns=['Energies / eV 0'])
df.insert(column='Total DOS (up) 1', value=tdos_up, loc=1)
df.insert(column='Total DOS (down, negative) 2', value=-1 * tdos_down, loc=2)

orbits = []
spins = []
for i in spd_dos:
    orbits.append(i)

for i in spd_dos[orbits[0]].densities:
    spins.append(i)

print(orbits)
print(spins)

col_num = 3

for i in orbits:
    for j in spins:
        orb_dos = spd_dos[i].densities[j]
        print(orb_dos)
        print(j)
        if j.name == 'down':
            orb_dos = -1 * orb_dos
        col_name = str(i) + '(' + j.name + ') ' + str(col_num)
        df.insert(loc=col_num, column=col_name, value=orb_dos)
        col_num = col_num + 1

elem_names = []
for i in elem_dos:
    elem_names.append(i)

for i in elem_names:
    for j in spins:
        ielem_dos = elem_dos[i].densities[j]
        if j.name == 'down':
            ielem_dos = -1 * ielem_dos
        col_name = str(i.name) + '(' + j.name + ') ' + str(col_num)
        df.insert(loc=col_num, column=col_name, value=ielem_dos)
        col_num = col_num + 1

pdos = dosvasprun.complete_dos.pdos
elems = []
for i in pdos.keys():
    elems.append(i)

orbitsxyz = []
for i in pdos[elems[0]]:
    print(i)
    orbitsxyz.append(i)

for i in orbitsxyz:
    for j in elems:
        for k in spins:
            elem_spd_dos = pdos[j][i][k]
            print(elem_spd_dos)
            if k.name == 'down':
                elem_spd_dos = -1 * elem_spd_dos
            col_name = str(j.species) + '(' + str(i.name) + ',' + str(k.name) + ') ' + str(col_num)
            # col_name = str(col_num)
            df.insert(loc=col_num, column=col_name, value=elem_spd_dos)
            col_num = col_num + 1

df.to_csv('dos.csv', encoding='utf-8')
