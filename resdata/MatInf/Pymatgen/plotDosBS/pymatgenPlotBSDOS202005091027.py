# -*- coding: utf-8 -*-
"""
@Project : plotDosBS
@Author  : Xu-Shan Zhao
@Filename: pymatgenPlotBSDOS202005091027.py
@IDE     : PyCharm
@Time1   : 2020-05-09 10:27:36
@Time2   : 2020/5/9 10:27
@Month1  : 5月
@Month2  : 五月
"""

from pymatgen.electronic_structure.plotter import DosPlotter, BSPlotter, BSDOSPlotter
from pymatgen.io.vasp import Vasprun, BSVasprun

vasprun_file = './AlEuO3_Perovskite_DOS/vasprun.xml'
vasprun = Vasprun(vasprun_file)
tdos = vasprun.tdos
plotter = DosPlotter()
plotter.add_dos('Total DOS', tdos)
# plotter.show(xlim=[-20, 20], ylim=[-6, 6])
plotter.show()

completeDos = vasprun.complete_dos
element_dos = completeDos.get_element_dos()
plotter = DosPlotter()
plotter.add_dos_dict(element_dos)
plotter.add_dos('Total DOS', tdos)
plotter.show()

spd_dos = completeDos.get_spd_dos()
plotter = DosPlotter()
plotter.add_dos_dict(spd_dos)
plotter.add_dos('Total DOS', tdos)
plotter.show()

bsvasprun_file = './AlEuO3_Perovskite_BS/vasprun.xml'
kpoint_file = './AlEuO3_Perovskite_BS/KPOINTS'
bsvasprun = BSVasprun(bsvasprun_file, parse_projected_eigen=True)
bs = bsvasprun.get_band_structure(kpoints_filename=kpoint_file, line_mode=True)
plotter = BSPlotter(bs)
plotter.get_plot(vbm_cbm_marker=True)
plotter.show()

# banddos_fig = BSDOSPlotter()
# banddos_fig = BSDOSPlotter(bs_projection=None, dos_projection=None)
banddos_fig = BSDOSPlotter(bs_projection='elements', dos_projection='elements')
banddos_fig.get_plot(bs=bs, dos=completeDos)

import matplotlib.pyplot as plt
plt.show()
