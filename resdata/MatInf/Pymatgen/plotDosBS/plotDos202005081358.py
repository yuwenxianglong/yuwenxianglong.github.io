# -*- coding: utf-8 -*-
"""
@Project : plotDosBS
@Author  : Xu-Shan Zhao
@Filename: plotDosBS202005081358.py
@IDE     : PyCharm
@Time1   : 2020-05-08 13:58:56
@Time2   : 2020/5/8 13:58
@Month1  : 5月
@Month2  : 五月
"""

from pymatgen import Element
from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.electronic_structure.core import Spin, OrbitalType

import chart_studio
import chart_studio.plotly as pltly
import chart_studio.tools as tls
import plotly.graph_objs as go

# chart_studio.tools.set_credentials_file(username='yuwenxianglong', api_key='YbYbBzHWn0IVxHZyLM73')

dosrun = Vasprun('DOS\\vasprun.xml')
spd_dos = dosrun.complete_dos.get_spd_dos()

trace_tdos = go.Scatter(
    x=dosrun.tdos.densities[Spin.up],
    y=dosrun.tdos.energies - dosrun.efermi,
    mode='lines',
    name='Total DOS',
    line=dict(
        color='#444444'
    ),
    fill='tozeroy'
)

# 3s contribution to the total DOS
trace_3s = go.Scatter(
    x=spd_dos[OrbitalType.s].densities[Spin.up],
    y=dosrun.tdos.energies - dosrun.efermi,
    mode="lines",
    name="3s",
    line=dict(color="red")
)

# 3p contribution to the total DOS
trace_3p = go.Scatter(
    x=spd_dos[OrbitalType.p].densities[Spin.up],
    y=dosrun.tdos.energies - dosrun.efermi,
    mode="lines",
    name="3p",
    line=dict(color="green")
)

# dosdata = go.Data([trace_tdos, trace_3s, trace_3p])
dosdata = [trace_tdos, trace_3s, trace_3p]

dosxaxis = go.layout.XAxis(
    title="Density of states",
    showgrid=True,
    showline=True,
    range=[.01, 3],
    mirror="ticks",
    ticks="inside",
    linewidth=2,
    tickwidth=2
)
dosyaxis = go.layout.YAxis(
    title="$E - E_f \quad / \quad \\text{eV}$",
    showgrid=True,
    showline=True,
    ticks="inside",
    mirror='ticks',
    linewidth=2,
    tickwidth=2,
    zerolinewidth=2
)
doslayout = go.Layout(
    title="Density of states of Silicon",
    xaxis=dosxaxis,
    yaxis=dosyaxis
)

dosfig = go.Figure(data=dosdata, layout=doslayout)
plot_url = pltly.plot(dosfig, filename="DOS_Al", auto_open=True)
print(tls.get_embed(plot_url))

import plotly
plotly.io.write_image(dosfig, 'DOS_Al.jpeg')
