# -*- coding: utf-8 -*-
"""
@Project : plotDosBS
@Author  : Xu-Shan Zhao
@Filename: plotBS202005081622.py
@IDE     : PyCharm
@Time1   : 2020-05-08 16:22:52
@Time2   : 2020/5/8 16:22
@Month1  : 5月
@Month2  : 五月
"""

import plotly
from pymatgen.io.vasp import BSVasprun, Vasprun
from pymatgen.electronic_structure.core import Spin
import plotly.graph_objs as go
import chart_studio.plotly as pltly
import chart_studio.tools as tls

dosrun = Vasprun('AlEuO3_Perovskite_BS/vasprun.xml')

run = BSVasprun("AlEuO3_Perovskite_BS/vasprun.xml",
                parse_projected_eigen=True)  # 读取vasprun.xml
bands = run.get_band_structure(kpoints_filename="AlEuO3_Perovskite_BS/KPOINTS",
                               line_mode=True, efermi=dosrun.efermi)

emin = 1e100
emax = -1e100
for spin in bands.bands.keys():
    for band in range(bands.nb_bands):
        emin = min(emin, min(bands.bands[spin][band]))
        emax = max(emax, max(bands.bands[spin][band]))
emin = emin - bands.efermi - 1
emax = emax - bands.efermi + 1

kptslist = [k for k in range(len(bands.kpoints))]
bandTraces = list()
for band in range(bands.nb_bands):
    bandTraces.append(
        go.Scatter(
            x=kptslist,
            y=[e - bands.efermi for e in bands.bands[Spin.up][band]],
            mode="lines",
            line=dict(color="#666666"),
            showlegend=False
        )
    )

labels = [r"$L$", r"$\Gamma$", r"$X$", r"$U,K$", r"$\Gamma$"]
step = len(bands.kpoints) / (len(labels) - 1)
# vertical lines
vlines = list()
for i, label in enumerate(labels):
    vlines.append(
        go.Scatter(
            x=[i * step, i * step],
            y=[emin, emax],
            mode="lines",
            line=dict(color="#111111", width=1),
            showlegend=False
        )
    )
# Labels of highsymetry k-points are added as Annotation object
annotations = list()
for i, label in enumerate(labels):
    annotations.append(
        go.layout.Annotation(
            x=i * step, y=emin,
            xref="x1", yref="y1",
            text=label,
            xanchor="center", yanchor="top",
            showarrow=False
        )
    )

bandxaxis = go.layout.XAxis(
    title="K-points",
    titlefont=dict(
        size=24,
        family='Times New Roman'
    ),
    range=[0, len(bands.kpoints)],
    showgrid=True,
    showline=True,
    ticks="",
    showticklabels=False,
    mirror=True,
    linewidth=2
)
bandyaxis = go.layout.YAxis(
    title="$ E - E_f \quad / \quad \\text{eV} $",  # 费米能级对其
    titlefont=dict(size=20),
    range=[emin, emax],
    showgrid=True,
    showline=True,
    zeroline=True,
    mirror="ticks",
    ticks="inside",
    tickfont=dict(
        family='Times New Roman',
        size=20
    ),
    linewidth=2,
    tickwidth=2,
    zerolinewidth=2
)
bandlayout = go.Layout(
    title="Bands diagram of Al",
    titlefont=dict(
        size=22,
        family='Times New Roman',
        color='grey'
    ),
    xaxis=bandxaxis,
    yaxis=bandyaxis,
    annotations=annotations
)

bandfig = go.Figure(data=bandTraces + vlines, layout=bandlayout)
plot_url = pltly.plot(bandfig, filename="Bands_Al", auto_open=True)
print(tls.get_embed(plot_url))

# plotly.offline.iplot(bandfig, filename='Bands_Al', image='jpeg')
# plotly.offline.iplot(bandfig, filename='Bands_Al')

print(bandfig)
plotly.io.write_image(bandfig, 'Bands_Al.jpeg')
