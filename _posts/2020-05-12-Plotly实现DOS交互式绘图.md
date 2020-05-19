---
title: Plotly实现DOS交互式绘图
author: 赵旭山
tags: 随笔
typora-root-url: ..
---



Plotly是一个开源的数据交互式可视化框架，其在线绘图的Plotly API已更名为[Chart Studio](https://chart-studio.plotly.com)，可为免费的个人用户存储100张在线图片。

#### 1. 在线账户认证

```python
username = 'xxx'
api_key = 'xxxxxx'
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
```

#### 2. 读入文件

```python
dosrun = Vasprun('AlEuO3_Perovskite_DOS/vasprun.xml')
spd_dos = dosrun.complete_dos.get_spd_dos()
```

#### 3. 设置绘图数据

EXCEL的图表有如下操作，Plotly绘图主题思路与此类似：

（1）选择x、y列数据；

（2）设置绘图类型：“line”、“line+scatter”、“text”、“none”等。

##### 3.1 设置数据点标签

`mode='text'`，需实现定义`test=`，则每一个数据点上都会显示定义的text。

##### 3.2 设置绘图填充

`mode=‘none’`，什么也不显示。如果不定义`fill=`填充方式，就啥也看不到了。

##### 3.3 设置线型颜色

`line=dict(color="red", dash='dot')`，dash包括“dash”、“dot”、“dashdot”。color的方式比较灵活，如`royalblue`、`#444444`（十六进制）、`rgb(115, 115, 115)`等方式。

##### 3.4 定义数据对名称

`name='TDOS down'`，用于显示图例。

```python
trace_tdos_down = go.Scatter(
    y=-1 * dosrun.tdos.densities[Spin.down],
    x=dosrun.tdos.energies - dosrun.efermi,
    # text='Spin down',
    # mode='text',
    mode='none',
    name='TDOS down',
    line=dict(
        color='#444444',
        dash='dot'  # dash options include 'dash', 'dot', and 'dashdot'
    ),
    fill='tozeroy',
)

...

# 3s down contribution to the total DOS
trace_3s_down = go.Scatter(
    y=-1 * spd_dos[OrbitalType.s].densities[Spin.down],
    x=dosrun.tdos.energies - dosrun.efermi,
    mode="lines",
    name="3s down",
    line=dict(color="red", dash='dot')
)
```

#### 4. 设置绘图区格式

设置坐标轴名称、字体，刻度标签、字体，绘图网格等。

```python
dosxaxis = go.layout.XAxis(
    title="$E - E_f \quad / \quad \\text{eV}$",
    titlefont=dict(
        size=20,
        family='Times New Roman'
    ),
    showgrid=True,
    showline=True,
    ticks="inside",
    tickfont=dict(
        size=22,
        family='Times New Roman'
    ),
    mirror='ticks',
    linewidth=2,
    tickwidth=2,
    zerolinewidth=2,
)
dosyaxis = go.layout.YAxis(
    title="<b> Density of states <b>",
    titlefont=dict(
        size=22,
        family="Times New Roman"
    ),
    showgrid=True,
    showline=True,
    # range=[-10, 10],
    mirror="ticks",
    ticks="inside",
    tickfont=dict(
        size=22,
        family='Times New Roman'
    ),
    linewidth=2,
    tickwidth=2
)
```

#### 5. 在线绘图并获取链接

通过`Figure`函数初始化绘图，`update_layout`增加绘图区格式设置。

```python
dosdata = [trace_tdos_up, trace_3s_up, trace_3p_up, trace_3d_up,
           trace_tdos_down, trace_3s_down, trace_3p_down, trace_3d_down]

dosfig = go.Figure(data=dosdata, layout=doslayout)
dosfig.update_layout(legend=dict(
    x=0.287,
    y=0.76,
    traceorder="normal",
    font=dict(
        family="Times New Roman",
        size=18,
        color='black'
    ),
    bgcolor='LightSteelBlue',  # set backgroud color
    bordercolor='yellow',  # set border color 框线颜色
    borderwidth=2,
))
plot_url = pltly.plot(dosfig, filename="DOS_AlEuO3", auto_open=True)
print(tls.get_embed(plot_url))
```

在线绘图，并获取链接：

```python
# print(plot_url)
https://plotly.com/~yuwenxianglong/228/
# tls.get_embed(plot_url)
'<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~yuwenxianglong/228.embed" height="525" width="100%"></iframe>'
```

上述代码块中，第一个是访问链接，第二个为可以插入到网页中的Embed格式链接。

 <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~yuwenxianglong/228.embed" height="525" width="100%"></iframe>

通过`write_image`函数也可以保存本地图片：

```python
plotly.io.write_image(dosfig, 'DOS_AlEuO3.jpeg')
```



![](/assets/images/plotlyDOS202005121709.jpeg)



#### References：

* [Python API reference for `plotly`](https://plotly.com/python-api-reference/)

* [plotly.graph_objects.Scatter](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html)

* [Line Charts in Python](https://plotly.com/python/line-charts/)

* [CSS 颜色十六进制值](https://www.w3school.com.cn/cssref/css_colorsfull.asp)

* [W3School 颜色测试](https://www.w3school.com.cn/tiy/color.asp?hex=0000CD)

* [plotting the density of states and the band diagram using pymatgen and plotly in Python/v3](https://plotly.com/python/v3/ipython-notebooks/density-of-states/)

* [如何优雅地画态密度和能带结构?](http://www.jwzhang.xyz/2019/08/30/dos_plot/)

* [如何优雅地画态密度和能带结构?](https://zhuanlan.zhihu.com/p/80447349)

* [https://github.com/jwz-ecust/dos_and_band_plot](https://github.com/jwz-ecust/dos_and_band_plot)

* [https://github.com/gVallverdu/myScripts](https://github.com/gVallverdu/myScripts)

* [https://github.com/gVallverdu/cookbook/blob/master/plotly_bandDiagram.ipynb](https://github.com/gVallverdu/cookbook/blob/master/plotly_bandDiagram.ipynb)

  

