# -*- coding: utf-8 -*-
"""
@Project : plotDosBS
@Author  : Xu-Shan Zhao
@Filename: delOnePublicPlot202005131032.py
@IDE     : PyCharm
@Time1   : 2020-05-13 10:32:13
@Time2   : 2020/5/13 10:32
@Month1  : 5月
@Month2  : 五月
"""
import requests
from requests.auth import HTTPBasicAuth
import chart_studio

plot_id = 225

username = 'yuwenxianglong'
api_key = 'YbYbBzHWn0IVxHZyLM73'
auth = HTTPBasicAuth(username, api_key)
headers = {'Plotly-Client-Platform': 'python'}
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

fid_permanent_delete = username + ':' + str(plot_id)
print(fid_permanent_delete)
del_public_plot = requests.delete('https://api.plot.ly/v2/files/'+fid_permanent_delete+'/permanent_delete', auth=auth, headers=headers)
print(del_public_plot.status_code)