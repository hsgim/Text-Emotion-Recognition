# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:33:03 2020

@author: IBIS
"""

import urllib.parse
import urllib.request
import json
raw_text = '나 진짜 너무 슬퍼서 계속 눈물이 흘러'
text = urllib.parse.quote(raw_text)
url = 'http://ibis_api.hanyang.ac.kr:5000/sentiment_analysis?text=' + text
print('running......\n')
data = urllib.request.urlopen(url).read()
# print(data)
data = data.decode('utf-8')
j_data = json.loads(data)
print(j_data)