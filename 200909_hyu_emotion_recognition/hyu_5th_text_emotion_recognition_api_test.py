import urllib.parse
import urllib.request
import json
raw_text = '나 오늘 기분 진짜 좋아'
text = urllib.parse.quote(raw_text)
url = 'http://ibis_api.hanyang.ac.kr:5000/sentiment_analysis?text=' + text
print('running......\n')
data = urllib.request.urlopen(url).read()
data = data.decode('utf-8')
j_data = json.loads(data)
print(j_data)