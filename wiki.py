import json
import requests
import urllib3
api_token = 'وقال وزير النفط والغاز السوداني عبد الرحمن عثمان في بيان إن إنتاج حقل في منطقة جنوب كردفان (جنوب) سجل ارتفاعا بواقع ثلاثة آلاف برميل يوميا بعمالة سودانية خالصة.'
url = 'http://arabicnlp.pro/alp/'
input = 'وقال وزير النفط والغاز السوداني عبد الرحمن عثمان في بيان إن إنتاج حقل في منطقة جنوب كردفان (جنوب) سجل ارتفاعا بواقع ثلاثة آلاف برميل يوميا بعمالة سودانية خالصة.'
result={}
result = urllib3.encode_multipart_formdata(input)
header ={ 'Content-type': 'application/x-www-form-urlencoded; charset=utf-8',
'Accept-Language': 'ar-AR',
'content': 'http_build_query {result}'
          }
#x = requests.get(url, headers=header)
#print(x.text)
response = requests.post('http://arabicnlp.pro/alp/', data = None, json = result)
