import json
import csv
from pprint import pprint
from glob import glob
import pandas as pd
import numpy as np
import string

data = pd.read_csv("/home/rumman/data/apk.csv", header=None,
               names=["permissions", "services"])

# app_info = data.head()

# def get_res():
#     # name = url.split('/')[-1].split('.')[1:][0]
#     return 1
#
# data['malicious'] = data.services.map(get_res)
print(data.shape)
# extracted=data
data['malicious'] = 1
# extracted.drop(columns=['usefeatures', 'version_name','target_sdk', 'activities', 'providers', 'additional_permissions', 'app_file_name' ])
# df.loc[condition, 'Col3'] = 42

rows=data.shape[0]
print(rows)
data.permissions = data.permissions.astype(str)

# for perm_list in data.permissions :

    # perm_list.replace("\]", "")
    # perm_list.replace("\[", "")
    # perm_list.replace("\'", "")

# data.loc[ data.permissions != '[', 'permissions'] = data.permissions.replace("\]", "")
print(data.permissions[1])
print(data.permissions.dtype)




# url="/home/rumman/ml/Rumman_Ai/Malware_Apps/ApkMetaReport/*.json"
#
# apk_data = open('/home/rumman/data/apk.csv', 'w')
# csvwriter = csv.writer(apk_data)
# data=glob(url)
#
# count=0
#
# data=glob(url)
# for f_name in data:
#     with open(f_name) as f:
#         data = json.load(f)
#     if count==0 :
#         header = data.keys()
#         csvwriter.writerow(header)
#         count=1
#     csvwriter.writerow(data.values())
#     # apk_data.close()
#
# apk_data.close()

# pprint(data['permissions'])