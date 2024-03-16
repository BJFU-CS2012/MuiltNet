import os
import csv
import pandas as pd

l=[]
files = []
labels = []
pwd = r'E:\Z_Share\TreeData18(full)\train'
for filename in os.listdir(pwd):
    l.append(filename)


for index,value in enumerate(l):
    for filename in os.listdir(os.path.join(pwd,value)):
        files.append('images/'+value+'/'+filename)
        labels.append(index)
# labels = [1 for index in range(len(files))]

with open("train.csv","w",newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['img_path','label'])
    writer.writerows(zip(files,labels))
f.close()
