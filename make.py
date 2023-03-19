import os
from tqdm import tqdm

file1 = open('obj_names.txt', 'r')
Lines = file1.readlines()
  
# Strips the newline character
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)


for i in tqdm(range(151)):
    path = "" + myl[i]
    os.mkdir(path)