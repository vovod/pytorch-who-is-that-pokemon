import os
from tqdm import tqdm
for i in tqdm(range(151)):
    path = "E:\\data\\pkm_c_aug\\" + str(i)
    os.mkdir(path)