from bing_image_downloader import downloader
from tqdm import tqdm

path = "E:\\data\\pokemon_classify\\"

file1 = open('obj_names.txt', 'r')
Lines = file1.readlines()
  
# Strips the newline character
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)

# print(myl[0])

for i in tqdm(range(122,124)):
    print("\n",i, myl[i], sep=" ", end="\n")
    new_path = path + str(i) + "\\\\"
    downloader.download(myl[i]+"pokemon", limit=120,  output_dir=new_path, adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    
#last:110