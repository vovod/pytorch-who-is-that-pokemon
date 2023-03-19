from PIL import Image 
import os 
from tqdm import tqdm
path = 'E:/data/pokemon_classify_png_1_aug/'
new_path = "E:/data/pkm_classify_png/"

file1 = open('obj_names.txt', 'r')
Lines = file1.readlines()
  
# Strips the newline character
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)

for i in range(2,151):
    r_path = path + str(i)
    new_r_path = new_path + myl[i]
    print(i)
    cnt = 1
    for file in tqdm(os.listdir(r_path)):
        if file != 'save.png':
            name = "img_" + str(cnt)
            cnt = cnt + 1
            img = Image.open("{}/{}".format(r_path, file))
            if img.mode == "CMYK":
                img = img.convert("RGB")
            # print('{}/{}.png'.format(new_r_path, name))
            if img.mode == "RGB":
                img.save('{}/{}.png'.format(new_r_path, name))
        # img.show()
