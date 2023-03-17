from PIL import Image 
import os 
from tqdm import tqdm
path = 'E:/data/pokemon_classify/'
new_path = "E:/data/pokemon_classify_png_1/"
for i in range(123,151):
    r_path = path + str(i)
    new_r_path = new_path + str(i)
    print(i)
    cnt = 1
    for file in tqdm(os.listdir(r_path)): 
        name = "img_" + str(cnt)
        cnt = cnt + 1
        img = Image.open("{}/{}".format(r_path, file))
        if img.mode == "CMYK":
            img = img.convert("RGB")
        # print('{}/{}.png'.format(new_r_path, name))
        if img.mode == "RGB":
            img.save('{}/{}.png'.format(new_r_path, name))
        # img.show()
