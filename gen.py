import os, shutil
from PIL import Image
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,
                            width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             rotation_range=40,
                             fill_mode='nearest'
                            )

train_path = 'E:\data\pokemon_classify_png_1_aug'

# for folders in os.listdir(train_path):
#     print(folders)
# #     defining the image path or the path of folder in which images are present
#     images_path = os.path.join(train_path,folders)
# #     counting the numvber of image si particular folder
#     img_count = len(os.listdir(images_path))
#     if(img_count <= 107):
#         img_arr = os.listdir(images_path)
        
#         for img in tqdm(img_arr):
#             img_ = tf.keras.preprocessing.image.load_img(os.path.join(images_path,img),target_size=(240,240))
#             img_ = tf.keras.preprocessing.image.img_to_array(img_)
#             img_ = img_.reshape(1,240,240,3)
            
#             limit = np.floor(213/img_count)

#             i = 0
#             for x in datagen.flow(img_,batch_size=1,save_to_dir = images_path,save_prefix = folders,save_format = 'png'):
#                 i += 1
#                 x = x.reshape(240,240,3)
#                 img = Image.fromarray(x,'RGB')
#                 pathii = os.path.join(images_path,'save.png')
#                 img.save(pathii)
#                 if i>=limit:
#                     break

file1 = open('obj_names.txt', 'r')
Lines = file1.readlines()
  
# Strips the newline character
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)


for folder in os.listdir(train_path):
    print(folder)
    train_new_path = os.path.join(train_path,folder)
    images = os.listdir(train_new_path)
#     split = 100
    images = images[:100]
    for img in tqdm(images):
        src = os.path.join(train_new_path,img)
        d = os.path.join('E:\data\pkm_c_aug_new',myl[int(folder)])
        des = os.path.join(d,img)
        shutil.move(src,des)

count = []

lis = os.listdir('E:\data\pkm_c_aug_new')

for folder in lis:
    p = os.path.join('E:\data\pkm_c_aug_new',folder)
    count.append(len(os.listdir(p)))
    print(str(folder) + ' count is :'+str(len(os.listdir(p))))