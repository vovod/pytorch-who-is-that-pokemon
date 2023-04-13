import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import torch
from torchvision import models
from transform import data_transforms

device = 'cuda'

#Load model
path="weights/Res18-Acc0.6927.pth"
TRAIN_MODE = {"pkm": 151, "pkm_t":3}
model =  models.resnet18(num_classes=151).to(device)
model.load_state_dict(torch.load(path))
model.eval()

#Read obj_names
file1 = open('obj_names.txt', 'r')
Lines = file1.readlines()
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)
myl.sort()

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('POKEMON Classification')
top.configure(background='#CDCDCD')
label = Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
    
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25) , (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        img = Image.open(file_path)
        img = data_transforms(img).to(device)
        img = img.unsqueeze(0)
        #Predict
        output = model(img)
        _, predicted = torch.max(output, 1)
        sign = "Model predict that pokemon is: " + myl[predicted]
        print(sign)
        label.configure(foreground='#011638', text=sign) 
        # show_classify_button(file_path)
    except:
        pass
upload = Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="POKEMON Classification",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()