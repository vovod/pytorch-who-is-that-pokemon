# FIND WHO'S THAT POKEMON WITH PYTORCH
#### All 151 classes pokemon Generation 1 classification with torchvision model.  

## Demo:
This is the result that my model predicted, you can run ```predict_single.py``` to try yourself.
  
<img src="https://i.postimg.cc/wj8mhmbk/Figure-1.png" alt="test" style="width:250px;height:220px;"><img src="https://i.postimg.cc/QNppMSMq/Figure-2.png" alt="test1" style="width:250px;height:220px;"><img src="https://i.postimg.cc/W4g47WCW/Figure-3.png" alt="test2" style="width:250px;height:220px;">

## Requirements:
`CUDA`, `Conda` installed.  

```
pip install bing_image_downloader  
pip install tensorflow
pip install matplotlib
conda create --name torch-cuda
conda activate torch-cuda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Dataset:
I used my custom dataset, it could take a long time so i wrote a tool `simp.py` to crawl data from website.  
Here is my dataset after augmentations: [KAGGLE: Pokemon gen1 images classification dataset](https://www.kaggle.com/datasets/hongdcs/pokemon-gen1-151-classes-classification).

