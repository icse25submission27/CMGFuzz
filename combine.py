import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

losses = []



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = "CMGFuzz/seeds_jpg/OF17"
output_dir= "CMGFuzz/seeds_jpg/OF17_result"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
train_src_dir=os.path.join(data_dir,"train")
test_src_dir=os.path.join(data_dir,"test")

train_dst_dir=os.path.join(output_dir,"train")
test_dst_dir=os.path.join(output_dir,"test")
add_dst_dir=os.path.join(output_dir,'result')
if not os.path.isdir(train_dst_dir):
    os.mkdir(train_dst_dir)
if not os.path.isdir(test_dst_dir):
    os.mkdir(test_dst_dir)
if not os.path.isdir(add_dst_dir):
    os.mkdir(add_dst_dir)
std=70

# for class_name in os.listdir(test_src_dir):
#     img_dst_dir=os.path.join(test_dst_dir,class_name)
#     if not os.path.isdir(img_dst_dir):
#         os.mkdir(img_dst_dir)
#     img_src_dir=os.path.join(test_src_dir,class_name)
#     for img_name in os.listdir(img_src_dir):
#         img_src_path=os.path.join(img_src_dir,img_name)
#         img=Image.open(img_src_path)
#         img_dst_path=os.path.join(img_dst_dir,img_name)
#         img.save(img_dst_path)

for class_name in os.listdir(train_src_dir):
    img_dst_dir=os.path.join(train_dst_dir,class_name)
    add_img_dst_dir=os.path.join(add_dst_dir,class_name)
    if not os.path.isdir(img_dst_dir):
        os.mkdir(img_dst_dir)
    if not os.path.isdir(add_img_dst_dir):
        os.mkdir(add_img_dst_dir)
    img_src_dir=os.path.join(train_src_dir,class_name)
    result_cnt=0
    for img_name in os.listdir(img_src_dir):
        img_src_path=os.path.join(img_src_dir,img_name)
        if not os.path.isdir(img_src_path):
            img=Image.open(img_src_path)
            # img_dst_path=os.path.join(img_dst_dir,img_name)
            img_save_name=os.path.join(img_dst_dir,'result'+str(result_cnt)+".jpg")
            img.save(img_save_name)
            result_cnt+=1
        else:
            img_clus_path=img_src_path
            for i in range(3):
                img_src_path=os.path.join(img_clus_path,'Cluster_'+str(i))
                img_src_path=os.path.join(img_src_path,"result")
                for img_res_path in os.listdir(img_src_path):
                    img_name=img_res_path
                    # print(img_name)
                    img_res_path=os.path.join(img_src_path,img_res_path)
                    img=Image.open(img_res_path)
                    img_dst_path=os.path.join(add_img_dst_dir,str(i)+img_name)
                    img.save(img_dst_path)
