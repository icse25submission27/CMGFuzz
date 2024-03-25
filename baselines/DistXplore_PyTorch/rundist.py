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
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import sys
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import TensorDataset,ConcatDataset
import subprocess
import time
import argparser
# data_dir="/home/arily/datasets/tinyImageNet200/tinyImageNet200LT"
parser = argparse.ArgumentParser(description='')
parser.add_argument('-dataset',type=str, choices=['OF17', 'IN100', 'FOOD101'],default='OF17')
parser.add_argument('-data_dir', type=str)
parser.add_argument('-model_dir', type=str)

args = parser.parse_args()

command = [
    "python3",
    "AttackSetRe.py",
    "-i", os.path.join(args.data_dir,"class_0_seed.npy"),
    "-o", "GA_output/GA_100_logits_Flws_resnet/100_50",
    "-pop_num", "100",
    "-subtotal", "50",
    "-type", "mnist",
    "-model", args.model_dir,
    "-target", "1",
    "-max_iteration", "25"
]
if 1:
    start=time.time()
    file=open("/home/ubuntu/hbl/largeD/Flower17/DistXplore/DistXplore/dist-guided/save_dist_flw.txt","w+")
    tocnt=0
    for label in range(17):
        start1=time.time()
        src=label
        tarcnt=0
        for target in range(17):
            out_path="/home/ubuntu/hbl/largeD/Flower17/DistXplore/DistXplore/dist-guided/GA_output/GA_100_logits_Flws_resnet/100_50/"
            out_path=os.path.join(out_path,"class_"+str(src)+"_seed_output_"+str(target))
            out_path=os.path.join(out_path,"best_mmds")
            if os.path.isdir(out_path):
                tarcnt+=1
                tocnt=0
                continue
            if target==src:
                continue
            srcpos=3
            targetpos=15
            srcpath=f'/home/ubuntu/hbl/largeD/Flower17/classwise_data/class_{src}_seed.npy'
            command[srcpos]=srcpath
            command[targetpos]=str(target)
            subprocess.run(command,stdout=file)
            out_path="/home/ubuntu/hbl/largeD/Flower17/DistXplore/DistXplore/dist-guided/GA_output/GA_100_logits_Flws_resnet/100_50/"
            out_path=os.path.join(out_path,"class_"+str(src)+"_seed_output_"+str(target))
            out_path=os.path.join(out_path,"best_mmds")
            if not os.path.isdir(out_path) and tocnt<=5:
                tocnt+=1
                target-=1
                continue
            tocnt=0
            tarcnt+=1
        end1=time.time()
        print("timing: ",src," ",end1-start1)
    end=time.time()
    print("timing: ",end-start)
    file.close()
        # cnt+=1