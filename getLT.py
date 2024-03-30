import argparse
import itertools
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import accelerate
import argparse
import math
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from slugify import slugify
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
from huggingface_hub import create_repo

from diffusers import DPMSolverMultistepScheduler

import json

import os
import torch
import torchvision
from sklearn.cluster import KMeans
from torchvision.utils import save_image
import shutil
import subprocess



data_dir="OF17"
source_path=data_dir
dst_path=os.path.join(data_dir,"LT")
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
    ib_fac=20


    src_train_path=os.path.join(source_path,"train")
    src_test_path=os.path.join(source_path,"test")

    dst_train_path=os.path.join(dst_path,"train")
    dst_test_path=os.path.join(dst_path,"test")
    if not os.path.exists(dst_train_path):
        os.mkdir(dst_train_path)
    if not os.path.exists(dst_test_path):
        os.mkdir(dst_test_path)

    img_cnt=dict()
    img_cnt_list=[]
    for class_name in os.listdir(src_train_path):
        src_cls_path=os.path.join(src_train_path,class_name)
        res_cnt=0
        res_img=[]
        img_cnt[class_name]=len(os.listdir(src_cls_path))
        img_cnt_list.append(len(os.listdir(src_cls_path)))

    img_cnt_pos=list(range(len(img_cnt_list)))
    img_cnt_zip=list(zip(img_cnt_pos,img_cnt_list))
    img_cnt_zip=sorted(img_cnt_zip,key=lambda x:x[1], reverse=True)
    img_cnt_key=list(img_cnt.keys())
    max_img_cnt=img_cnt_zip[0][1]
    dec_fac=pow(ib_fac,1/(len(img_cnt_zip)-1))

    
    for i in img_cnt_zip:
        key=img_cnt_key[i[0]]
        real_cnt=int(max_img_cnt)
        img_cnt[key]=min(real_cnt,img_cnt[key])
        max_img_cnt/=dec_fac
        
    for class_name in os.listdir(src_train_path):
        src_cls_path=os.path.join(src_train_path,class_name)
        dst_cls_path=os.path.join(dst_train_path,class_name)
        if not os.path.isdir(dst_cls_path):
            os.mkdir(dst_cls_path)
        res_cnt=0
        res_img=[]
        for img_name in os.listdir(src_cls_path):
            img_path=os.path.join(src_cls_path,img_name)
            res_img.append(Image.open(img_path))
        idxs=np.random.choice(len(res_img),img_cnt[class_name],replace=False)
        for idx in idxs:
            res_img[idx].save(os.path.join(dst_cls_path,'result'+str(res_cnt)+".jpg"))
            res_cnt+=1
    for class_name in os.listdir(src_test_path):
        src_cls_path=os.path.join(src_test_path,class_name)
        dst_cls_path=os.path.join(dst_test_path,class_name)
        if not os.path.isdir(dst_cls_path):
            os.mkdir(dst_cls_path)
        res_cnt=0
        res_img=[]
        for img_name in os.listdir(src_cls_path):
            img_path=os.path.join(src_cls_path,img_name)
            res_img.append(Image.open(img_path))
        for img in res_img:
            img.save(os.path.join(dst_cls_path,'result'+str(res_cnt)+".jpg"))
            res_cnt+=1
    print("LT Saved")
