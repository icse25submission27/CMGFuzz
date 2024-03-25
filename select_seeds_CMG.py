import argparse
import numpy as np
import os
import shutil

parser = argparse.ArgumentParser(description='Input the data_dir and the seed_num')
parser.add_argument('-dataset',type=str, choices=['OF17', 'IN100', 'FOOD101'],default='OF17')
parser.add_argument('-data_dir', type=str)
parser.add_argument('-seed_num', type=int, default=100)

args = parser.parse_args()

parent_dir=os.getcwd()
output_dir=os.path.join(parent_dir,'seeds_jpg')
output_dir=os.path.join(output_dir,args.dataset)
input_dir=args.data_dir

if not os.path.isdir(input_dir):
    print("Data_dir not exists.")
    exit(0)

os.makedirs(output_dir, exist_ok=True)
for cls_path in os.listdir(input_dir):
    output_cls_path=os.path.join(output_dir, cls_path)
    os.makedirs(output_cls_path, exist_ok=True)
    input_cls_path=os.path.join(input_dir, cls_path)
    imgs=[]
    for img in os.listdir(input_cls_path):
        imgs.append(img)
    idxs=[]
    if len(imgs)>args.seed_num:
        idxs=np.random.choice(len(imgs), args.seed_num, replace=False)
    else:
        idxs=list(range(len(imgs)))
    for idx in idxs:
        img=imgs[idx]
        shutil.copy(os.path.join(input_cls_path, img), os.path.join(output_cls_path, img))

    
# for seed_path in os.listdir(input_dir):
#     seed_path_whole=os.path.join(input_dir,seed_path)
#     seed=np.load(seed_path_whole)
#     if len(seed)>args.seed_num:
#         idxs=np.random.choice(len(seed), args.seed_num, replace=False)
#     else:
#         idxs=list(range(len(seed)))
#     out_seed=[]
#     for idx in idxs:
#         out_seed.append(seed[idx])
#     out_seed=np.array(out_seed)
#     save_path=os.path.join(output_dir,seed_path)
#     np.save(save_path,out_seed)
    


