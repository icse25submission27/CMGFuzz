import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import yaml
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model.VAE import VAEAnomalyTabular
from dataset import rand_dataset
import dataset
from torchvision import datasets, models, transforms
import os 
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

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 创建逆变换
verse_transform=transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
])


#load X_test&model
model_pth="/home/ubuntu/hbl/largeD/VAE_anomaly_detection/saved_models/flwLT/{epoch:02d}-{val_loss:.2f}/last.ckpt"
model = VAEAnomalyTabular.load_from_checkpoint(model_pth, input_size=3, latent_size=50)
# load saved parameters from a run
X_test=dataset.flw17_dataset_add('')
train_dloader = DataLoader(X_test, 32)
cls_to_idx={'bluebell': 0, 'buttercup': 1, 'colts_foot': 2, 'cowslip': 3, 'crocus': 4, 'daffodil': 5, 'daisy': 6, 'dandelion': 7, 'fritillary': 8, 'iris': 9, 'lily_valley': 10, 'pansy': 11, 'snowdrop': 12, 'sunflower': 13, 'tigerlily': 14, 'tulip': 15, 'windflower': 16}
idx_to_cls={0: 'bluebell', 1: 'buttercup', 2: 'colts_foot', 3: 'cowslip', 4: 'crocus', 5: 'daffodil', 6: 'daisy', 7: 'dandelion', 8: 'fritillary', 9: 'iris', 10: 'lily_valley', 11: 'pansy', 12: 'snowdrop', 13: 'sunflower', 14: 'tigerlily', 15: 'tulip', 16: 'windflower'}
cls_to_idx=X_test.class_to_idx
idx_to_cls={j:i for i,j in cls_to_idx.items()}

output_dir="./CMGOOD"
output_ID="./CMGID"
os.makedirs(output_dir,exist_ok=True)

cnt=0
sum=0
c=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for inputs,labels in train_dloader:
    sum+=len(inputs)
    inputs=inputs.to(device)
    outliers = model.is_anomaly(inputs)
    print(outliers)
    outliers=outliers.cpu()
    for i in range(len(outliers)):
        if outliers[i]:
            idx=labels[i]
            cls_dir=os.path.join(output_dir,idx_to_cls[idx.item()])
            os.makedirs(cls_dir,exist_ok=True)
            t=inverse_transform(inputs[i])
            t=transforms.ToPILImage()(t)
            t.save(os.path.join(cls_dir,str(c)+'.jpg'))
            c+=1
        else:
            idx=labels[i]
            cls_dir=os.path.join(output_ID,idx_to_cls[idx.item()])
            os.makedirs(cls_dir,exist_ok=True)
            t=inverse_transform(inputs[i])
            t=transforms.ToPILImage()(t)
            t.save(os.path.join(cls_dir,str(c)+'.jpg'))
            c+=1
    cnt+=torch.sum(outliers)

print("Ratio of OOD:", cnt/sum)
