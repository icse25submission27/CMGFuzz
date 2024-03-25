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
import sys
import numpy as np
import PIL
from torch.utils.data import Subset
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.transforms import functional
import time
import argparse

def AnyRand(low,high):
    res=np.random.rand()
    times=high-low
    res*=times
    res+=low
    return res

class GA:
    def __init__(self, converts, scopes, popsize, prob):
        self.converts=converts
        self.scopes=scopes
        self.popsize=popsize
        self.prob=prob

        while(len(scopes)<len(converts)):
            scopes.append((0,10))
        self.Gsize=len(scopes)
        self.InitPop()
    
    def InitPop(self):
        pop=[]
        while len(pop)<self.popsize:
            newGene=[]
            for scope in self.scopes:
                newGene.append(AnyRand(scope[0],scope[1]))
            pop.append(newGene)
        return pop
    
    def selectParents(self,pop):
        l=len(pop)
        p=np.random.choice(l,2,replace=False)
        return pop[p[0]],pop[p[1]]
    
    def selectParent(self,pop):
        l=len(pop)
        p=np.random.choice(l,1,replace=False)
        return pop[p[0]]
    
    def crossover(self,p1,p2):
        r=np.random.randint(0,self.Gsize)
        newGene=p1[0:r+1]
        newGene+=p2[r+1:]
        #print(newGene)
        return newGene
    
    def mutate(self,p):
        op=np.random.randint(0,self.Gsize)
        p[op]=AnyRand(self.scopes[op][0],self.scopes[op][1])
        return p
    
    def GenPop(self,pop):
        if len(pop)==0:
            return
        newPop=[]
        while len(newPop)<self.popsize:
            r=np.random.rand()
            if r<self.prob:
                p=self.selectParent(pop)
                child=self.mutate(p)
                newPop.append(child)
            else:
                p1,p2=self.selectParents(pop)
                child=self.crossover(p1,p2)
                newPop.append(child)
        return newPop
    def tran(self,img,op,oprd):
        if op=="rotate":
            return img.rotate(oprd,expand=True)
        if op=="translate":
            width,height=img.size 
            r=np.random.choice([0,1],1)
            if r:
                offset=int(width*oprd)
                new_img=PIL.ImageChops.offset(img,offset,0)
                new_img.paste("black",(0,0,offset,height))
                return new_img
            else:
                offset=int(height*oprd)
                new_img=PIL.ImageChops.offset(img,0,offset)
                new_img.paste("black",(0,0,width,offset))
                return new_img
        if op=="shear":
            width,height=img.size
            x=int(width*oprd)
            if x<0:
                x=-x
                new_img=img.crop((x,0,width,height))
            else:
                new_img=img.crop((0,0,width-x,height))
            #print(oprd," ",width," ",x)
            return new_img
        if op=="zoom":
            width,height=img.size 
            width=int(oprd*width)
            height=int(oprd*height)
            new_img=img.resize((width, height), PIL.Image.ANTIALIAS)
            return new_img

        if op=="bright":
            brightEnhancer = PIL.ImageEnhance.Brightness(img)
            new_img = brightEnhancer.enhance(oprd)
            return new_img
        if op=="contrast":
            contrastEnhancer = PIL.ImageEnhance.Contrast(img)
            new_img = contrastEnhancer.enhance(oprd)
            return new_img

    def trans(self, img, gene):
        new_img=img
        # print(self.converts)
        # print(self.scopes)
        # print(gene)
        for i in range(len(gene)):
            new_img=self.tran(new_img,self.converts[i],gene[i])
        return new_img
    def select(self,children, fitness):
        children=np.array(children)
        fitness=np.array(fitness)
        idx=np.arange(self.popsize)
        idx=np.random.choice(idx,size=self.popsize,replace=True,p=(fitness)/(fitness.sum()))
        return children[idx].tolist()
    def selectBest(self,children, fitness):
        maxval=-100
        maxidx=0
        for idx in range(len(fitness)):
            if maxval<fitness[idx]:
                maxval=fitness[idx]
                maxidx=idx
        return maxidx
    def selectBest_n(self,children, fitness, n):
        idx=list(range(len(fitness)))
        idxwf=list(zip(idx,fitness))
        idxwf.sort(key=lambda x: x[1],reverse=True)
        result=[]
        for i in range(n):
            result.append(idxwf[i][0])
        return result












losses = []
parser = argparse.ArgumentParser(description='Input the data_dir and the seed_num')
parser.add_argument('-dataset',type=str, choices=['OF17', 'IN100', 'FOOD101'],default='OF17')
parser.add_argument('-data_dir', type=str)

args = parser.parse_args()
data_dir = args.data_dir
savedStdout = sys.stdout  

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

# mode=["few", "medium", ""]

if 1:
    start=time.time()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=32, num_workers=4)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    num_classes = len(image_datasets['train'].classes)
    
    cls2idx=image_datasets['train'].class_to_idx
    idx2cls={j:i for i,j in cls2idx.items()}
    
    output_dir=os.path.join(os.getcwd(),args.dataset)
    os.makedirs(output_dir,exist_ok=True)
    for i in image_datasets['train'].classes:
        os.makedirs(os.path.join(output_dir,i),exist_ok=True)

    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cnt_dict=dict()
    phase_cnt={'few':0,"medium":0,"many":0}
    phase_acc={'few':0,"medium":0,"many":0}
    ExGA=GA(["rotate","translate","shear","zoom","bright","contrast"],[(-30,30),(-0.1,0.1),(-0.1,0.1),(0.9,1.1),(0.8,1.2),(0.8,1.2)],15,0.001)
    pops=[]
    best_g=[]
    for i in range(dataset_sizes['train']):
        pops.append(ExGA.InitPop())
        best_g.append([])
    
    org_rang=list(range(dataset_sizes['train']))
    rang=list(range(dataset_sizes['train']))
    pos_dict=dict(zip(org_rang,rang))
    epochs=25
    img_transformer=transforms.ToPILImage()
    ts_transformer=transforms.ToTensor()
    out_cnt={i:0 for i in range(num_classes)}
    for epoch in range(epochs):
        for i in out_cnt.keys():
             out_cnt[i]=0
        model.train()
        running_loss = 0.0
        for x in ['train']:
            d=image_datasets[x]
            # 生成一个随机的索引序列
            perm = torch.randperm(len(d))
            k_rang=list(pos_dict.keys())
            v_rang=list(pos_dict.values())
            rang=list(range(len(d)))
            # new_pos=dict(zip(perm.tolist(),rang))
            # 使用这个索引序列来创建一个Subset
            image_datasets[x] = Subset(d, perm.tolist())
            new_pos=dict(zip(rang,perm.tolist()))
            new_v_rang=[v_rang[new_pos[idx]] for idx in k_rang]
            pos_dict=dict(zip(k_rang,new_v_rang))
            # new_k_rang = [new_pos[idx] for idx in k_rang]
            # pos_dict=dict(zip(new_k_rang,v_rang))
        dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=32, num_workers=4)
        for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
            for _,l in enumerate(labels):
                cnt_dict[l.item()]=cnt_dict.get(l.item(),0)+1
            for i in range(len(inputs)):
                img=inputs[i]
                img=inverse_transform(img)
                img=img_transformer(img)
                pos=pos_dict[batch_idx*32+i]
                children=ExGA.GenPop(pops[pos])
                imgClass=labels[i]
                fitness=[]
                newimg=[]
                for child in children:
                    chimg=ExGA.trans(img, child)
                    newimg.append(chimg)
                    model.eval()
                    with torch.no_grad():
                        topre = ts_transformer(chimg).unsqueeze(0).to(device)
                        p = model(topre)
                    model.train()
                    classL=[0]*len(p[0])
                    classL[imgClass]=1
                    classL=[classL]
                    classL=torch.Tensor(classL)
                    classL=classL.to(device)
                    fitness.append(F.cross_entropy(p,classL).item())
                for kk in range(len(fitness)):
                    fitness[kk]=fitness[kk]
                inputs[i]=verse_transform(ts_transformer((newimg[ExGA.selectBest(children, fitness)].resize((224,224),PIL.Image.ANTIALIAS))))
                if np.array(fitness).sum()!=0.0:
                    pops[pos]=ExGA.select(children,fitness)
                best_g[pos]=children[ExGA.selectBest(children, fitness)]
                out_cls_dir=os.path.join(output_dir,idx2cls[int(imgClass.item())])
                sel_idx=ExGA.selectBest_n(children, fitness, 10)
                for i in sel_idx:
                    newimg[i].save(os.path.join(out_cls_dir, str(epoch)+"_"+str(out_cnt[int(imgClass.item())])+".jpg"))
                    out_cnt[int(imgClass.item())]+=1
                    
                    
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / dataset_sizes["train"]:.4f}')
        end=time.time()
        print(end-start)
        losses.append(running_loss / dataset_sizes["train"])
        
    for key in cnt_dict.keys():
        cnt_dict[key]/=epochs
    end=time.time()
    print("\n\n\ntiming: ",end-start)
    