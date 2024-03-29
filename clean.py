from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, pairwise_distances
import seaborn as sns
import numpy as np
from torch.utils.data import random_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
data_dir = "CMGFuzz/seeds_jpg/OF17_result"
std=70

def dict_to_dataloader(data_by_label, batch_size):
    datasets = []
    for label, data in data_by_label.items():
        images = torch.stack(data["images"])
        labels = torch.Tensor(data["labels"]).long()
        datasets.append(TensorDataset(images, labels))
    
    combined_dataset = ConcatDataset(datasets)
    
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def data_to_dict(image_dataset):
    data_by_label = {}  

    for i in range(len(image_dataset)):
        image, label = image_dataset[i]  
        label = int(label)  

        if label not in data_by_label:
            data_by_label[label] = {"images": [], "labels": []}

        data_by_label[label]["images"].append(image)
        data_by_label[label]["labels"].append(label)
    return data_by_label

def sin_data_to_dict(image_dataset, std_label):
    sin_data_by_label = {} 

    for i in range(len(image_dataset)):
        image, label = image_dataset[i]  #
        label = int(label)  
        if label!=std_label:
            continue
        sin_data_by_label["images"].append(image)
        sin_data_by_label["labels"].append(label)
    return sin_data_by_label
        
def extract_data_by_label(dataloader):
    data_by_label = {}

    for inputs, labels in dataloader:
        for i, label in enumerate(labels):
            label = label.item()
            if label not in data_by_label:
                data_by_label[label] = {"images": [], "labels": []}

            data_by_label[label]["images"].append(inputs[i])
            data_by_label[label]["labels"].append(label)

    return data_by_label


def extract_features(input, model, device):
    input = torch.stack(input).to(device) 
    model.eval()
    with torch.no_grad():
        x = model.conv1(input)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
    model.train()
    features = torch.flatten(x, 1).squeeze(0)
    features=features.cpu().numpy()
    return features

def calc_drop(data_by_label):
    res=dict()
    for key,value in data_by_label.items():
        value=len(value['images'])
        value=int(0.2*value)
        res[key]=value
    return res
def L2vec(a,b):
    res=0
    for i in a:
        for j in b:
            res+=(i-j)**2
    return res
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
    ]),
    'result':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# mode=["few", "medium", ""]
if 1:
    # 创建逆变换
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    verse_transform=transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    img_transformer=transforms.ToPILImage()
    ts_transformer=transforms.ToTensor()
    
    tmp_dir=os.path.join(data_dir, "output")
    data_dir="LTOrg"
    output_dir=os.path.join(data_dir,'cleanedCRA')
    image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'add': datasets.ImageFolder(os.path.join(data_dir, 'result'),, data_transforms['result']),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    }
    
    
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, num_workers=4) for x in ['train', 'test', 'add']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'add']}

    num_classes = len(image_datasets['train'].classes)
    cls_to_idx = image_datasets['train'].class_to_idx
    idx_to_cls = {j:i for i,j in cls_to_idx.items()}
    print(cls_to_idx)
    for cls in cls_to_idx.keys():
        p=os.path.join(output_dir,cls)
        os.makedirs(p,exist_ok=True)
    
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cnt_dict=dict()
    epochs=25
    train_data_by_label = data_to_dict(image_datasets['train']) 
    # print(0)
    # gen_data_by_label = data_to_dict(image_datasets['add'])
    # print(1)
    # drop_threshold=calc_drop(gen_data_by_label)
    
    val_accuracies = []
    n=1
    drop_idxs=[]
    drop_threshold=dict()
    drop_rank=0.05
    cls_drop_threshold=0.3
    droped=dict()
    drop_mut=0.05
    cnt_midict=dict()
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        add_running_loss = 0.0
        for x in ['train']:
            d=image_datasets[x]
            # 生成一个随机的索引序列
            perm = torch.randperm(len(d))
            # 使用这个索引序列来创建一个Subset
            image_datasets[x] = Subset(d, perm.tolist())
        for x in ['add']:
            d=image_datasets[x]
            # 生成一个随机的索引序列
            perm = torch.randperm(len(d))
            rang=list(range(len(d)))
            pos_dict=dict(zip(perm.tolist(),rang))
            # 使用这个索引序列来创建一个Subset
            image_datasets[x] = Subset(d, perm.tolist())
            # print(drop_idxs)
            drop_idxs = [pos_dict[idx] for idx in drop_idxs]
        dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=32, num_workers=4)
        dataloaders['add'] = DataLoader(image_datasets['add'], batch_size=32, num_workers=4)
        print(1)
        for inputs, labels in dataloaders['train']:
            
            if epoch==0:
                for _,l in enumerate(labels):
                    cnt_dict[l.item()]=cnt_dict.get(l.item(),0)+1
                    cnt_midict[l.item()]=cnt_dict.get(l.item(),0)+1
                    drop_threshold[l.item()]=drop_threshold.get(l.item(),0)+1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        for batch_idx, (inputs, labels) in enumerate(dataloaders['add']):
            if epoch==0:
                for _,l in enumerate(labels):
                    cnt_midict[l.item()]=cnt_midict.get(l.item(),0)+0.5
                    drop_threshold[l.item()]=drop_threshold.get(l.item(),0)+1
            if epoch>=5:
                batch_size = inputs.size(0)
                start_idx=batch_idx*batch_size
                end_idx=(batch_idx+1)*batch_size
                now_drop_idxs=[i-start_idx for i in drop_idxs if i>=start_idx and i<end_idx]
                drop_mask = torch.ones(inputs.size(0), dtype=torch.bool)
                drop_mask[now_drop_idxs] = False
                org_inputs=inputs
                org_labels=labels
                inputs = inputs[drop_mask]
                labels = labels[drop_mask]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            add_running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {((running_loss +add_running_loss) / (dataset_sizes["add"]+dataset_sizes["train"]-len(drop_idxs))):.4f}')
        losses.append(((running_loss +add_running_loss) / (dataset_sizes["add"]+dataset_sizes["train"]-len(drop_idxs))))
        exp_lr_scheduler.step()  
        if epoch==0:
            for key, value in drop_threshold.items():
                drop_threshold[key]=value*cls_drop_threshold
        if epoch>=5:
            clus_features=dict()
            means=dict()
            sigma=dict()
            for key,value in train_data_by_label.items():
                value=value['images']
                features=[]
                for v in value:
                    features.append(extract_features([v],model,device))
                scaler = StandardScaler()
                scaled_original_features = scaler.fit_transform(features)
                means[key] = scaler.mean_
                sigma[key] = np.sqrt(scaler.var_)
                if cnt_dict[key]<20:
                    if len(features)>10:
                        k = 5 
                        kmeans = KMeans(n_clusters=k)
                        # kmeans.fit(scaled_original_features)
                        cluster_labels = kmeans.fit_predict(scaled_original_features)
                        cluster_centers = kmeans.cluster_centers_
                    
                    clus_features[key]=[]
                    if len(features)>10:
                        for i in cluster_centers:
                            clus_features[key].append(i.tolist())
                    else:
                        for i in scaled_original_features:
                            clus_features[key].append(i.tolist())
            
            print(means)
            print(sigma)
            start=0
            end=len(image_datasets['add'])
            to_drop_idxs=[]
            to_drop_dis=[]
            s3_drop_idxs=[]
            s3_drop_dis=[]
            start=int(epoch%15*(len(image_datasets['add'])/15))
            end=int((epoch+1)%15*(len(image_datasets['add'])/15))
            if end==0:
                end=len(image_datasets['add'])
            for i in range(start,end):
                sample, label = image_datasets['add'][i]
                if drop_threshold[label]<=0:
                    continue
                # sample = sample.unsqueeze(0)
                if cnt_dict[label]<20:
                    feature = extract_features([sample], model, device)
                    # scaled_generated_feature = scaler.transform(feature)
                    feature=feature-means[label]
                    if (sigma[label]>0).all():
                        feature=feature/sigma[label]
                    now_fea=feature.tolist()
                    min_dis=-1
                    for j in clus_features[label]:
                        distance=L2vec(j,now_fea)
                        if min_dis<0:
                            min_dis=distance
                        else:
                            min_dis=min(min_dis,distance)
                    to_drop_idxs.append(i)
                    to_drop_dis.append(min_dis)
                else:
                    feature = extract_features([sample], model, device)
                    lower_bound=means[label]-3*sigma[label]
                    upper_bound=means[label]+3*sigma[label]
                    to_drop=((lower_bound > feature) | (feature > upper_bound))
                    if not to_drop.any():
                        continue
                    else:
                        min_dis = np.sum(to_drop)
                        s3_drop_idxs.append(i)
                        s3_drop_dis.append(min_dis)
            # print(to_drop_idxs)
            to_drop_zip=list(zip(to_drop_idxs,to_drop_dis))
            to_drop_zip=sorted(to_drop_zip, key=lambda x:x[1], reverse=True)
            to_drop_idxs=[i[0] for i in to_drop_zip if i[0] not in drop_idxs]
            s3_drop_zip=list(zip(s3_drop_idxs,s3_drop_dis))
            s3_drop_zip=sorted(s3_drop_zip, key=lambda x:x[1], reverse=True)
            s3_drop_idxs=[i[0] for i in s3_drop_zip if i[0] not in drop_idxs]
            nsr=len(to_drop_idxs)/(len(to_drop_idxs)+len(s3_drop_idxs))
            sr=len(s3_drop_idxs)/(len(to_drop_idxs)+len(s3_drop_idxs))
            # print(to_drop_zip)
            # print(to_drop_idxs)
            for i in range(int(len(to_drop_idxs)*drop_rank*nsr/10)):
                # p=np.random.rand()
                # if p<drop_mut:
                #     continue
                # print(1)
                sample, label=image_datasets['add'][to_drop_idxs[i]]
                if drop_threshold[label.item()]:
                    drop_idxs.append(to_drop_idxs[i])
                    drop_threshold[label.item()]-=1
                    droped[label.item()]=droped.get(label.item(),0)+1
                    cnt_midict[label.item()]-=0.5
            for i in range(int(len(s3_drop_idxs)*drop_rank*sr/10)):
                # p=np.random.rand()
                # if p<drop_mut:
                #     continue
                sample, label=image_datasets['add'][s3_drop_idxs[i]]
                if drop_threshold[label.item()]:
                    drop_idxs.append(s3_drop_idxs[i])
                    drop_threshold[label.item()]-=1
                    droped[label.item()]=droped.get(label.item(),0)+1
                    cnt_midict[label.item()]-=0.5
    print("Clean End")
    
    selected_idxs=list(range(len(image_datasets['add'])))
    selected_idxs=list(set(selected_idxs).difference(set(drop_idxs)))
    cleaned_add=Subset(image_datasets['add'],selected_idxs)
    dataloaderX=DataLoader(cleaned_add, shuffle=True, batch_size=32, num_workers=5)
    cnt=0
    for inputs, labels in dataloaderX:
        for i in range(len(inputs)):
            print(1,end='')
            clsidx=labels[i].item()
            if cnt_dict[clsidx]<70:
                cnt_dict[clsidx]+=1
            else:
                continue
            clsn=idx_to_cls[clsidx]
            pth=os.path.join(output_dir,clsn)
            pth=os.path.join(pth, str(cnt)+".jpg")
            cnt+=1
            img=inputs[i]
            img=inverse_transform(img)
            img=img_transformer(img)
            img.save(pth)
    
