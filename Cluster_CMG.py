import os
import torch
import torchvision
from torchvision import transforms
from sklearn.cluster import KMeans
from torchvision.utils import save_image
import shutil
from PIL import Image

input_path='/home/ubuntu/hbl/largeD/Flower17/LT2/train'

def getAllChildren(path):
    res=[]
    isChidren=1
    for file_path in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_path)):
            res+=getAllChildren(os.path.join(path, file_path))
            isChidren=0
        
    if isChidren:
        res+=[path]
    return res
        

model = torchvision.models.resnet50(pretrained=True)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 创建逆变换
verse_transform=transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
])


data_paths=getAllChildren(input_path)
print(data_paths)
for data_path in data_paths:
    if 'tmp' in data_path:
        continue
    # 创建目录保存聚类图像
    output_dir = os.path.join(data_path,'clusRes')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    
    # 加载图像数据集
    tmp_path=os.path.join(input_path, 'tmp')
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    tmp_parents_dir=os.path.join(tmp_path,"parent")
    if not os.path.isdir(tmp_parents_dir):
        os.mkdir(tmp_parents_dir)
    cnt=0
    for img_path in os.listdir(data_path):
        img_path=os.path.join(data_path,img_path)
        img=Image.open(img_path)
        res_path=os.path.join(tmp_parents_dir,"result"+str(cnt)+".jpg")
        img.save(res_path)
        cnt+=1

    
    
    dataset = torchvision.datasets.ImageFolder(tmp_path,transform=transform)

    # 提取图像特征
    features = []
    for image, _ in dataset:
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            feature = model(image)
        features.append(feature.squeeze().numpy())

    # 转换特征为Tensor
    features = torch.tensor(features)

    # 聚类
    kmeans = KMeans(n_clusters=3)  # 设置聚类簇的数量
    cluster_labels = kmeans.fit_predict(features)
    print(cluster_labels)

    
    os.makedirs(output_dir, exist_ok=True)
    # 按分类保存图像
    for i, (image, _) in enumerate(dataset):
        cluster_label = cluster_labels[i]
        cluster_dir = os.path.join(output_dir, f'Cluster_{cluster_label}')
        os.makedirs(cluster_dir, exist_ok=True)
        image_name = f'image_{i}.jpg'
        image_path = os.path.join(cluster_dir, image_name)
        save_image(inverse_transform(image),image_path)

        print(f'Saved image {image_path}')
    shutil.rmtree(tmp_path)
    print(data_path)
