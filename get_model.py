import shutil
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
from torch.optim import lr_scheduler
import numpy as np
import argparse
losses = []


parser = argparse.ArgumentParser(description='Input the data_dir and the seed_num')
parser.add_argument('-data_dir', type=str)
args = parser.parse_args()
data_dir = args.data_dir

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

if 1:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    num_classes = len(image_datasets['train'].classes)

    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cnt_dict=dict()
    for inputs, labels in dataloaders['train']:
        for _,l in enumerate(labels):
            cnt_dict[l.item()]=cnt_dict.get(l.item(),0)+1
    epochs=25
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / dataset_sizes["train"]:.4f}')
        losses.append(running_loss / dataset_sizes["train"])
        exp_lr_scheduler.step()
    # test
    model.eval()
    correct = 0
    total = 0

    correct_dict=dict()
    test_cnt_dict=dict()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            for i in range(len(predicted)):
                test_cnt_dict[labels[i].to("cpu").item()]=test_cnt_dict.get(labels[i].to("cpu").item(),0)+1
                if predicted[i]==labels[i]:
                    correct_dict[labels[i].to("cpu").item()]=correct_dict.get(labels[i].to("cpu").item(),0)+1
                    correct+=1
    
    # cm = confusion_matrix(all_labels, all_preds)

    for key,value in correct_dict.items():
        correct_dict[key]/=test_cnt_dict[key]
    
    correct_dict=dict(sorted(correct_dict.items(), key=lambda k:k[0], reverse=True))
    phase_cnt={'few':0,"medium":0,"many":0}
    phase_acc={'few':0,"medium":0,"many":0}
    for key,value in correct_dict.items():
        if cnt_dict[key]<=20:
            phase_acc['few']+=value
            phase_cnt['few']+=1
        elif cnt_dict[key]<=100:
            phase_acc['medium']+=value
            phase_cnt['medium']+=1
        else:
            phase_acc['many']+=value
            phase_cnt['many']+=1
        print(f"{key}: {value}, {cnt_dict[key]}")
    zcnt=0
    for key,value in cnt_dict.items():
        if not correct_dict.get(key,0):
            print(f"{key}: {0}, {cnt_dict[key]}, {test_cnt_dict[key]}")
            zcnt+=1
    for key,value in phase_acc.items():
        if key=='few':
            phase_cnt[key]+=zcnt
            if phase_cnt[key]:
                phase_acc[key]/=phase_cnt[key]
        else:
            if phase_cnt[key]:
                phase_acc[key]/=phase_cnt[key]
    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%')
    print(f"few: {phase_acc['few']} \n medium: {phase_acc['medium']}\n  many: {phase_acc['many']}")
    if not os.path.isdir("./models"):
        os.makedirs("./models",exist_ok=True)
    torch.save(model, 'models/resnet50_flw17.pth')
