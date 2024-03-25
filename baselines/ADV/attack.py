
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import Counter
import argparse
import foolbox
from foolbox.models import PyTorchModel
from foolbox.attacks import FGSM, PGD, L2CarliniWagnerAttack
from torchvision import transforms



def empty_processing(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # if len(x.shape)>=4: 
    #     x=np.transpose(x, (0, 3, 1, 2))
    # else:
    #     x=np.transpose(x, (2, 0, 1))
    # 创建逆变换
    verse_transform=transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    x=torch.from_numpy(x)
    # x=inverse_transform(x)
    return x


def target_adv_attack(tmodel, seeds, labels, method, para_0, para_1):
    # model = models.resnet18(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    fmodel = foolbox.PyTorchModel(model=tmodel, bounds=(-55, 55))
    # seeds, xs = foolbox.utils.samples(fmodel, dataset='imagenet', batchsize=len(seeds))
    # distance = foolbox.distances.Linfinity
    # criteria = foolbox.criteria.TargetClass
    # criteria = criteria(target_label)
    # print(seeds)
    seeds=seeds.to('cpu')
    labels=labels.to('cpu')
    if method == "bim":
        attack = foolbox.attacks.L2BasicIterativeAttack()
        adversarials,x,y = attack(fmodel,seeds.to("cuda:0"), labels.to("cuda:0"), epsilons=para_0)
        return adversarials
    elif method == "pgd":
        attack = foolbox.attacks.PGD()
        adversarials,x,y = attack(fmodel,seeds.to("cuda:0"), labels.to("cuda:0"), epsilons=para_0)
        return adversarials
    elif method == "cw":
        attack = foolbox.attacks.carlini_wagner()
        adversarials = attack(fmodel,seeds, labels, learning_rate=para_0, initial_const=para_1)
        return adversarials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data_path and model_path")
    parser.add_argument("-model_pth", type=str)
    parser.add_argument("-data_dir", type=str)
    parser.add_argument("-std_cnt", type=int)
    args = parser.parse_args()

    model=torch.load(args.model_pth,map_location=torch.device('cpu'))
    model=model.to("cuda:0")
    model.eval()

    
    # Load pre-generated adversarial seed data
    adv_seed_dir = args.data_dir
    all_class_adv_seeds = []
    std=args.std_cnt
    for file_index in range(len(os.listdir(adv_seed_dir))):
        temp_adv_seeds = np.load(os.path.join(adv_seed_dir, "class_%s_seed.npy" % file_index))
        temp_adv_seeds = empty_processing(temp_adv_seeds)
        all_class_adv_seeds.append(temp_adv_seeds)
    # all_class_adv_seeds = np.array(all_class_adv_seeds)

    for idx, class_data in enumerate(all_class_adv_seeds):
        # print(class_data.max().item())
        print(idx)
        print(len(class_data))
        sys.stdout.flush()
        if 1:
            target_list = np.arange(std)
            target_list = np.delete(target_list, idx)
            class_label = torch.ones(len(class_data), dtype=torch.long) * idx
            # print(class_label)
            # print(idx, torch.mean((torch.argmax(model(class_data), dim=1) == class_label).float()))

            index = 0
            for method in ["bim", "pgd"]:
                adv_save_dir = "./{}/".format(method)
                adv_npy="adv_data_class_{}_0.npy"
                if(os.path.isfile(os.path.join(adv_save_dir,adv_npy))):
                    continue
                cnt=0
                while(cnt<std):
                    kwargs = {}
                    if method == "bim":
                        kwargs['epsilon'] = 0.1
                        kwargs['iteations'] = 10
                    elif method == "pgd":
                        kwargs['epsilon'] = 0.3
                        kwargs['iterations'] = 10
                    elif method == "cw":
                        kwargs['initial_const'] = 1e-2
                        kwargs['learning_rate'] = 5e-3
                    # exit(0)
                    adv = target_adv_attack(model, class_data, class_label, method, kwargs['epsilon'], 10)
                    cnt+=int(len(adv))
                    
                    if adv is not None:
                        # print(idx, target_label, torch.mean((torch.argmax(model(adv), dim=1) == class_label).float()))
                        # print(Counter(np.argmax(model(adv).numpy(), axis=1)))

                        
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_{}_{}.npy".format(idx, index)), adv.cpu().numpy())
                        index += 1