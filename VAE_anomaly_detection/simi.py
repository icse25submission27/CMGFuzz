import torch
import clip
from PIL import Image
import os
import shutil
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)



text = clip.tokenize(["a photo of flower", "not a photo of flower"]).to(device)

dist_dir="./distOOD"
dist_out="./distValid"
our_dir="./CMGOOD"
our_out="./CMGValid"
sensei_dir="./senseiOOD"
sensei_out="./senseiValid"
bim_dir="./bimOODt"
bim_out="./bimValidt"
cnt=0
scnt=0
sum=0

for i in os.listdir(our_dir):
    clsp=os.path.join(our_dir,i)
    oclsp=os.path.join(our_out,i)
    os.makedirs(oclsp,exist_ok=True)
    for j in os.listdir(clsp):
        imagep=os.path.join(clsp,j)
        image=transform(Image.open(imagep)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        if probs[1]>probs[0]:
            cnt+=1
        else:
            shutil.copy(imagep,os.path.join(oclsp,j))
        sum+=1
print("Valid data in OOD {}",cnt/sum)
