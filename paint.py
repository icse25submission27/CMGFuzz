import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from PIL import Image
from matplotlib.lines import Line2D

# 定义文件夹和颜色
lb= ["CMGFuzz", "SENSEI", "DistXplore","Train", "Full"]
folders = ["CMGFuzz", "SENSEI", "DistXplore", "train",  'real']
colors = ['#D8383A', "#2ECC71", "#3498DB", '#9B59B6', '#95A5A6']
markers = ['s', '.', 'v', '.', '*']
# 初始化t-SNE模型
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, perplexity=200)

x=0.5
data = []
labels = []
for i, folder in enumerate(folders):
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if np.random.rand()<0.5:
                    continue
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img = Image.open(os.path.join(subfolder_path, filename)).resize((224,224))
                    img_array = np.array(img).flatten()
                    data.append(img_array)
                    labels.append(i)
idxs=np.random.choice(len(labels),1000,replace=False)
print(1)
X = np.array(data)
labels=np.array(labels)
plt.figure(figsize=(8, 8))

for kk in range(2):  
    sample_idxs = np.random.choice(len(X), size=1500, replace=False)  
    X_sample = X[sample_idxs]
    labels_sample = labels[sample_idxs]

    # 使用t-SNE进行降维
    X_tsne = tsne.fit_transform(X_sample)
    if kk==0:
        X0=X_tsne[0]
    if kk==1:
        X1=X_tsne[0]
    else:
        X2=X_tsne[0]
    # 绘tu
    for i in range(X_tsne.shape[0]):
        if labels_sample[i]==0:
            continue
        if labels_sample[i]==1 or labels_sample[i]==2:
            continue
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=colors[labels_sample[i]], s=20,marker=markers[labels_sample[i]])
    for i in range(X_tsne.shape[0]):
        if labels_sample[i]!=0:
            continue
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=colors[labels_sample[i]], s=20,marker=markers[labels_sample[i]])
# plt.scatter(X0[0], X0[1], s=50, color=colors[labels_sample[0]])
# plt.scatter(X1[0], X1[1], s=50, color=colors[labels_sample[0]])
# plt.scatter(X2[0], X2[1], s=50, color=colors[labels_sample[0]])

# print(0)
# # 使用t-SNE进行降维
# X_tsne = tsne.fit_transform(X)


# plt.figure(figsize=(8, 8))
# for i in range(X_tsne.shape[0]):
#     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=colors[labels[i]])

legend_elements = [Line2D([0], [0], marker=marker, color='w', label=lbs, 
                          markerfacecolor=color, markersize=10, markeredgewidth=2) for marker, lbs, color in zip(markers[0:1]+markers[3:], lb[0:1]+lb[3:], colors[0:1]+colors[3:])]

plt.legend(handles=legend_elements)
plt.savefig('scatter_plot_cmg.png')  
plt.show()