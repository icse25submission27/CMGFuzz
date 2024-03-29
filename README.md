# CMGFuzz-demo
The code demo of CMGFuzz

## Installation

We have tested CMGFuzz based on Python 3.11 on Ubuntu 20.04, theoretically it should also work on other operating systems. To get the dependencies of CMGFuzz it is sufficient to run the following command.

`pip install -r requirements.txt`

## Get model
```
python get_model.py -data_dir dataset/OF17_jpg
```

## Running CMGFuzz
`python select_seeds_CMG -dataset OF17  -data_dir dataset/OF17_jpg`
```
python Cluster_CMG.py seeds_jpg/OF17
python  Generate_CMG.py -dataset_path seeds_jpg/OF17 -std 70 -domain flower
python combine.py
python clean.py
```

## OOD data Detection
```
cd VAE_anomaly_detection
python train.py
python Detect.py
python simi.py
python combine.py
```
Remember to change the data_path in each file.  If you want to use your own dataset, implement it in VAE_anomaly_detection/dataset.py

## Coverage
```
cd NeuraL-Coverage
python Coverage_Main.py
```
Remember to change the data_path in each file.

## Baseline
### DistXplore
We implement DistXplore using PyTorch.
You can still run DistXplore as is shown in https://github.com/l1lk/DistXplore
```
cd baselines/DistXplore_PyTorch/
python3  AttackSet_PyTorch.py   -i (your class seed path)  -o (output path)  -pop_num 100  -subtotal 50  -type mnist(this argument is abandoned) -model (your model path)  -target (int: target class)  -max_iteration 20
```

### BIM & PGD
```
cd baselines/ADV/
python attack.py -model_pth models/resnet50_flw17.pth -data_dir seeds/OF17 -std_cnt 70
```

### SENSEI
We implement SENSEI using PyTorch.
```
cd baselines/SENSEI/
python trainSENSEI.py -dataset OF17  -data_dir seeds_jpg/OF17
```
