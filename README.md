# Training-Configs for multi-datasets

## Quick-start
### 1. Clone the repo or copy the config-file
### 2. [Download the coco-format annotations](./data/annotations/annotations_bdd/README.md)
### 3. Modify the "ann_file" and "img_prefix" in the config-file (for the train, validation and test)
e.g. For training on BDD:   
train=dict(  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type=dataset_type,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ann_file='mmdetections/data/annotations/annotations_bdd/bdd_100k_train.json'  ,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img_prefix='mmdetections/data/bdd_100k/images/100k/train/'',  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pipeline=train_pipeline,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;classes=classes,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)
               
### 4. Training on single-gpu:
```shell
python tools/train.py /path/to/your/config/
```
### 5. Parallel-training on multiple-gpu:
```shell
bash tools/dist_train.sh /path/to/your/config/ num_gpus 
```
