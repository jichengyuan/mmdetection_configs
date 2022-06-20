# Training-Configs for RCV Challenge datasets (COCO, Objects365, OID, Mapillary)

## Quick-start
### 1. Clone the repo or copy the config-file
### 2. Preparing coco-format annotations for [RVC-Challenge](http://www.robustvision.net/) datasets (using VisionKG Label-Space)
### 3. Modify #TODO in the config files, such as number of classes, classes_names (visionkg label-space), path_to_images, and path_to annotation_files
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
or
```shell
python tools/train.py /path/to/your/config/ \
       --work-dir /path/to/your/workdir \
       --cfg-options data.train.ann_file=/path/to/your/train_annotaitons_file/ \
       data.train.img_prefix=/path/to/your/train_images/ \
       data.val.ann_file=/path/to/your/train_annotaitons_file/ \
       data.val.img_prefix=/path/to/your/val_images/
```

### 5. Evaluation on single-gpu:
```shell
bash python tools/test.py \
     /path/to/your/config/ \
     /path/to/your/checkpoints.pth \
     --eval bbox --options "classwise=True" 
```

### 6. Parallel-training on multiple-gpus:
```shell
bash tools/dist_train.sh /path/to/your/config/ num_gpus 
```

### 7. Input [wandb API keys](https://wandb.ai/authorize) for tracking and visualizing pipelines
