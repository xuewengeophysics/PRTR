# Pose Recognition with Cascade Transformers笔记

+ Paper: [Pose Recognition with Cascade Transformers](https://arxiv.org/abs/2104.06976)
+ Code: [mlpc-ucsd/PRTR](https://github.com/mlpc-ucsd/PRTR)



query是啥？输入有  输出也有



|      |         TransPose          |                 PRTR                  |
| ---- | :------------------------: | :-----------------------------------: |
|      |       heatmap-based        |           regression-based            |
|      | discrete coordinate system |      continuous coordinate space      |
|      |     two-stage process      | two-stage process, end-to-end fashion |
|      |  only transformer encoder  |      transformer encoder-decoder      |
|      |                            |                                       |



Spatial Transformer Network (STN)   是啥？

each query dynamically predicts its preferred keypoint type？ 



先把两阶段模型中的关键点检测训练

## 1. Introduction

### 1.1 Why

#### 	1.1.1 Motivation

+ 

#### 1.1.2 Challenges

+ 

### 1.3 How

+ 

### 1.4 Contributions

+ 



## 2. Approach



## 3. Code=f(method)

+ 以COCO train2017 dataset为例：

```shell
python tools/train.py \
    --cfg experiments/coco/transformer/w32_384x288_adamw_lr1e-4.yaml
```

+ 以two_stage/experiments/coco/transformer/w32_384x288_adamw_lr1e-4.yaml为例：

```yaml
AUTO_RESUME: true
CUDNN:
  BENCHMARK: true
  DETERMINISTIC: false
  ENABLED: true
DATA_DIR: ''
GPUS: (0,1,2,3)
OUTPUT_DIR: 'output'
LOG_DIR: 'log'
WORKERS: 8
PRINT_FREQ: 50

DATASET:
  COLOR_RGB: true
  DATA_FORMAT: jpg
  DATASET: 'coco'
  ROOT: 'data/coco/'
  TEST_SET: 'val2017'
  TRAIN_SET: 'train2017'
  FLIP: true
  ROT_FACTOR: 40
  SCALE_FACTOR: 0.3
MODEL:
  INIT_WEIGHTS: true
  NAME: 'pose_transformer'
  PRETRAINED: 'models/pytorch/imagenet/hrnetv2_w32_imagenet_pretrained.pth'
  TARGET_TYPE: 'coord'  ##应用坐标回归
  HEATMAP_SIZE:
  - 72
  - 96
  IMAGE_SIZE:
  - 288
  - 384
  NUM_JOINTS: 17
  EXTRA:
    STAGE1:
      NUM_MODULES: 1
      NUM_RANCHES: 1
      BLOCK: BOTTLENECK
      NUM_BLOCKS:
      - 4
      NUM_CHANNELS:
      - 64
      FUSE_METHOD: SUM
    STAGE2:
      NUM_MODULES: 1
      NUM_BRANCHES: 2
      BLOCK: BASIC
      NUM_BLOCKS:
      - 4
      - 4
      NUM_CHANNELS:
      - 32
      - 64
      FUSE_METHOD: SUM
    STAGE3:
      NUM_MODULES: 4
      NUM_BRANCHES: 3
      BLOCK: BASIC
      NUM_BLOCKS:
      - 4
      - 4
      - 4
      NUM_CHANNELS:
      - 32
      - 64
      - 128
      FUSE_METHOD: SUM
    STAGE4:
      NUM_MODULES: 3
      NUM_BRANCHES: 4
      BLOCK: BASIC
      NUM_BLOCKS:
      - 4
      - 4
      - 4
      - 4
      NUM_CHANNELS:
      - 32
      - 64
      - 128
      - 256
      FUSE_METHOD: SUM
    ENC_LAYERS: 6  ##编码网络层数
    DEC_LAYERS: 6  ##解码网络层数
    DIM_FEEDFORWARD: 2048  ##FFN网络
    DROPOUT: 0.0
    NHEADS: 8  ##Multi-Head Attention的头数
    NUM_QUERIES: 100  ##keypoint queries
    HIDDEN_DIM: 256
    PRE_NORM: false
    AUX_LOSS: true
    NUM_LAYERS: hrnet
    DILATION: false
    POS_EMBED_METHOD: 'sine'
    EOS_COEF: 0.1
    KPT_LOSS_COEF: 5.0
LOSS:
  USE_TARGET_WEIGHT: true
TRAIN:
  BATCH_SIZE_PER_GPU: 16
  SHUFFLE: true
  BEGIN_EPOCH: 0
  END_EPOCH: 200
  OPTIMIZER: 'adamW'
  LR: 1.0e-4
  LR_BACKBONE: 1.0e-4
  CLIP_MAX_NORM: 0.1
  LR_FACTOR: 0.5
  LR_STEP:
  - 70
  - 95
  WD: 0.0001
  GAMMA1: 0.99
  GAMMA2: 0.0
  MOMENTUM: 0.9
  NESTEROV: false
TEST:
  BATCH_SIZE_PER_GPU: 32
  COCO_BBOX_FILE: 'data/coco/person_detection_results/COCO_val2017_detr_detections.json'
  BBOX_THRE: 1.0
  IMAGE_THRE: 0.0
  IN_VIS_THRE: 0.2
  MODEL_FILE: ''
  NMS_THRE: 1.0
  OKS_THRE: 0.9
  FLIP_TEST: true
  POST_PROCESS: true
  SHIFT_HEATMAP: true
  USE_GT_BBOX: true
DEBUG:
  DEBUG: true
  SAVE_BATCH_IMAGES_GT: true
  SAVE_BATCH_IMAGES_PRED: true
  SAVE_HEATMAPS_GT: false
  SAVE_HEATMAPS_PRED: false
```



### 3.1 Input

### 3.2 Process

### 3.3 Output



## 参考资料



更新于2021-05-14

