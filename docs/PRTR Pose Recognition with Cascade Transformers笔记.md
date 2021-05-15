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

+ 基于热图的关键点检测方法需要各种启发式设计，大多数情况下不能做到端到端；而基于回归的方法具有较少的不可微分的中间过程，它去除了一些复杂的前/后处理步骤、需要的启发式设计更少。(In general, heatmap-based methods achieve higher accuracy but are subject to various heuristic designs (not end-to-end mostly), whereas regression-based approaches attain relatively lower accuracy but they have less intermediate non-differentiable steps. It removes complex pre/post-processing procedures and requires fewer heuristic designs compared with existing heatmap-based approaches.)

#### 1.1.2 Challenges

+ 

### 1.3 How

+ 

### 1.4 Contributions

+ 



### 1.5 Future works

+ 更强的骨干网络(more powerful backbone networks)
+ 将基于回归的人体检测和关键点检测以更灵活的方式结合(combine regression-based human detection and pose recognition in a more flexible manner)



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

two_stage/lib/core/function.py

```python
for i, (input, target, target_weight, meta) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    # compute output
    outputs = model(input)
```

### 3.1 Input

two_stage/lib/dataset/JointsDataset.py

```python
##使用坐标，而不是高斯热图
else:
	target = np.array(joints[:, 0:2] / self.image_size, dtype=np.float32)
```

疑问：如果使用高斯热图，效果如何呢？与TransPose相比？



### 3.2 Process

#### Build model

two_stage/lib/models/pose_transformer.py

```python
def get_pose_net(cfg, is_train, **kwargs):
    extra = cfg.MODEL.EXTRA

    transformer = build_transformer(hidden_dim=extra.HIDDEN_DIM, dropout=extra.DROPOUT, nheads=extra.NHEADS, dim_feedforward=extra.DIM_FEEDFORWARD,
                                    enc_layers=extra.ENC_LAYERS, dec_layers=extra.DEC_LAYERS, pre_norm=extra.PRE_NORM)
    pretrained = is_train and cfg.MODEL.INIT_WEIGHTS
    backbone = build_backbone(cfg, pretrained)
    model = PoseTransformer(cfg, backbone, transformer, **kwargs)

    return model
```

two_stage/lib/models/pose_transformer.py

```python
class PoseTransformer(nn.Module):

    def __init__(self, cfg, backbone, transformer, **kwargs):
        super(PoseTransformer, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.num_queries = extra.NUM_QUERIES
        self.num_classes = cfg.MODEL.NUM_JOINTS
        self.transformer = transformer
        self.backbone = backbone
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels[0], hidden_dim, 1)

        self.aux_loss = extra.AUX_LOSS

    def forward(self, x):
        ##Backbone
        src, pos = self.backbone(x)
        ##Transformer Encoder-Decoder
        hs = self.transformer(self.input_proj(src[-1]), None,
                              self.query_embed.weight, pos[-1])[0]

        ##Keypoint Classifier
        outputs_class = self.class_embed(hs)
        ##Keypoint Coordinate Regressor
        outputs_coord = self.kpt_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1],
               'pred_coords': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)
        return out
```

#### Backbone

two_stage/lib/models/backbone.py

```python
def build_backbone(cfg, pretrained):
    extra = cfg.MODEL.EXTRA
    num_layers = extra.NUM_LAYERS
    if type(num_layers) == str:
        name = num_layers
    else:
        name = f'resnet{num_layers}'
    position_embedding = build_position_encoding(
        extra.HIDDEN_DIM, extra.POS_EMBED_METHOD)
    return_interm_layers = hasattr(
        extra, 'NUM_FEATURE_LEVELS') and extra.NUM_FEATURE_LEVELS > 1
    if name.startswith('resnet'):
        backbone = ResNetBackbone(
            name, train_backbone=True, return_interm_layers=return_interm_layers, pretrained=pretrained, dilation=extra.DILATION)
    elif name == 'hrnet':
        backbone = HRNetBackbone(
            cfg, pretrained=pretrained, return_interm_layers=return_interm_layers)
    else:
        raise NotImplementedError(f'Unsupported backbone type: {name}')
    model = Joiner(backbone, position_embedding)
    return model
```

#### Transformer

two_stage/lib/models/transformer.py

```python
def build_transformer(**kwargs):
    return Transformer(
        d_model=kwargs['hidden_dim'],
        dropout=kwargs['dropout'],
        nhead=kwargs['nheads'],
        dim_feedforward=kwargs['dim_feedforward'],
        num_encoder_layers=kwargs['enc_layers'],
        num_decoder_layers=kwargs['dec_layers'],
        normalize_before=kwargs['pre_norm'],
        return_intermediate_dec=True,
    )
```

two_stage/lib/models/transformer.py

```python
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        ##collapse the spatial dimensions of z0 into one dimension(DETR)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
```

##### Transformer Encoder

two_stage/lib/models/transformer.py

```python
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
```

two_stage/lib/models/transformer.py

```python
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
```

##### Transformer Decoder

two_stage/lib/models/transformer.py

```python
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)
```

two_stage/lib/models/transformer.py

```python
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
```



### 3.3 Output

two_stage/lib/core/function.py

```python
preds, maxvals, pred = get_final_preds_match(config, output, c, s)
```

two_stage/lib/core/inference.py

```python
def get_final_preds_match(config, outputs, center, scale, flip_pairs=None):
    pred_logits = outputs['pred_logits'].detach().cpu()
    pred_coords = outputs['pred_coords'].detach().cpu()

    num_joints = pred_logits.shape[-1] - 1

    if config.TEST.INCLUDE_BG_LOGIT:
        prob = F.softmax(pred_logits, dim=-1)[..., :-1]
    else:
        prob = F.softmax(pred_logits[..., :-1], dim=-1)

    score_holder = []
    coord_holder = []
    orig_coord = []
    for b, C in enumerate(prob):
        _, query_ind = linear_sum_assignment(-C.transpose(0, 1)) # Cost Matrix: [17, N]
        score = prob[b, query_ind, list(np.arange(num_joints))][..., None].numpy()
        pred_raw = pred_coords[b, query_ind].numpy()
        if flip_pairs is not None:
            pred_raw, score = fliplr_joints(pred_raw, score, 1, flip_pairs, pixel_align=False, is_vis_logit=True)
        # scale to the whole patch
        pred_raw *= np.array(config.MODEL.IMAGE_SIZE)
        # transform back w.r.t. the entire img
        pred = transform_preds(pred_raw, center[b], scale[b], config.MODEL.IMAGE_SIZE)
        orig_coord.append(pred_raw)
        score_holder.append(score)
        coord_holder.append(pred)
    
    matched_score = np.stack(score_holder)
    matched_coord = np.stack(coord_holder)

    return matched_coord, matched_score, np.stack(orig_coord)
```



## 参考资料

1. [Pose Recognition with Cascade Transformers 论文笔记](https://www.yuque.com/jinluzhang/researchblog/prtr)
2. [【CVPR 2021】PRTR：基于transformer的2D Human Pose Estimation](https://zhuanlan.zhihu.com/p/368067142)



更新于2021-05-15

