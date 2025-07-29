# UMMSS
This repository is the official implementation of UMMSS

![UMMSS](pic/main.png)

## 📋Requirements
The yml file we provided can be used to install requirements:
```setup
conda env create -f environment.yml
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```


## 📋Training
First, set up the basic environment:
```basic
cd path/to/RobustSeg
conda activate RobustSeg
export PYTHONPATH="path/to/UMMSS"
```

The training command for the full-modality teacher model is as follows:
```train teacher
python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train_teacher.py --cfg configs/deliver_seg_t.yaml
```
The training command for the robust teacher model is as follows (stage 1):
```train robust teacher
python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train_deliver_mad50_proto_cur_reg.py --cfg configs/deliver_seg_s.yaml
```

>  The second-stage training will be released after the paper is accepted. In the meantime, you can test the weights provided in the 'weights' folder.

## Evaluation

To evaluate my model on Deliver, run:

```eval
CUDA_VISIBLE_DEVICES=1 python tools/val_mm_mmkd.py --cfg configs/deliver_seg_s.yaml
```
or

```eval
CUDA_VISIBLE_DEVICES=1 python tools/val_mm_EMM.py --cfg configs/deliver_seg_s.yaml
CUDA_VISIBLE_DEVICES=1 python tools/val_mm_RMM.py --cfg configs/deliver_seg_s.yaml
CUDA_VISIBLE_DEVICES=1 python tools/val_mm_NM.py --cfg configs/deliver_seg_s.yaml
```

## Data Preparation
Used Datasets: 
[MUSES](https://muses.vision.ee.ethz.ch/) / [DELIVER](https://github.com/jamycheung/DELIVER) / [MCubeS](https://github.com/kyotovision-public/multimodal-material-segmentation)

## Pre-trained Weights of RobustSeg
You can directly use the pre-trained models in the weights folder, which represent the highest mIoU in the paper.

## References
We appreciate the previous open-source works: [DELIVER](https://github.com/jamycheung/DELIVER) / [SegFormer](https://github.com/NVlabs/SegFormer) / [MUSES](https://muses.vision.ee.ethz.ch/) / [AnySeg](https://github.com/zhengxuJosh/AnySeg)