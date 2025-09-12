# Object-Centric Pretraining via Target Encoder Bootstrapping (ICLR 2025)

The official PyTorch implementation of the [ICLR 2025 paper OCEBO](https://openreview.net/pdf?id=7d2JwGbxhA).

## Dependencies

To run the training code, install the following libraries: `PyTorch 2.1.2`, `numpy 1.26.4` and `timm`. In addition, the evaluation code requires `pycocotools`, `torchmetrics` and `scipy`.

## Training

The code supports training on COCO and COCO+ datasets. First, download `train2017` from [COCO](https://cocodataset.org) and annotations `instances_train2017.json`. To train on COCO+, additionally download `unlabeled2017` and `image_info_unlabeled2017.json`, ensuring that both image directories are in the same directory and that both annotation files are in the same directory. In the training script, provide paths to `train2017` and `instances_train2017.json` via `--data_path` and `--ann_path` arguments, respectively.

Run the training script with the arguments provided below. The model was trained on 8 GPUs with the batch size of 256 (32 per GPU).

```
python main_ocebo.py \
--arch vit_small \
--patch_size 16 \
--out_dim 8192 \
--patch_out_dim 8192 \
--shared_head True \
--norm_last_layer True \
--num_slots 7 \
--slot_dim 256 \
--slot_enc_mlp_dim 1024 \
--slot_dec_mlp_dim 2048 \
--target_image_size 224 \
--momentum_teacher 0.996 \
--lambda1 1 \
--lambda2 1 \
--warmup_teacher_temp 0.04 \
--teacher_temp 0.07 \
--warmup_teacher_patch_temp 0.04 \
--teacher_patch_temp 0.07 \
--warmup_teacher_temp_epochs 30 \
--use_fp16 False \
--weight_decay 0.04 \
--weight_decay_end 0.4 \
--clip_grad 0.1 \
--batch_size_per_gpu 32 \
--epochs 400 \
--mask_sharpening_epochs 100 \
--freeze_last_layer 1 \
--lr 0.0003 \
--warmup_epochs 10 \
--min_lr 1e-06 \
--optimizer adamw \
--crops_min_scale 0.25 \
--dataset coco \
--data_path $TMPDIR/datasets/coco/images/train2017 \
--ann_path $TMPDIR/datasets/coco/annotations/instances_train2017.json \
--output_dir output_dir_path_here \
--saveckp_freq 30 \
--seed 0 \
--num_workers 7
```

## Pretrained models

The checkpoints for OCEBO pretrained on COCO and COCO+ can be found below.

[COCO](https://rdr.kuleuven.be/api/access/datafile/272702)

[COCO+](https://rdr.kuleuven.be/api/access/datafile/272703)

## Evaluation

See README in the `evaluation/ocl` directory.

## Acknowledgements

The elements of this repository were built upon [iBOT](https://github.com/bytedance/ibot), [SlotCon](https://github.com/CVMI-Lab/SlotCon), [SPOT](https://github.com/CVMI-Lab/SlotCon), [MAE](https://github.com/facebookresearch/mae).

## Citation

```
@InProceedings{Dukic_2025_ICLR,
    author    = {{\DJ}uki\'c, Nikola and Lebailly, Tim and Tuytelaars, Tinne},
    title     = {Object-Centric Pretraining via Target Encoder Bootstrapping},
    booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
    year      = {2025},
}
```

