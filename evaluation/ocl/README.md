# Evaluation

We support evaluation on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [MOVi](https://github.com/google-research/kubric/tree/main/challenges/movi#multi-object-video-movi-datasets) (MOVi-C and MOVi-E) and [EntitySeg](https://github.com/adobe-research/EntitySeg-Dataset). Download and extract the datasets.

Run the `eval_ocl.py` script from the root of this repository via following commands.

## Pascal VOC
```
python eval_ocl.py \
--arch vit_small \
--patch_size 16 \
--checkpoint_path path_to_checkpoint \
--out_dim 8192 \
--patch_out_dim 8192 \
--dataset voc \
--data_path $TMPDIR/datasets/VOC12/VOCdevkit/VOC2012/ \
--eval_batch_size 4 \
--num_slots 7 \
--interpolation_mode bilinear \
```

## MOVi-C

```
python eval_ocl.py \
--arch vit_small \
--patch_size 16 \
--checkpoint_path path_to_checkpoint \
--out_dim 8192 \
--patch_out_dim 8192 \
--dataset movi \
--data_path $TMPDIR/datasets/MOVi/c \
--eval_batch_size 4 \
--num_slots 11 \
--val_image_size 224 \
--val_mask_size 128 \
--interpolation_mode bilinear \
```

## MOVi-E

```
python eval_ocl.py \
--arch vit_small \
--patch_size 16 \
--checkpoint_path path_to_checkpoint \
--out_dim 8192 \
--patch_out_dim 8192 \
--dataset movi \
--data_path $TMPDIR/datasets/MOVi/e \
--eval_batch_size 4 \
--num_slots 24 \
--val_image_size 224 \
--val_mask_size 128 \
--interpolation_mode bilinear \
```

# EntitySeg

```
python eval_ocl.py \
--arch vit_small \
--patch_size 16 \
--checkpoint_path path_to_checkpoint \
--out_dim 8192 \
--patch_out_dim 8192 \
--dataset entityseg \
--data_path $TMPDIR/datasets/val_set \
--eval_batch_size 4 \
--num_slots 7 \
--interpolation_mode bilinear \
```