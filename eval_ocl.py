import os.path
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils

from models.ocebo import OCEBO

from evaluation.ocl.datasets import PascalVOC, COCO2017, MOVi, EntitySeg
from evaluation.ocl.ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_image_size', type=int, default=224)
    parser.add_argument('--val_mask_size', type=int, default=320)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--viz_resolution_factor', type=float, default=0.5)

    parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
    parser.add_argument('--dataset', default='coco', help='coco or voc')
    parser.add_argument('--data_path',  type=str, help='dataset path')

    parser.add_argument('--arch', default='vit_small', type=str,
            choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                    'swin_tiny','swin_small', 'swin_base', 'swin_large'],
            help="""Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--num_slots', default=7, type=int, help="Number of objects")
    parser.add_argument('--slot_dim', default=256, type=int, help="Slot dimension")
    parser.add_argument('--slot_enc_mlp_dim', default=1024, type=int, help="Slot mlp dimension in encoder")
    parser.add_argument('--slot_dec_mlp_dim', default=2048, type=int, help="Slot mlp dimension in decoder")
    parser.add_argument('--target_image_size', default=224, type=int, help="Target image size")
    parser.add_argument('--interpolation_mode', default='bilinear', type=str, help="""Interpolation mode.""")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    log_path = os.path.join(os.path.dirname(args.checkpoint_path), "eval_ocl_output.txt")

    if args.dataset == 'voc':
        val_dataset = PascalVOC(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'coco':
        val_dataset = COCO2017(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'movi':
        val_dataset = MOVi(root=os.path.join(args.data_path, 'validation'), split='validation', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'entityseg':
        val_dataset = EntitySeg(root=args.data_path, image_size=args.val_image_size, mask_size=args.val_mask_size)

    val_sampler = None

    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True,
    }

    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last=False, batch_size=args.eval_batch_size, **loader_kwargs)

    val_epoch_size = len(val_loader)

    model = OCEBO(args)
    # move networks to gpu and load the checkpoint
    model = model.cuda()
    chkp = torch.load(args.checkpoint_path)
    from models.vision_transformer import interpolate_pos_embed
    interpolate_pos_embed(model, chkp['student'])
    chkp_dec_pos_emb_size = int(chkp['student']['module.slot_model.slot_decoder.pos_embed'].size(1) ** 0.5)
    new_dec_pos_emb_size = int(model.slot_model.slot_decoder.pos_embed.size(1) ** 0.5)
    if chkp_dec_pos_emb_size != new_dec_pos_emb_size:
        chkp_dec = chkp['student']['module.slot_model.slot_decoder.pos_embed']
        chkp_dec = chkp_dec.reshape(1, chkp_dec_pos_emb_size, chkp_dec_pos_emb_size, -1)
        new_chkp_dec = F.interpolate(chkp_dec.permute(0, 3, 1, 2), size=(new_dec_pos_emb_size, new_dec_pos_emb_size), mode='bilinear')
        chkp['student']['module.slot_model.slot_decoder.pos_embed'] = new_chkp_dec.flatten(2).transpose(1, 2)
    msg = model.load_state_dict({k.replace('module.', ''): v for k, v in chkp['student'].items()})

    print(f"Loaded model with message {msg}")

    MBO_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
    MBO_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
    miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).cuda()
    ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).cuda()

    with torch.no_grad():
        model.eval()
        counter = 0

        for batch, (image, true_mask_i, true_mask_c, mask_ignore) in enumerate(tqdm(val_loader)):
            image = image.cuda()
            true_mask_i = true_mask_i.cuda()
            true_mask_c = true_mask_c.cuda()
            mask_ignore = mask_ignore.cuda()

            batch_size = image.shape[0]
            counter += batch_size

            img_feats = model.backbone(image)
            torch.rand(img_feats.size(0), img_feats.size(1), device=img_feats.device)
            _, _, masks = model.slot_model(img_feats)

            bs, s, hw, _ = masks.size()
            dec_slots_attns = masks.squeeze(-1).reshape(bs, s, int(hw ** 0.5), int(hw ** 0.5))
            if args.interpolation_mode == 'bilinear':
                dec_attns = F.interpolate(dec_slots_attns, size=args.val_mask_size, mode='bilinear')
                dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]
                pred_dec_mask = dec_attns.argmax(1).squeeze(1)
            else:
                pred_dec_mask = dec_slots_attns.argmax(1).unsqueeze(1)
                pred_dec_mask = F.interpolate(pred_dec_mask.float(), size=args.val_mask_size, mode='nearest').long().squeeze(1)
                
            # Compute ARI, MBO_i and MBO_c, miou scores for both slot attention and decoder
            true_mask_i_reshaped = torch.nn.functional.one_hot(true_mask_i).to(torch.float32).permute(0,3,1,2).cuda()
            true_mask_c_reshaped = torch.nn.functional.one_hot(true_mask_c).to(torch.float32).permute(0,3,1,2).cuda()
            pred_dec_mask_reshaped = torch.nn.functional.one_hot(pred_dec_mask).to(torch.float32).permute(0,3,1,2).cuda()

            MBO_i_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
            MBO_c_metric.update(pred_dec_mask_reshaped, true_mask_c_reshaped, mask_ignore)
            miou_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
            ari_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)

        ari = 100 * ari_metric.compute()
        miou = 100 * miou_metric.compute()
        mbo_i = 100 * MBO_i_metric.compute()
        mbo_c = 100 * MBO_c_metric.compute()

        print(f"Dataset: {args.dataset}")
        print(f"FG-ARI: {ari:.3f}")
        print(f"mIOU: {miou:.3f}")
        print(f"MBO_i: {mbo_i:.3f}")
        print(f"MBO_c: {mbo_c:.3f}")
