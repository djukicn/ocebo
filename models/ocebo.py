import torch
from torch import nn
from torchvision import models as torchvision_models
from .head import OCEBOHead
from .slot_attention import SlotAttentionModel
from .vision_transformer import vit_tiny, vit_small, vit_base, vit_large

MODEL_DICT = {
    'vit_tiny': vit_tiny,
    'vit_small': vit_small,
    'vit_base': vit_base,
    'vit_large': vit_large
}

def build_backbone(args):
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    if args.arch in MODEL_DICT.keys():
        backbone = MODEL_DICT[args.arch](
            img_size=args.target_image_size,
            patch_size=args.patch_size,
        )
        args.embed_dim = backbone.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        backbone = torchvision_models.__dict__[args.arch]()
        args.embed_dim = backbone.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")
    
    return backbone


class OCEBO(nn.Module):

    def __init__(self, args):
        super(OCEBO, self).__init__()

        backbone = build_backbone(args)
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone

        self.slot_model = SlotAttentionModel(args) 
    
        self.head = OCEBOHead(
            args.embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        )

    def forward(self, x, teacher=False, masks_only=False):
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        img_feats = self.backbone(x)
        sa_out, _, masks = self.slot_model(img_feats)
        if masks_only:
            return masks
        
        # Output[1]: for student, return features after reconstruction but before head, for teacher return backbone features
        if teacher:
            output = (self.head(img_feats), img_feats[:, 1:])
        else:
            output = (self.head(sa_out), sa_out[:, 1:])

        return output