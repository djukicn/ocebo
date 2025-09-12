''' Based on SLATE and BOQSA libraries:
https://github.com/singhgautam/slate/blob/master/slot_attn.py
https://github.com/YuLiu-LY/BO-QSA/blob/main/models/slot_attn.py
'''

from timm.models.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):

    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_iter,
        input_size,
        slot_size, 
        mlp_size, 
        truncate,
        heads,
        epsilon=1e-8, 
        drop_path=0,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.input_size = input_size
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.truncate = truncate
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_size, slot_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        print(self.truncate)
        assert self.truncate in ['bi-level', 'fixed-point', 'none']


    def forward(self, inputs, slots_init):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].
        slots = slots_init
        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k
        
        # Multiple rounds of attention.
        for i in range(self.num_iter):
            if i == self.num_iter  - 1:
                if self.truncate == 'bi-level':
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == 'fixed-point':
                    slots = slots.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))                          # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)                # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_vis = attn.sum(1)                                                      # Shape: [batch_size, num_inputs, num_slots].
            
            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)                           # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)                       # Shape: [batch_size, num_slots, slot_size].
            
            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, N_q, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis, attn_logits

class SlotAttentionEncoder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_channels, slot_size, mlp_hidden_size, pos_channels, truncate='bi-level', init_method='embedding', num_heads = 1, drop_path = 0.0):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels
        self.init_method = init_method

        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        
        assert init_method in ['shared_gaussian', 'embedding', 'cls_based']
        if init_method == 'shared_gaussian':
            # Parameters for Gaussian init (shared by all slots).
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif init_method == 'embedding':
            self.slots_init = nn.Embedding(num_slots, slot_size)
            nn.init.xavier_uniform_(self.slots_init.weight)
        elif init_method == 'cls_based':
            self.patch_to_slot = nn.Linear(input_channels, slot_size)
        else:
            raise NotImplementedError
        
        self.slot_attention = SlotAttention(
            num_iterations,
            input_channels, slot_size, mlp_hidden_size, truncate, num_heads, drop_path=drop_path)
    
    def forward(self, x):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        # Slot Attention module.
        B, *_ = x.size()
        dtype = x.dtype
        device = x.device
        init_slots = self.slots_initialization(B, dtype, device, x)

        x = x[:, 1:]
        x = self.mlp(self.layer_norm(x))
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].
        slots, attn, attn_logits = self.slot_attention(x, init_slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
        
        return slots, attn, init_slots, attn_logits
    
    def slots_initialization(self, B, dtype, device, x=None):
        # The first frame, initialize slots.
        if self.init_method == 'shared_gaussian':
            slots_init = torch.empty((B, self.num_slots, self.slot_size), dtype=dtype, device=device).normal_()
            slots_init = self.slot_mu + torch.exp(self.slot_log_sigma) * slots_init
        elif self.init_method == 'embedding':
            slots_init = self.slots_init.weight.expand(B, -1, -1).contiguous()
        elif self.init_method == 'cls_based':
            with torch.no_grad():
                cls_token = F.normalize(x[:, [0], :], dim=-1)
                patch_tokens = F.normalize(x[:, 1:, :], dim=-1)
                sims = torch.einsum('b n d, b c d -> b n c', patch_tokens, cls_token).squeeze(-1)

                idx = sims.argsort(dim=1).chunk(self.num_slots, dim=1)
                min_chunk_size = min([i.size(1) for i in idx])
                choice = torch.randint(0, min_chunk_size, size=(self.num_slots, ))
                idx = torch.stack([i[:, c] for i, c in zip(idx, choice.tolist())], dim=1)
                
                slots_init = torch.gather(patch_tokens, dim=-1, index=idx.unsqueeze(-1).repeat(1, 1, patch_tokens.size(-1)))
            
            slots_init = self.patch_to_slot(slots_init)
        
        return slots_init


class MlpDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        hidden_features: Dimension of hidden layers.
    """

    def __init__(self, object_dim, output_dim, num_patches, hidden_features = 2048):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, object_dim) * 0.02)
        self.decoder = build_mlp(object_dim, output_dim + 1, hidden_features)

    def forward(self, encoder_output):

        initial_shape = encoder_output.shape[:-1]
        encoder_output = encoder_output.flatten(0, -2)

        encoder_output = encoder_output.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = encoder_output + self.pos_embed

        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)
        
        return reconstruction, masks
    
    
def build_mlp(input_dim = int, output_dim = int, hidden_features = 2048, n_hidden_layers = 3):
    
    layers = []
    current_dim = input_dim
    features = [hidden_features]*n_hidden_layers

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU(inplace=True))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)

    return nn.Sequential(*layers)


class SlotAttentionModel(nn.Module):

    def __init__(self, args):

        super(SlotAttentionModel, self).__init__()

        self.slot_encoder = SlotAttentionEncoder(3, args.num_slots, args.embed_dim, args.slot_dim, args.slot_enc_mlp_dim, 4, truncate='none', init_method='shared_gaussian')

        self.slot_decoder = MlpDecoder(args.slot_dim, args.embed_dim, (args.target_image_size // args.patch_size) ** 2, args.slot_dec_mlp_dim)

    def forward(self, x, *args, **kwargs):

        slots, _, _, _ = self.slot_encoder(x)

        recon, masks = self.slot_decoder(slots)

        # add back the cls
        recon = torch.cat((x[:, [0]], recon), dim=1)

        return recon, slots, masks.unsqueeze(-1)