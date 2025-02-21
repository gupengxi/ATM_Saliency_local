from omegaconf import OmegaConf
from pytorch_grad_cam import GradCAM, FullGrad, EigenCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, GradCAMElementWise, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
# import vc_models
# from vc_models.models.vit import model_utils
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from einops import rearrange, repeat
from atm.utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video
from atm.model.track_patch_embed import TrackPatchEmbed
from atm.dataloader import BCDataset, get_dataloader
from tqdm import tqdm
from atm.model.track_transformer import TrackTransformer
from collections import deque

track_cfg = OmegaConf.load(f"results/track_transformer/libero_track_transformer_libero-spatial/config.yaml")
track_cfg.model_cfg.load_path = "results/track_transformer/libero_track_transformer_libero-spatial/model_best.ckpt"
track_cls = eval(track_cfg.model_name)
track_model = track_cls(**track_cfg.model_cfg)
track_model.eval()


def _get_view_one_hot(tr):
    """ tr: b, v, t, tl, n, d -> (b, v, t), tl n, d + v"""
    b, v, t, tl, n, d = tr.shape
    tr = rearrange(tr, "b v t tl n d -> (b t tl n) v d")
    one_hot = torch.eye(v, device=tr.device, dtype=tr.dtype)[None, :, :].repeat(tr.shape[0], 1, 1)
    tr_view = torch.cat([tr, one_hot], dim=-1)  # (b t tl n) v (d + v)
    tr_view = rearrange(tr_view, "(b t tl n) v c -> b v t tl n c", b=b, v=v, t=t, tl=tl, n=n, c=d + v)
    return tr_view

def track_encode(track_model, track_obs, task_emb):
    """
    Args:
        track_obs: b v t tt_fs c h w
        task_emb: b e
    Returns: b v t track_len n 2
    """
    num_track_ids = 32
    b, v, t, *_ = track_obs.shape

    track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")

    grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
    grid_sampled_track = repeat(grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=16)
    grid_sampled_track = rearrange(grid_sampled_track, "b v t tl n d -> (b v t) tl n d")

    expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
    expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
    with torch.no_grad():
        pred_tr, _ = track_model.reconstruct(track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
        recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

    recon_tr = recon_tr[:, :, :, :16, :, :]  # truncate the track to a shorter one
    _recon_tr = recon_tr.clone()  # b v t tl n 2
    with torch.no_grad():
        tr_view = _get_view_one_hot(recon_tr)  # b v t tl n c

    tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
    track_proj_encoder = TrackPatchEmbed(
            num_track_ts=16,
            num_track_ids=32,
            patch_size=4,
            in_dim=2 + 2,  # X, Y, one-hot view embedding
            embed_dim=128)
    tr = track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
    tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=num_track_ids)  # (b t n) (v patch_num) d

    return tr, _recon_tr


train_dataset = BCDataset(dataset_dir="data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo",   
                          img_size=[128,128],
                          frame_stack=10,
                          num_track_ts=16,
                          num_track_ids=32,
                          track_obs_fs=1,
                          augment_track=False,
                          extra_state_keys=["joint_states", "gripper_states"],
                          cache_all=True,
                          cache_image=True,
                          aug_prob=0.9)
train_loader = get_dataloader(train_dataset,
                                    mode="train",
                                    num_workers=1,
                                    batch_size=4)

for obs, track_obs, track, task_emb, action, extra_states in tqdm(train_loader):
    track_obs_queue = deque(maxlen=10)

    # obs, track_obs, track, task_emb, action = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
    # extra_states = {k: v.bfloat16() for k, v in extra_states.items()}

    # while len(track_obs_queue) < 10:
    #     track_obs_queue.append(torch.zeros_like(obs))
    track_obs_queue.append(obs.clone())
    track_obs = torch.cat(list(track_obs_queue), dim=2)  # b v fs c h w
    track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")
    print(f"track_obs shape is {track_obs.shape}")

    track_encoded, recon_track = track_encode(track_model, track_obs, task_emb)
    print(f"shape track_encoded is {track_encoded.shape}")
    print(f"shape recon is {recon_track.shape}")
    