import os

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import lightning
from lightning.fabric import Fabric

from atm.model import *
from atm.dataloader import SaliencyDataset, get_dataloader
from atm.utils.log_utils import BestAvgLoss, MetricLogger
from atm.utils.train_utils import init_wandb, setup_lr_scheduler, setup_optimizer

from atm.dataloader import BCDataset, get_dataloader
from tqdm import tqdm
from atm.utils.flow_utils import ImageUnNormalize, tracks_to_video, tracks_to_binary_img
import imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import argparse
from tqdm import tqdm

from models.ooal import Net as ooal_model

from ooal_utils.utils.viz import viz_pred_test
from ooal_utils.utils.util import set_seed, process_gt, normalize_map
from ooal_utils.utils.evaluation import cal_kl, cal_sim, cal_nss
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, FullGrad, EigenCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, GradCAMElementWise, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ooal_data.data.agd20k_ego import TestData, SEEN_AFF, UNSEEN_AFF
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from atm.utils.flow_utils import sample_tracks_nearest_to_grids,sample_tracks, sample_tracks_object,sample_tracks_object_1
import grounding_dino.groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir, compose

import os
import torch
import supervision as sv
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
if GlobalHydra().is_initialized():
    GlobalHydra().clear()
@hydra.main(config_path="../conf/train_saliencymap_transformer", version_base="1.3")
def main(cfg: DictConfig):
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    additional_config_path = "/home/pengxi/Documents/ATM/"  # Adjust with your actual path
    initialize_config_dir(config_dir=additional_config_path)
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), precision="bf16-mixed" if cfg.mix_precision else None, strategy="deepspeed")
    fabric.launch()

    None if (cfg.dry or not fabric.is_global_zero) else init_wandb(cfg)
    print(UNSEEN_AFF)

    train_dataset = SaliencyDataset(dataset_dir=cfg.train_dataset, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset, mode="train", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    train_vis_dataset = SaliencyDataset(dataset_dir=cfg.train_dataset, vis=True, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_vis_dataloader = get_dataloader(train_vis_dataset, mode="train", num_workers=1, batch_size=1)

    val_dataset = SaliencyDataset(dataset_dir=cfg.val_dataset, **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size * 2)

    val_vis_dataset = SaliencyDataset(dataset_dir=cfg.val_dataset, vis=True, **cfg.dataset_cfg, aug_prob=0.)
    val_vis_dataloader = get_dataloader(val_vis_dataset, mode="val", num_workers=1, batch_size=1)

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)


    # 这是什么？应该要改
    lbd_track = cfg.lbd_track
    lbd_img = cfg.lbd_img
    p_img = cfg.p_img

    # Pick ckpt based on  the average of the last 5 epochs
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

    OUTPUT_DIR = Path("outputs/check_box")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device="cuda")
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    # parser = argparse.ArgumentParser()
    ##  path
    # parser.add_argument('--data_root', type=str, default='./dataset/')
    # parser.add_argument('--model_file', type=str, default="./ooal-models/unseen_best")
    # parser.add_argument('--save_path', type=str, default='./save_preds')
    ##  image
    # parser.add_argument('--divide', type=str, default='Seen')
    # parser.add_argument('--crop_size', type=int, default=128)
    # parser.add_argument('--resize_size', type=int, default=630)
    #### test
    # parser.add_argument("--test_batch_size", type=int, default=1)
    # parser.add_argument('--test_num_workers', type=int, default=8)
    # parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--viz', action='store_true', default=False)
    # args = parser.parse_args()
    # args.mask_root = os.path.join('./dataset/', 'Seen', "testset", "GT")
    print(UNSEEN_AFF)
    class_names_ooal = UNSEEN_AFF
    from models.ooal import Net as ooal_model
    ooal_model = ooal_model(class_names_ooal, 768, 512).cuda()
    ooal_model.eval()
    state_dict = torch.load("./ooal-models/unseen_best")['model_state_dict']
    ooal_model.load_state_dict(state_dict, strict=False)
    grounding_model = load_model(
        model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        model_checkpoint_path="gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
        device="cuda"
    )

    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            scheduler=scheduler,
            mix_precision=cfg.mix_precision,
            clip_grad=cfg.clip_grad,
            sam2_predictor=sam2_predictor,
            grounding_model=grounding_model,
            ooal_model=ooal_model,
            class_names_ooal=class_names_ooal
        )

        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

            if epoch % cfg.val_freq == 0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    lbd_track=lbd_track,
                    lbd_img=lbd_img,
                    p_img=p_img,
                    mix_precision=cfg.mix_precision,
                    tag="val",
                )

                # Save best checkpoint
                metric_logger.update(**val_metrics)

                val_metrics = {**val_metrics}
                loss_metric = val_metrics["val/loss"]
                is_best = best_loss_logger.update_best(loss_metric, epoch)

                if is_best:
                    model.save(f"{work_dir}/model_best.ckpt")
                    with open(f"{work_dir}/best_epoch.txt", "w") as f:
                        f.write(
                            "Best epoch: %d, Best %s: %.4f"
                            % (epoch, "loss", best_loss_logger.best_loss)
                        )
                None if cfg.dry else wandb.log(val_metrics, step=epoch)

            if epoch % cfg.save_freq == 0:
                model.save(f"{work_dir}/model_{epoch}.ckpt")

                def vis_and_log(model, vis_dataloader, mode="train"):
                    vis_dict = visualize(model, vis_dataloader, mix_precision=cfg.mix_precision)

                    caption = f"reconstruction (right) @ epoch {epoch}; \n Track MSE: {vis_dict['track_loss']:.4f}"
                    wandb_vis_track = wandb.Video(vis_dict["combined_track_vid"], fps=10, format="mp4", caption=caption)
                    None if cfg.dry else wandb.log({f"{mode}/reconstruct_track": wandb_vis_track}, step=epoch)

                vis_and_log(model, train_vis_dataloader, mode="train")
                vis_and_log(model, val_vis_dataloader, mode="val")

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(fabric,
                  model,
                  dataloader,
                  optimizer,
                  lbd_track,
                  lbd_img,
                  p_img,
                  mix_precision=False,
                  scheduler=None,
                  clip_grad=1.0,
                  sam2_predictor=None,
                  grounding_model=None,
                  ooal_model=None,
                  class_names_ooal=None):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    saliency_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0

    model.train()
    i = 0
    # 修改读取的数据,应该可以更简化一点，加上saliency map的生成
    # for vid, track, vis, task_emb in tqdm(dataloader):
    print(f"??????????????????????????????????{class_names_ooal}")
    for obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori in tqdm(dataloader):
        obs,track_ori ,vi_ori = obs.cuda(),track_ori.cuda() ,vi_ori.cuda()
        if mix_precision:
            # vid, track, vis, task_emb = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16()
            obs, track_obs, track, task_emb, action, track_ori ,vi_ori = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16(), track_ori.bfloat16() ,vi_ori.bfloat16()

        

        # TODO:生成saliency map,注意这个函数根据任务不同也不一样
        saliency_obs = generate_saliency_obs(obs,track_ori ,vi_ori,sam2_predictor=sam2_predictor,grounding_model=grounding_model,ooal_model=ooal_model,class_names_ooal=class_names_ooal)

        # 修改训练模型输入
        b, v, t, c, h, w = obs.shape
    

        obs_reshape = obs.view(b*v, t, c, h, w)
        saliency_obs = saliency_obs.view(b*v, t, c, h, w)
        # reshape一下
        b, emb_size = task_emb.shape
        task_emb_reshape = task_emb.unsqueeze(1).expand(b, v, emb_size).reshape(b * v, emb_size)
    
        # b, v, tl, n, _ = track_ori.shape
        
        # track_ori = track_ori.view(b*v, tl, n, _)
        # b, v, tl, n = vi_ori.shape
        # vi_ori = vi_ori.view(b*v, tl, n)

        loss, ret_dict = model.forward_loss(
            obs_reshape,#b*v,t,c,h,w
            saliency_obs,#b*v,t,c,h,w
            task_emb_reshape,#b*v, emb_size
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img)  # do not use vis
        optimizer.zero_grad()
        fabric.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        saliency_loss += ret_dict["saliency_loss"]
        vid_loss += ret_dict["img_loss"]
        tot_loss += ret_dict["loss"]
        tot_items += b

        i += 1

    out_dict = {
        "train/saliency_loss": saliency_loss / tot_items,
        "train/vid_loss": vid_loss / tot_items,
        "train/loss": tot_loss / tot_items,
    }

    if scheduler is not None:
        scheduler.step()

    return out_dict

@torch.no_grad()
def evaluate(model, dataloader, lbd_track, lbd_img, p_img, mix_precision=False, tag="val"):
    saliency_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0
    model.eval()

    i = 0
    for obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori in tqdm(dataloader):
        obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori = obs.cuda(), track_obs.cuda(), track.cuda(), task_emb.cuda(), action.cuda(), extra_states.cuda(), track_ori.cuda() ,vi_ori.cuda()
        if mix_precision:
            # vid, track, vis, task_emb = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16()
            obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16(), extra_states.bfloat16(), track_ori.bfloat16() ,vi_ori.bfloat16()
    # for vid, track, vis, task_emb in tqdm(dataloader):
    #     vid, track, vis, task_emb = vid.cuda(), track.cuda(), vis.cuda(), task_emb.cuda()
    #     if mix_precision:
    #         vid, track, vis, task_emb = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16()
        b, v, t, c, h, w = obs.shape
        obs_reshape = obs.view(b*v, t, c, h, w)
        b, emb_size = task_emb.shape
        task_emb_reshape = task_emb.unsqueeze(1).expand(b, v, emb_size).reshape(b * v, c)
        # b, tl, n, _ = track.shape
        saliency_obs, saliency_obs_vis = generate_saliency_obs(obs_reshape)

        _, ret_dict = model.forward_loss(
            obs_reshape,
            saliency_obs,
            task_emb_reshape,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            vis=None)

        saliency_loss += ret_dict["saliency_loss"]
        vid_loss += ret_dict["img_loss"]
        tot_loss += ret_dict["loss"]
        tot_items += b

        i += 1

    out_dict = {
        f"{tag}/track_loss": saliency_loss / tot_items,
        f"{tag}/vid_loss": vid_loss / tot_items,
        f"{tag}/loss": tot_loss / tot_items,
    }

    return out_dict

@torch.no_grad()
def visualize(model, dataloader, mix_precision=False):
    model.eval()
    keep_eval_dict = None
    for i, (obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori) in enumerate(dataloader):
        obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori = obs.cuda(), track_obs.cuda(), track.cuda(), task_emb.cuda(), action.cuda(), extra_states.cuda(), track_ori.cuda() ,vi_ori.cuda()
        if mix_precision:
            # vid, track, vis, task_emb = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16()
            obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16(), extra_states.bfloat16(), track_ori.bfloat16() ,vi_ori.bfloat16()
    # for vid, track, vis, task_emb in tqdm(dataloader):
    #     vid, track, vis, task_emb = vid.cuda(), track.cuda(), vis.cuda(), task_emb.cuda()
    #     if mix_precision:
    #         vid, track, vis, task_emb = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16()
        # b, tl, n, _ = track.shape
        b,v,t,c,h,w = obs.shape
        obs_reshape = obs.view(b*v, t, c, h, w)
        b, emb_size = task_emb.shape
        task_emb_reshape = task_emb.unsqueeze(1).expand(b, v, emb_size).reshape(b * v, c)
        saliency_obs, saliency_obs_vis = generate_saliency_obs(obs_reshape)
    # for i, (vid, track, vis, task_emb) in enumerate(dataloader):
    #     vid, track, task_emb = vid.cuda(), track.cuda(), task_emb.cuda()
    #     if mix_precision:
    #         vid, track, task_emb = vid.bfloat16(), track.bfloat16(), task_emb.bfloat16()
        _, eval_dict = model.forward_vis(obs_reshape, saliency_obs, task_emb_reshape, p_img=0)
        if keep_eval_dict is None or torch.rand(1) < 0.1:
            keep_eval_dict = eval_dict

        if i == 10:
            break
    return keep_eval_dict
def tracks_to_grayscale_maps(tracks, img_size):
    """
    Generate a saliency map that emphasizes the trajectory regions and de-emphasizes the differences between trajectories.
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    img_size: (H, W), the size of the output image
    return: Grayscale saliency map (B, H, W), where each pixel's intensity emphasizes trajectory regions
    """
    B, T, N, _ = tracks.shape

    # 通过 tracks_to_binary_img 将轨迹转换为二值图像
    binary_vid = tracks_to_binary_img(tracks, img_size=img_size).float()  # (B, T, 1, H, W)
    
    # 这里去除冗余通道，使 binary_vid 为三通道 (B, T, 3, H, W)
    binary_vid[:, :, 0] = binary_vid[:, :, 1]
    binary_vid[:, :, 2] = binary_vid[:, :, 1]

    print(f"binary_vid shape is {binary_vid.shape}")

    # Sum across all time steps (T) to emphasize trajectory regions
    saliency_map = binary_vid.sum(dim=1)  # (B, 3, H, W)

    # Normalize to [0, 1] for consistency
    saliency_map = saliency_map / torch.amax(saliency_map, dim=(1, 2, 3), keepdim=True)

    # Apply Gaussian smoothing
    gaussian_blur = transforms.GaussianBlur(kernel_size=33, sigma=5.0)
    saliency_map = torch.stack([gaussian_blur(frame) for frame in saliency_map])  # (B, 3, H, W)
    
    # Scale back to 0-255 for visualization (still retaining grayscale)
    saliency_map = (saliency_map * 255)
    print(f"shape of saliency_map is {saliency_map.shape}")
    # Convert to numpy for further processing (B, H, W)
    saliency_map_np = saliency_map.sum(dim=1).cpu().numpy()  # Shape: (B, H, W)

    print(f"shape of saliency_map_np is {saliency_map_np.shape}")
    
    # Convert each frame to grayscale (B, H, W)
    grayscale_saliency_maps = []
    for i in range(B):
        frame = saliency_map_np[i]
        
        # Ensure the frame is of type uint8 (8-bit unsigned integer)
        frame = np.uint8(frame)
        
        # No color map applied, keeping it grayscale
        grayscale_saliency_maps.append(frame)
    
    # Stack the grayscale maps to form the output (B, H, W)
    grayscale_saliency_maps = np.stack(grayscale_saliency_maps, axis=0)  # Shape: (B, H, W)
    
    return grayscale_saliency_maps

@torch.no_grad()
def process_images(class_names_ooal, obs, model, crop_size, device='cuda'):
    batch_size, views, frames, channels, height, width = obs.shape
    processed_images = []

    # 定义预处理流程
    preprocess = transforms.Compose([
        transforms.Resize(630),  # 调整大小
        # transforms.ToTensor(),   # 转换为张量
    ])
    reshape = transforms.Resize(128)
    
    for b in range(batch_size):
        for v in range(views):
            for f in range(frames):
                img = obs[b,v,f,:,:,:]
                print(f"shape img is {img.shape}")
                # 执行预处理
                img_t = preprocess(img).unsqueeze(0).to(device)  # 添加batch维度并移动到指定设备
        
                
                # 执行模型预测
                ego_pred = model(img_t.cuda(), gt_aff=[class_names_ooal.index("open"),class_names_ooal.index("middle_drawer")])
                
                ego_pred = np.array(ego_pred.squeeze().data.cpu())  # 将结果转换为numpy数组

                # 对预测结果进行归一化
                ego_pred = normalize_map(ego_pred, crop_size)
                # ego_pred = cv2.resize(ego_pred, dsize=(crop_size, crop_size))


                
                
                # 将每张预测图像保存或处理
                processed_images.append(ego_pred)

    # processed_images = np.reshape(processed_images,(batch_size,views,frames,channels,height,width))
    processed_images = np.array(processed_images)
    print(f"shape processed images is {processed_images.shape}")
    # 这里你可以将处理后的图像批量返回
    return processed_images

@torch.no_grad()
def overlay_images_v1(background, saliency_map, alpha=0.5, beta=0.5):
    """
    Overlay a saliency map onto the background image.
    Args:
        background (np.ndarray): The original observation frame (H, W, 3).
        saliency_map (np.ndarray): The saliency map (H, W).
        alpha (float): Weight for the background image.
        beta (float): Weight for the saliency map.
    Returns:
        np.ndarray: The blended image.
    """
    # Ensure the images are in the correct format (uint8, [0, 255])
    background = background.astype(np.uint8)

    # Apply a color map to the grayscale saliency map
    saliency_map_colored = cv2.applyColorMap(saliency_map.cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)

    # heatmap = cv2.applyColorMap(frame[0], colormap=cv2.COLORMAP_JET)
    saliency_map_colored = cv2.cvtColor(saliency_map_colored, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    # heatmap = np.float32(heatmap)  

    # Blend the images together using weighted sum
    blended_image = cv2.addWeighted(background, alpha, saliency_map_colored, beta, 0)

    return blended_image

@torch.no_grad()
def normalize_grayscale_image(img):
    img_min = torch.min(img)
    img_max = torch.max(img)
    return (img - img_min) / (img_max - img_min)


import torchvision.transforms.functional as F

@torch.no_grad()
def transform_img(image):
    # Resize (keep aspect ratio, and constrain max size)
    image_resized = F.resize(image, (800, 800))  # or any other size
    
    # Normalize (mean and std as per ImageNet pre-trained models, for example)
    image_normalized = F.normalize(image_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image_normalized
@torch.no_grad()
def generate_saliency_obs(obs,track_ori ,vi_ori,sam2_predictor,grounding_model,ooal_model,class_names_ooal):
    print(f"shape obs in new saliency gene func is {obs.shape}")
    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{class_names_ooal}")
    b_size,v_size,t_size,_,_,_ = obs.shape
    sample_track, sample_vi = [], []
    for b in range (b_size):
        sample_track_per_view, sample_vi_per_view = [], []
        for i in range(v_size):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(t_size):
                image = obs[b, i, t]
                img = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image_copy = image
                image_copy = image_copy.permute(1,2,0).cpu().numpy()
                print(f"shape set is {image_copy.shape}")
                sam2_predictor.set_image(img)
                print(f"image shape here is {image.shape}")
                # print(image)
                image = transform_img(image)
                # print(image)
                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=image,
                    caption="the gray bowl. the pink-white plate. the white head. the white robot.",
                    box_threshold=0.15,
                    text_threshold=0.15,
                )
                filtered_indices = [j for j, class_name in enumerate(labels) if class_name != ""]
                boxes = boxes[filtered_indices]
                confidences = [confidences[j] for j in filtered_indices]
                labels = [labels[j] for j in filtered_indices]
                h = 128
                w = 128
                boxes = boxes * torch.Tensor([w, h, w, h])
                boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()                    
                print(boxes)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes,
                        multimask_output=False,
                    )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                class_names = labels
                class_ids = np.array(list(range(len(class_names))))
                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(class_names, confidences)
                ]
                detections = sv.Detections(
                    xyxy=boxes,  # (n, 4)
                    mask=masks.astype(bool),  # (n, h, w)
                    class_id=class_ids
                )
                box_annotator = sv.BoxAnnotator()

                scene_tensor = obs[b, i, t]

                # 将 Tensor 转换为 NumPy ndarray
                scene_ndarray = scene_tensor.permute(1, 2, 0).cpu().numpy()
                scene_ndarray = scene_ndarray.astype(np.uint8)

                
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)
                # track_i_t, vi_i_t = sample_tracks(tracks=track[i, t], vis=vi[i, t], num_samples=64)
                track_i_t, vi_i_t = sample_tracks_object_1(tracks=track_ori[b, i, t], vis=vi_ori[b, i, t], num_samples=128, uniform_ratio=0.25, object_boxes=boxes)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track_per_view.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi_per_view.append(torch.stack(sample_vi_per_time, dim=0))
        sample_track.append(torch.stack(sample_track_per_view, dim=0))
        sample_vi.append(torch.stack(sample_vi_per_view, dim=0))
    track_obj = torch.stack(sample_track, dim=0)
    vi_obj = torch.stack(sample_vi, dim=0)

    track_obj = track_obj.reshape(-1, 16, 128, 2)
    # track_vid = tracks_to_video(track_obj, img_size=128)

    # track_vid_reshape = track_vid.view(4, 2, 10, 3, 128, 128)
    # track_vid = tracks_to_saliency_map_v2(track_obj, img_size=128)
    track_gray_maps = tracks_to_grayscale_maps(track_obj, img_size=128)
    
    # track_vid_reshape = np.reshape(track_vid, (4,2,10,128,128,3))

    # track_vid_reshape = track_vid.view(4,2,10,3,128,128)#uncomment when using other methods
    # track_gray_maps_reshape = np.reshape(track_gray_maps, (4,2,10,128,128,3))
    # print(f"shape of track_vid_reshape is {track_gray_maps_reshape.shape}")

    ooal_images = process_images(class_names_ooal, obs, ooal_model, 128)
    track_gray_maps = torch.tensor(track_gray_maps)
    ooal_images = torch.tensor(ooal_images)


    # normalize method
    # track_gray_maps_norm = normalize_grayscale_image(track_gray_maps)
    # # print(torch.max(track_gray_maps_norm))
    # ooal_images_norm = normalize_grayscale_image(ooal_images)
    # # print(torch.max(ooal_images_norm))
    # norm_add = track_gray_maps_norm + ooal_images_norm 
    # print(torch.max(norm_add))
    # norm_add_norm = normalize_grayscale_image(norm_add)
    # print(torch.max(norm_add_norm))
    # combined_images = norm_add_norm * 250.00
    # combined_images = combined_images.to(torch.uint8)


    # hyperparam method
    ooal_images =  ooal_images*1000
    track_gray_maps = track_gray_maps 
    # # 设定阈值
    # threshold = 50

    # # 将小于阈值的像素值置为0
    # ooal_images = torch.where(ooal_images < threshold, torch.tensor(0.0), ooal_images)

    # 按元素相加
    combined_images = track_gray_maps + ooal_images
    print(track_gray_maps)
    # print(ooal_images)
    # combined_images = combined_images.to(torch.uint8)

    # # 确保图像值在 0 到 255 之间，并转换回 uint8 类型
    # combined_images = torch.clamp(combined_images, 0, 255).to(torch.uint8)
    # 先分别scale到0-1，加起来之后再scale到0-1

    # 将 track_gray_maps 和 ooal_images 都归一化到 0 到 1 之间
    # track_min, track_max = track_gray_maps.min(), track_gray_maps.max()
    # ooal_min, ooal_max = ooal_images.min(), ooal_images.max()

    # track_gray_maps_norm = (track_gray_maps - track_min) / (track_max - track_min)
    # ooal_images_norm = (ooal_images - ooal_min) / (ooal_max - ooal_min)

    # 将它们放缩到 0 到 255 之间
    # track_gray_maps_scaled = (track_gray_maps_norm * 255).to(torch.uint8)
    # ooal_images_scaled = (ooal_images_norm * 255).to(torch.uint8)

    # 合并图像
    # combined_images = track_gray_maps_scaled + ooal_images_scaled

    print(f"shape combined images is {combined_images.shape}")
    combined_images_reshape = combined_images.view(b_size*v_size*t_size,128,128)
    # Apply a color map to the grayscale saliency map
    saliency_map_colored = []
    for batch in range(b_size*v_size*t_size):
        saliency_map_colored_b=cv2.applyColorMap(combined_images_reshape[batch].cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)

    # heatmap = cv2.applyColorMap(frame[0], colormap=cv2.COLORMAP_JET)
        saliency_map_colored_b = cv2.cvtColor(saliency_map_colored_b, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        saliency_map_colored_b = torch.from_numpy(saliency_map_colored_b).clone()
        saliency_map_colored.append(saliency_map_colored_b)
    # saliency_map_colored = torch.from_numpy(saliency_map_colored).clone()
    saliency_map_colored = torch.stack(saliency_map_colored,dim=0)
    saliency_map_colored = saliency_map_colored.view(b_size,v_size,t_size,3,128,128)

    return saliency_map_colored




def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()
