import hydra
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import lightning
from lightning.fabric import Fabric

import os
import wandb
import json
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atm.dataloader import BCDataset, get_dataloader
from atm.policy import *
from atm.utils.train_utils import setup_optimizer, setup_lr_scheduler, init_wandb
from atm.utils.log_utils import MetricLogger, BestAvgLoss
from atm.utils.env_utils import build_env
from engine.utils import rollout, merge_results
import torch.multiprocessing as mp
import tempfile 
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from pytorch_grad_cam import GradCAM, FullGrad, EigenCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, GradCAMElementWise, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import os
import torchvision.models as models

import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as T
from atm.utils.flow_utils import sample_tracks_nearest_to_grids

if GlobalHydra().is_initialized():
    GlobalHydra().clear()
@hydra.main(config_path="../conf/train_bc", version_base="1.3")

def main(cfg: DictConfig):
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    additional_config_path = "/home/pengxi/Documents/ATM/"  # Adjust with your actual path
    initialize_config_dir(config_dir=additional_config_path)
    cam_model = models.resnet50(pretrained=True).to(torch.device("cuda")).eval()
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))
    train_dataset = BCDatasetCAM(dataset_dir=cfg.train_dataset, cam_model=cam_model, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset,
                                mode="train",
                                num_workers=cfg.num_workers,
                                batch_size=cfg.batch_size)
    train_vis_dataset = BCDatasetCAM(dataset_dir=cfg.train_dataset, cam_model=cam_model, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_vis_dataloader = get_dataloader(train_vis_dataset,
                                              mode="train",
                                              num_workers=1,
                                              batch_size=1)

    val_dataset = BCDatasetCAM(dataset_dir=cfg.val_dataset, num_demos=cfg.val_num_demos, cam_model=cam_model,  **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    val_vis_dataset = BCDatasetCAM(dataset_dir=cfg.val_dataset, num_demos=cfg.val_num_demos, cam_model=cam_model,  **cfg.dataset_cfg, aug_prob=0.)
    val_vis_dataloader = get_dataloader(val_vis_dataset, mode="train", num_workers=1, batch_size=1)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), precision="bf16-mixed" if cfg.mix_precision else None, strategy="deepspeed")
    fabric.launch()

    None if (cfg.dry or not fabric.is_global_zero) else init_wandb(cfg)

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    # initialize the environments in each rank
    cfg.env_cfg.render_gpu_ids = cfg.env_cfg.render_gpu_ids[fabric.global_rank] if isinstance(cfg.env_cfg.render_gpu_ids, list) else cfg.env_cfg.render_gpu_ids
    env_num_each_rank = math.ceil(len(cfg.env_cfg.env_name) / fabric.world_size)
    env_idx_start_end = (env_num_each_rank * fabric.global_rank,  min(env_num_each_rank * (fabric.global_rank + 1), len(cfg.env_cfg.env_name)))
    rollout_env = build_env(img_size=cfg.img_size, env_idx_start_end=env_idx_start_end, **cfg.env_cfg)
    rollout_horizon = cfg.env_cfg.get("horizon", None)

    fabric.barrier()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    # Pick ckpt based on  the average of the last 5 epochs
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

    fabric.barrier()
    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            cfg.clip_grad,
            mix_precision=cfg.mix_precision,
            scheduler=scheduler,
        )


        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

            if epoch % cfg.val_freq == 0:
                val_metrics = evaluate(model,
                                          val_loader,
                                          mix_precision=cfg.mix_precision,
                                          tag="val",
                                          cam_model=cam_model)

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
            # tempfile.TemporaryFile().close()

            def vis_and_log(model, vis_dataloader, mode="train"):
                eval_dict = visualize(model, vis_dataloader, cam_model=cam_model, mix_precision=cfg.mix_precision)

                caption = f"reconstruction (right) @ epoch {epoch}; \n Track MSE: {eval_dict['track_loss']:.4f}; Img MSE: {eval_dict['img_loss']:.4f}"
                wandb_image = wandb.Image(eval_dict["combined_image"], caption=caption)
                wandb_cam_image = wandb.Image(eval_dict["cam_obs"])
                wandb_vid_rollout = wandb.Video(eval_dict["combined_track_vid"], fps=24, format="mp4", caption=caption)
                None if cfg.dry else wandb.log({f"{mode}/first_frame": wandb_image,
                                                f"{mode}/cam_image": wandb_cam_image,
                                                f"{mode}/rollout_track": wandb_vid_rollout},
                                                step=epoch)

            if fabric.is_global_zero and hasattr(model, "forward_vis"):
                vis_and_log(model, train_vis_dataloader, mode="train")
                vis_and_log(model, val_vis_dataloader, mode="val")

            gathered_results = [{} for _ in range(fabric.world_size)]
            # cam_model = val_vis_dataset.cam_model
            results = rollout(rollout_env, model, 20 // cfg.env_cfg.vec_env_num, horizon=rollout_horizon, cam_model=cam_model, dist=dist)
            fabric.barrier()
            dist.all_gather_object(gathered_results, results)
            if fabric.is_global_zero:
                gathered_results = merge_results(gathered_results)
                None if cfg.dry else wandb.log(gathered_results, step=epoch)

                for k in list(results.keys()):
                    if k.startswith("rollout/vis_"):
                        results.pop(k)

                metric_logger.update(**results)
        fabric.barrier()

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(fabric,
                  model,
                  dataloader,
                  optimizer,
                  clip_grad=1.0,
                  mix_precision=False,
                  scheduler=None,
                  ):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    tot_loss_dict, tot_items = {}, 0

    model.train()
    i = 0
    for obs, cam_obs, track_obs, track, task_emb, action, extra_states in tqdm(dataloader):
        if mix_precision:
            obs, cam_obs, track_obs, track, task_emb, action = obs.bfloat16(), cam_obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}
        print(f"shape of obs is {obs.shape}")

        loss, ret_dict = model.forward_loss(obs, cam_obs, track_obs, track, task_emb, extra_states, action)
        optimizer.zero_grad()
        fabric.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

        i += 1

    out_dict = {}
    for k, v in tot_loss_dict.items():
        out_dict[f"train/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    if scheduler is not None:
        scheduler.step()

    return out_dict


@torch.no_grad()
def evaluate(model, dataloader, mix_precision=False, tag="val", cam_model=None):
    print("#############evaluating#############")
    tot_loss_dict, tot_items = {}, 0
    model.eval()

    i = 0
    for obs, cam_obs, track_obs, track, task_emb, action, extra_states in tqdm(dataloader):
        obs, cam_obs, track_obs, track, task_emb, action = obs.cuda(), cam_obs.cuda(), track_obs.cuda(), track.cuda(), task_emb.cuda(), action.cuda()
        extra_states = {k: v.cuda() for k, v in extra_states.items()}
        if mix_precision:
            obs, cam_obs, track_obs, track, task_emb, action = obs.bfloat16(), cam_obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}

        _, ret_dict = model.forward_loss(obs, cam_obs, track_obs, track, task_emb, extra_states, action)

        i += 1

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

    out_dict = {}
    for k, v in tot_loss_dict.items():
        out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    return out_dict


@torch.no_grad()
def visualize(model, dataloader, cam_model=None, mix_precision=False):
    model.eval()
    keep_eval_dict = None

    for obs, cam_obs, track_obs, track, task_emb, action, extra_states in dataloader:
        obs, cam_obs, track_obs, track, task_emb = obs.cuda(), cam_obs.cuda(), track_obs.cuda(), track.cuda(), task_emb.cuda()
        extra_states = {k: v.cuda() for k, v in extra_states.items()}
        if mix_precision:
            obs, cam_obs, track_obs, track, task_emb = obs.bfloat16(), cam_obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}

        _, eval_dict = model.forward_vis(obs, cam_obs, track_obs, track, task_emb, extra_states, action)
        keep_eval_dict = eval_dict
        break

    return keep_eval_dict


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)

def apply_cam(obs, cam_model):
    """
    对观测数据 (obs) 应用 CAM 模型。
    参数:
    - obs (torch.Tensor): 形状为 (v, t, stack, c, h, w) 的观测数据。
    - cam_model (nn.Module): 用于生成显著性图的 CAM 模型。

    返回值:
    - cam_obs (torch.Tensor): 经过 CAM 处理的观测数据，形状与原始观测数据相同。
    """
    cam_obs = []

    for view in range(obs.shape[0]):  # 遍历所有视角 (v)
        view_cam_frames = []
        for t in range(obs.shape[1]):  # 遍历时间步长 (t)
            stack_cam_frames = []
            for stack in range(obs.shape[2]):  # 遍历堆叠的帧 (stack)
                img_tensor = obs[view, t, stack]  # 提取单个图像 (c, h, w)

                # 检查图像维度是否符合要求 (c, h, w)
                if img_tensor.dim() != 3:
                    raise ValueError(f"Expected img_tensor to have 3 dimensions (c, h, w), but got {img_tensor.shape}")

                # 将张量转换为 NumPy 数组并归一化
                img = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
                # 预处理图像以适应 CAM 模型的输入要求
                input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to("cuda")

                # 生成 CAM
                target_layers = [cam_model.layer4]
                cam_algorithm = KPCA_CAM(model=cam_model, target_layers=target_layers)
                grayscale_cam = cam_algorithm(input_tensor=input_tensor)[0]

                # 在原始图像上叠加 CAM
                cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

                # 将 CAM 图像调整为与原始 obs 尺寸匹配
                cam_resized = cv2.resize(cam_image, (img_tensor.shape[2], img_tensor.shape[1]))
                cam_resized = torch.tensor(cam_resized).permute(2, 0, 1).float() / 255.0  # 归一化并调整维度顺序

                stack_cam_frames.append(cam_resized)

            # 将堆叠帧组合为 (stack, c, h, w)
            stack_cam_frames = torch.stack(stack_cam_frames, dim=0)
            view_cam_frames.append(stack_cam_frames)

        # 将时间步叠加为 (t, stack, c, h, w)
        view_cam_frames = torch.stack(view_cam_frames, dim=0)
        cam_obs.append(view_cam_frames)

    # 将所有视角叠加为 (v, t, stack, c, h, w)
    cam_obs = torch.stack(cam_obs, dim=0)
    return cam_obs

def apply_cam_batch(obs, cam_model):
    # 获取输入的尺寸
    v, t, stack, c, h, w = obs.shape
    cam_obs = []

    # 对于每一个视角，批量处理时间帧和堆叠帧
    for view in range(v):
        obs_view = obs[view]  # 取出单个视角的所有时间帧和堆叠帧，形状为 (t, stack, c, h, w)
        
        # 重塑张量，以批量形式处理所有时间步和堆叠帧
        obs_batch = obs_view.reshape(-1, c, h, w)  # (t * stack, c, h, w)
        
        # 预处理整个批次
        # Note: 这里可以根据模型的需求调整 mean 和 std
        preprocess = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(obs_batch / 255.0).to("cuda")

        # 生成 CAM
        target_layers = [cam_model.layer4]
        cam_algorithm = GradCAM(model=cam_model, target_layers=target_layers)
        grayscale_cam = cam_algorithm(input_tensor=input_tensor)  # 形状为 (t * stack, h, w)

        # 将 CAM 叠加在原始图像上
        cam_images = []
        for i in range(grayscale_cam.shape[0]):
            img = obs_batch[i].permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
            cam_image = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
            cam_images.append(cam_image)

        # 将处理后的 CAM 图像转换为 Tensor，并恢复原始的形状
        cam_images = torch.tensor(cam_images).permute(0, 3, 1, 2).float() / 255.0  # (t * stack, c, h, w)
        cam_obs_view = cam_images.reshape(t, stack, c, h, w)  # 恢复为 (t, stack, c, h, w)

        cam_obs.append(cam_obs_view)

    # 将所有视角叠加为 (v, t, stack, c, h, w)
    cam_obs = torch.stack(cam_obs, dim=0)

    return cam_obs

class BCDatasetCAM(BCDataset):
    def __init__(self, cam_model=None, cam_method='fullgrad', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cam_model = cam_model
        self.cam_method = cam_method  

    def __getitem__(self, index):
        # 调用父类的 __getitem__ 方法来获得基础的数据
        obs, track_transformer_obs, track, task_embs, actions, extra_states = super().__getitem__(index)

        # Generate CAM dynamically
        cam_obs = self.generate_cam_for_obs(obs)

        return obs, cam_obs, track_transformer_obs, track, task_embs, actions, extra_states

    def generate_cam_for_obs(self, obs):
        """批量生成 CAM，以提高效率。"""
        cam_obs = []

        for view in range(obs.shape[0]):  # 遍历每个视角
            obs_view = obs[view]  # (t, c, h, w)
            obs_batch = obs_view.reshape(-1, *obs_view.shape[1:])  # 将时间步展开以批量处理，形状为 (t, c, h, w)
            
            # 预处理整个批次
            preprocess = T.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(obs_batch / 255.0).to("cuda")  # 将输入移动到 GPU 上并进行归一化

            # 生成 CAM
            target_layers = [self.cam_model.layer4]
            cam_algorithm = KPCA_CAM(model=self.cam_model, target_layers=target_layers)
            grayscale_cam = cam_algorithm(input_tensor=input_tensor)

            # 将 CAM 叠加到原始图像上
            cam_images = []
            for i in range(grayscale_cam.shape[0]):
                img = obs_batch[i].permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
                cam_image = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
                cam_images.append(cam_image)

            cam_images = torch.tensor(cam_images).permute(0, 3, 1, 2).float() / 255.0  # (t, c, h, w)
            cam_obs.append(cam_images)

        cam_obs = torch.stack(cam_obs, dim=0)  # (v, t, c, h, w)
        return cam_obs

if __name__ == "__main__":
    torch.cuda.empty_cache()
    mp.set_start_method('spawn', force=True)
    main()