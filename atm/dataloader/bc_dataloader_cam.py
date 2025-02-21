import numpy as np
import torch

from atm.dataloader.base_dataset import BaseDataset
from atm.utils.flow_utils import sample_tracks_nearest_to_grids
from pytorch_grad_cam import GradCAM, FullGrad, EigenCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, GradCAMElementWise, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, preprocess_image_tensor
import torch
import numpy as np
import cv2
import os
import torch
import torchvision.models as models
import json
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
class BCDatasetCAM(BaseDataset):
    def __init__(self, track_obs_fs=1, cam_model=None, cam_method='fullgrad', output_dir='output/check_tt_obs', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.cam_model = cam_model
        self.cam_method = cam_method  
        self.output_dir = output_dir  
        os.makedirs(self.output_dir, exist_ok=True)

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["root"]["task_emb_bert"]
        extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
                        demo['root']['extra_states'].items()}
        # cam_obs = []  # 存储CAM处理后的obs
        
        # for view in range(obs.shape[0]):  # 遍历视角
        #     view_cam_frames = []  # 当前视角的所有时间步的帧
        #     for t in range(obs.shape[1]):  # 遍历时间步长
        #         img_tensor = obs[view, t]  # 提取单个图像 (c, h, w)
        #         # tt_img_tensor = track_transformer_obs[view, t]
        #         # tt_img_tensor = tt_img_tensor.squeeze(0)
        #         # print(f"Shape of tt_img_tensor: {tt_img_tensor.shape}???????????????????????????????????????????????????????????????????????????????????")
                
        #         # 将张量转换为numpy数组以进行CAM处理 (h, w, c)，并归一化
        #         img = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
        #         img_resized = cv2.resize(img, (1000, 1000))
        #         # 预处理图像以适应CAM模型的输入要求
        #         input_tensor = preprocess_image(img_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to("cuda")

        #         # 生成CAM
        #         target_layers = [self.cam_model.layer4]  # 根据模型架构选择目标层
        #         cam_algorithm = KPCA_CAM(model=self.cam_model, target_layers=target_layers)
        #         grayscale_cam = cam_algorithm(input_tensor=input_tensor)[0]
        #         grayscale_cam = cv2.resize(grayscale_cam, (128, 128))

        #         # 在原始图像上叠加CAM，并转换为BGR格式
        #         cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        #         # img_bgr = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #         # tt_img_numpy = tt_img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # Convert to NumPy (h, w, c)
        #         # tt_img_bgr = cv2.cvtColor(tt_img_numpy, cv2.COLOR_RGB2BGR)  # Then apply cvtColor

        #         # cv2.imwrite(os.path.join(self.output_dir,f"ttimg_{index}.jpg"),tt_img_bgr)
        #         # cv2.imwrite(os.path.join(self.output_dir,f"img_{index}.jpg"),img_bgr)
        #         cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)  # 转换为BGR格式

        #         # 将CAM图像调整为obs的尺寸
        #         cam_resized = cv2.resize(cam_image_bgr, (img_tensor.shape[2], img_tensor.shape[1]))
        #         # cv2.imwrite(os.path.join(self.output_dir,f"cam_{index}.jpg"),cam_resized)
        #         # 将图像转换为Tensor格式并转回 (c, h, w)
        #         cam_resized = torch.tensor(cam_resized).permute(2, 0, 1)  # (h, w, c) -> (c, h, w)

        #         # 将RGB归一化到0-255并确保格式与obs匹配
        #         cam_resized = cam_resized.float() / 255.0  # 归一化

        #         view_cam_frames.append(cam_resized)  # 添加到时间步栈中

        #     cam_obs.append(torch.stack(view_cam_frames, dim=0))  # 将时间步栈叠加为 (t, c, h, w)

        # cam_obs = torch.stack(cam_obs, dim=0)  # 将视角叠加为 (v, t, c, h, w)
        # print(f"obs shape of BCDatasetCAM is {obs.shape}")
        # print(f"cam_obs shape of BCDatasetCAM is {cam_obs.shape}")
        cam_obs = generate_cam_resize(obs, self.cam_model)
        cam_obs = cam_obs.cpu()

        return  obs, cam_obs, track_transformer_obs, track, task_embs, actions, extra_states
import torch.nn.functional as F1
def generate_cam_resize(obs, cam_model):
    cam_obs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    views, stacks, c, h, w = obs.shape

    # 展平为批量处理格式：[views * stacks, c, h, w]
    obs_flat = obs.reshape(-1, c, h, w).to(device)
    obs_resized = F1.interpolate(
        obs_flat,  # 增加一个批次维度，形状变为 [1, C, H, W]
        size=(650,650),      # 指定目标尺寸
        mode='bilinear',       # 使用双线性插值
        align_corners=False
    ) #[N, C, H, W]
    obs_tensor = torch.stack([img.permute(2, 0, 1).float() / 255.0 for img in obs_resized]) #[N, W, C, H]
    grad_cams = process_grad_cam_in_batches_cuda(obs_tensor, cam_model)
    for cam, img in zip(grad_cams, obs_flat):
        img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0 #[h,w,c]
        cam_resized = F1.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(128,128), mode='bilinear', align_corners=False)
        cam_resized = cam_resized.squeeze(0).squeeze(0)
        cam_np = cam_resized.cpu().numpy() #?[c,w,h]
        # 调用原始的 show_cam_on_image 函数生成叠加的 CAM 图像
        cam_on_image = show_cam_on_image(img_np, cam_np, use_rgb=True)
        cam_on_image_tensor = torch.from_numpy(cam_on_image).to(device).permute(2, 0, 1).float() / 255.0
        cam_obs.append(cam_on_image_tensor)
    cam_obs = torch.stack(cam_obs,dim=0)
    cam_obs = cam_obs.reshape(views, stacks, *cam_obs.shape[1:])
    return cam_obs

def process_grad_cam_cuda(cropped_images, cam_model):
    """
    使用 Grad-CAM 处理裁剪后的图像
    输入:
        cropped_images: torch.Tensor，形状 [N, C, H, W]
        cam_model: 用于 Grad-CAM 的模型
    输出:
        grayscale_cams: torch.Tensor，Grad-CAM 输出
    """
    # 将张量移动到设备
    device = cam_model.device if hasattr(cam_model, 'device') else "cuda"
    cropped_images = cropped_images.permute(0,2,3,1).to(device)#?[N, W, C, H]--> [N, C, W, H]

    # 初始化 Grad-CAM 算法
    target_layers = [cam_model.layer4]
    cam_algorithm = KPCA_CAM(model=cam_model, target_layers=target_layers)

    # 应用 Grad-CAM
    grayscale_cams = cam_algorithm(input_tensor=cropped_images) # ?[N, c, W, H]
    
    return grayscale_cams

def process_grad_cam_in_batches_cuda(cropped_images, cam_model, batch_size=2):
    """
    分批处理 Grad-CAM
    输入:
        cropped_images: torch.Tensor, 形状 [N, C, H, W]
        cam_model: 用于 Grad-CAM 的模型
        batch_size: 每批处理的图像数量
    输出:
        cam_results: torch.Tensor, Grad-CAM 的输出
    """
    # print(f"test input shape process_grad_cam_in_batches_cuda {cropped_images.shape}") #?[N, W, C, H]
    cam_results = []

    # 确保图像数据在 GPU 上
    device = cam_model.device if hasattr(cam_model, 'device') else "cuda"
    cropped_images = cropped_images.to(device)

    # 分批处理图像
    for i in range(0, len(cropped_images), batch_size):
        batch = cropped_images[i:i + batch_size]  # [B, C, H, W] 批次
        # print(f"Processing batch shape: {batch.shape}")

        # 调用 process_grad_cam
        batch_cam_results = process_grad_cam_cuda(batch, cam_model)

        if isinstance(batch_cam_results, np.ndarray):
            batch_cam_results = torch.from_numpy(batch_cam_results).to(device)
        # 将结果添加到 cam_results 中
        cam_results.append(batch_cam_results)
        # print(f"Batch cam results shape: {batch_cam_results.shape}")

    # 将所有批次的结果连接成一个张量
    cam_results = torch.cat(cam_results, dim=0)


    return cam_results