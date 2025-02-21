import torch
import os
import hydra
import torch
import numpy as np
import lightning

import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from pytorch_grad_cam import GradCAM, FullGrad, EigenCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, GradCAMElementWise, KPCA_CAM, ScoreCAM
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
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict, batch_predict, batch_predict_1
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as T
import torch
import torchvision.transforms as T
from typing import Union
cam_model = models.resnet50(pretrained=True).to("cuda").eval()
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "./checkpoints/sam2.1_hiera_large.pt", device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    model_checkpoint_path="gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
    device="cuda"
)
import warnings

warnings.simplefilter("ignore")
def main():
    test_generate_cam_for_obs(cam_model, sam2_predictor, grounding_model, "obs_batches.pth")



def test_generate_cam_for_obs(cam_model, sam2_predictor, grounding_model, filename):
    # Load the saved batches of obs
    obs_batches = torch.load(filename)
    print(f"Loaded {len(obs_batches)} batches of 'obs'")

    # Run generate_cam_for_obs on the saved batches
    cam_obs = []
    for obs in obs_batches:
        print(f"obs shape is {obs.shape}")
        start_time = time.time()
        cam_obs_batch = generate_cam_for_obs_batch(obs, cam_model, sam2_predictor, grounding_model)
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.4f} seconds")
        # cam_obs_batch = generate_cam_for_obs(obs, cam_model, sam2_predictor, grounding_model)
        cam_obs.append(cam_obs_batch)
        print(f"cam obs batch size {cam_obs_batch.shape}") 
    cam_obs = torch.stack(cam_obs, dim=0)
    print(f"cam_obs shape {cam_obs.shape}")
    print(f"Generated CAM for {len(cam_obs)} batches.")
import time
@torch.no_grad()
def generate_cam_for_obs(obs, cam_model, sam2_predictor, grounding_model):
    print(f"obs shape in tqdm loading is {obs.shape}") #torch.Size([8, 2, 10, 3, 128, 128])
    cam_obs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Iterate over each item in the batch
    start_time = time.time()  # Start time for the batch
    for batch_idx in range(obs.shape[0]):
        batch_start_time = time.time()  # Start time for each batch
        print(f"Processing batch {batch_idx + 1}/{obs.shape[0]}")
        view_cam_frames = []

        # Iterate over each view
        view_start_time = time.time()  # Start time for views
        for view_idx in range(obs.shape[1]):
            view_end_time = time.time() - view_start_time
            print(f"Time for view {view_idx + 1}/{obs.shape[1]}: {view_end_time:.4f} seconds")
            frame_cam_frames = []

            # Iterate over each stacked frame (10 frames)
            frame_start_time = time.time()  # Start time for frames
            for stack_idx in range(obs.shape[2]):
                frame_end_time = time.time() - frame_start_time
                print(f"Time for stack {stack_idx + 1}/{obs.shape[2]}: {frame_end_time:.4f} seconds")
                
                img_tensor = obs[batch_idx, view_idx, stack_idx]  # Shape: [3, 128, 128]

                # Ensure img_tensor has 3 dimensions (c, h, w)
                if img_tensor.dim() != 3:
                    raise ValueError(f"Expected img_tensor to have 3 dimensions (c, h, w), but got {img_tensor.shape}")

                # Convert the tensor to a NumPy array and normalize
                img = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                # Timing for Grounding DINO prediction
                grounding_start_time = time.time()
                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=transform_image(img),
                    caption="gray_bowl. white plate.",
                    box_threshold=0.25,
                    text_threshold=0.1,
                    device=device
                )
                # Filter predictions
                filtered_indices = [i for i, class_name in enumerate(labels) if class_name != ""]
                boxes = boxes[filtered_indices]
                confidences = [confidences[i] for i in filtered_indices]
                labels = [labels[i] for i in filtered_indices]
                visualize_predictions(img,boxes,labels,save_path=f"result.png")
            
                grounding_end_time = time.time() - grounding_start_time
                print(f"Time for Grounding DINO prediction: {grounding_end_time:.4f} seconds")

                # Timing for SAM2 prediction
                sam2_start_time = time.time()
                # Resize boxes to original image dimensions
                h, w = img.shape[:2]
                boxes = boxes * torch.tensor([w, h, w, h])
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                sam2_predictor.set_image(img)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True

                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
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
                sam2_end_time = time.time() - sam2_start_time
                print(f"Time for SAM2 prediction: {sam2_end_time:.4f} seconds")

                # Timing for CAM generation
                cam_start_time = time.time()
                target_layers = [cam_model.layer4]
                cam_algorithm = KPCA_CAM(model=cam_model, target_layers=target_layers)
                final_image_with_all_cams = img.copy()
                for i, box in enumerate(input_boxes):
                    # Extract coordinates for the bounding box
                    x_min, y_min, x_max, y_max = map(int, box)
                    
                    # Extract and preprocess the cropped region
                    cropped_region = img[y_min:y_max, x_min:x_max]
                    cropped_region = np.float32(cropped_region) / 255.0
                    # Resize the cropped region before processing it in CAM
                    scale_factor = 20  # Adjust the scale factor as needed
                    height, width = cropped_region.shape[:2]
                    cropped_region = cv2.resize(cropped_region, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)
                
                    input_tensor = preprocess_image(
                        cropped_region, mean=[0.485, 0.456, 0.406], std=[0.01, 0.01, 0.01]
                    )
                    
                    # Run CAM and resize the overlay to match the original bounding box size
                    grayscale_cam = cam_algorithm(input_tensor=input_tensor)
                    cam_on_image = show_cam_on_image(cropped_region, grayscale_cam[0], use_rgb=True)
                    resized_cam_result = cv2.resize(cam_on_image, (x_max - x_min, y_max - y_min))
                    current_mask = masks[i, y_min:y_max, x_min:x_max].astype(np.uint8) 
                    transparent_region = np.zeros((y_max - y_min, x_max - x_min, 4), dtype=np.uint8)
                    for c in range(3):  # 处理 R, G, B 三个通道
                        transparent_region[:, :, c] = np.where(
                            current_mask == 1,  # 掩码区域
                            resized_cam_result[:, :, c],  # 使用 CAM 结果覆盖
                            final_image_with_all_cams[y_min:y_max, x_min:x_max,c] # 保留原始背景
                        )
                    # 设置 Alpha 通道，使得mask区域内不透明，其他区域透明
                    transparent_region[:, :, 3] = current_mask * 255

                    # 将处理后的透明图像叠加回原始图像
                    final_image_with_all_cams[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                        final_image_with_all_cams[y_min:y_max, x_min:x_max], 0.7,
                        transparent_region[:, :, :3], 0.3, 0
                    )

                scale_factor = 1
                height, width = img.shape[:2]
                img = cv2.resize(img, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)
                # Apply CAM to the entire image
                input_tensor = preprocess_image(
                    np.float32(img) / 255.0, mean=[0.485, 0.456, 0.406], std=[0.01, 0.01, 0.01]
                )
                

                # Generate CAM for the entire image
                grayscale_cam = cam_algorithm(input_tensor=input_tensor)
                print(f"normal shape img {img.shape}, normal shape cam[0] {grayscale_cam[0].shape} ")

                # Normalize to the range [0, 1] and create a mask for the lightest regions
                grayscale_cam = (grayscale_cam[0] - grayscale_cam[0].min()) / (grayscale_cam[0].max() - grayscale_cam[0].min())
                # lightest_mask = (grayscale_cam > 0.5).astype(np.uint8)  # Keep only the top 20% activation regions
                # Convert CAM output to a color overlay and resize to match original image
                cam_overlay = show_cam_on_image(np.float32(img) / 255.0, grayscale_cam, use_rgb=True)
                # highlighted_overlay = cv2.bitwise_and(cam_overlay, cam_overlay, mask=lightest_mask)
                cam_overlay = cv2.resize(cam_overlay,(height,width))
                # Combine highlighted CAM overlay with the image processed with bounding boxes
                final_image = cv2.addWeighted(final_image_with_all_cams, 0.9, cam_overlay, 0.1, 0)
                output_path = os.path.join(Path("outputs/grounded_sam2_local_demo"), "combined_highlighted_cam_results.jpg")
                cv2.imwrite(output_path, final_image)
                final_image = torch.from_numpy(final_image)
                frame_cam_frames.append(final_image)


            # Stack all frames for the current view (stack, c, h, w)
            frame_cam_frames = torch.stack(frame_cam_frames, dim=0)
            view_cam_frames.append(frame_cam_frames)

        # Stack all views for the current batch item (views, stack, c, h, w)
        view_cam_frames = torch.stack(view_cam_frames, dim=0)
        cam_obs.append(view_cam_frames)

        batch_end_time = time.time() - batch_start_time
        print(f"Time for batch {batch_idx + 1}: {batch_end_time:.4f} seconds")

    # Stack all batch items (batch, views, stack, c, h, w)
    cam_obs = torch.stack(cam_obs, dim=0)
    total_time = time.time() - start_time
    print(f"Total time for generating CAM for all batches: {total_time:.4f} seconds")
    return cam_obs


def transform_image(image: np.array) -> torch.Tensor:
    """
    Converts a NumPy image to a tensor suitable for model input.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an image with 3 channels (RGB), but got {image.shape[2]} channels.")
    
    transform = T.Compose(
        [
            T.ToPILImage(),  # Convert NumPy array to PIL image
            T.Resize((800, 800)),  # Resize
            T.ToTensor(),  # Convert to Tensor
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
        ]
    )
    return transform(image)


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)

@torch.no_grad()
def generate_cam_for_obs_batch(obs, cam_model, sam2_predictor, grounding_model):
    """
    输入:
        obs: torch.Size([batch, views, stacks, c, h, w])
    输出:
        combined_results: torch.Size([batch, views, stacks, h, w, 3]) (叠加后的最终结果)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, views, stacks, c, h, w = obs.shape

    # 展平为批量处理格式：[batch * views * stacks, c, h, w]
    obs_flat = obs.reshape(-1, c, h, w).to(device)

    # 1. Grounding DINO 批量框选
    results = process_grounding_dino(obs_flat, grounding_model, device)
    # 2. 提取 boxes，用于 SAM2
    boxes_list = [result[0] for result in results]
    # 2. SAM2 批量掩码生成
    masks, input_boxes_list = process_sam2_batch(obs_flat, boxes_list, sam2_predictor)

    # 3. Grad-CAM 批量处理目标框
    cropped_images, crop_boxes, num_of_crop_box = crop_and_resize_boxes_cuda(obs_flat, input_boxes_list)

    # 如果裁剪后的图片为空，跳过 Grad-CAM 处理
    if not cropped_images:
        print("No cropped images generated. Skipping Grad-CAM.")
        grad_cams = None
    else:
        # cropped_images_tensor = torch.stack([torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in cropped_images])
        cropped_images_tensor = torch.stack([img.permute(2, 0, 1).float() / 255.0 for img in cropped_images])

        grad_cams = process_grad_cam_in_batches_cuda(cropped_images_tensor, cam_model)

    if grad_cams is None:
        print("Grad-CAM results are empty. Skipping combination step.")
        return None
    else:
        combined_results = combine_cam_and_masks_cuda(obs_flat, grad_cams, masks, crop_boxes, num_of_crop_box)
        combined_results = combined_results.reshape(batch_size, views, stacks, *combined_results.shape[1:])
    # 4. 叠加结果
    # combined_results = combine_cam_and_masks(obs_flat, grad_cams, masks, crop_boxes, resize_scales)

    # 5. 重构结果为原始结构
    combined_results = combined_results.reshape(batch_size, *combined_results.shape[1:])
    return combined_results

def process_grounding_dino(obs_flat, grounding_model, device, caption="gray_bowl. white plate."):
    """
    对批量帧进行 Grounding DINO 框选
    """
    start_gd = time.time()

    images_np = preprocess_obs_flat(obs_flat)  # [batch * views * stacks, h, w, c]

    # 批量预处理
    images_tensor = batch_preprocess_images(images_np, device="cuda")  # [N, C, H, W]
    print(f"images_tensor shape {images_tensor.shape}")

    batch_size = 4

    # 使用混合精度和按批处理
    results = []
    with torch.cuda.amp.autocast(): 
        for i in range(0, len(images_tensor), batch_size):
            batch_images = images_tensor[i:i + batch_size]
            # with torch.cuda.amp.autocast():  # 混合精度
            batch_results = batch_predict_1(
                model=grounding_model,
                images=batch_images,
                caption=caption,
                box_threshold=0.25,
                text_threshold=0.1,
                device="cuda"
            )
            results.extend(batch_results)
    # for idx, (boxes, _, phrases) in enumerate(results):
    #     if boxes.numel() > 0:  # 确保有预测结果
    #         visualize_predictions(images_np[idx], boxes, phrases, save_path=f"result.png")
    # print(results)
    end_gd = time.time()
    print(f"gd time {end_gd-start_gd}")
    return results
import matplotlib.pyplot as plt

def visualize_predictions(image_np, boxes, phrases, save_path=None):
    """
    可视化预测结果并保存或显示图片
    输入:
        image_np: numpy array, 原始图片 [H, W, C]
        boxes: torch.Tensor, 预测的边界框 [N, 4]
        phrases: List[str], 预测的标签
        save_path: str, 可选，保存路径
    """
    image_vis = image_np.copy()
    h, w, _ = image_vis.shape

    # 绘制边界框和标签
    for box, phrase in zip(boxes, phrases):
        x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h])).int().tolist()
        cv2.rectangle(image_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边框
        cv2.putText(image_vis, phrase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存或显示
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
        print(f"Image saved to {save_path}")
    else:
        plt.imshow(image_vis)
        plt.axis("off")
        plt.show()
def preprocess_obs_flat(obs_flat: torch.Tensor) -> np.ndarray:
    """
    将 obs_flat 批量转换为 NumPy 格式 (h, w, c)
    输入:
        obs_flat: torch.Size([batch * views * stacks, c, h, w])
    输出:
        np.ndarray: 批量处理后的图片，形状为 [batch * views * stacks, h, w, c]
    """
    # 转换为 NumPy 格式
    # 先不转化为np
    pre_obs_flat_start = time.time()
    obs_flat_np = obs_flat.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # [N, h, w, c]
    # obs_flat = obs_flat.permute(0, 2, 3, 1)
    pre_obs_flat_end = time.time()
    print(f"pre_obs_flat time {pre_obs_flat_end-pre_obs_flat_start}")
    return obs_flat_np

import torchvision.transforms.functional as F

def batch_preprocess_images(images_np: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    批量预处理图片，支持 Resize 和 Normalize。
    输入:
        images_np: np.ndarray，形状为 [N, H, W, C]
        device: str，目标设备 ('cuda' 或 'cpu')
    输出:
        torch.Tensor: 预处理后的图片张量，形状为 [N, C, H, W]
    """
    batch_pre_img_start = time.time()
    # 转换为 PyTorch 张量，形状变为 [N, C, H, W]
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]

    # Resize 批量处理（支持统一尺寸），并移动到目标设备
    target_size = 800
    images_resized = F.resize(images_tensor, [target_size, target_size]).to(device) # 移动到目标设备

    # Normalize 批量处理
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)  # 广播到 [1, C, 1, 1]
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)    # 广播到 [1, C, 1, 1]
    images_normalized = (images_resized - mean) / std
    batch_pre_img_end = time.time()
    print(f"batch_pre_img time {batch_pre_img_end-batch_pre_img_start}")
    return images_normalized
# def batch_preprocess_images(images_tensor: torch.Tensor, device: str = "cuda", batch_size: int = 32) -> torch.Tensor:
#     """
#     批量预处理图片，支持 Resize 和 Normalize。
#     输入:
#         images_tensor: torch.Tensor，形状为 [N, C, H, W]
#         device: str，目标设备 ('cuda' 或 'cpu')
#         batch_size: int，每次处理的批次大小
#     输出:
#         torch.Tensor: 预处理后的图片张量，形状为 [N, C, H, W]
#     """
#     results = []
#     target_size = 800
    
#     # 分批处理以减小显存占用
#     for i in range(0, len(images_tensor), batch_size):
#         batch_images = images_tensor[i:i + batch_size].permute(0, 3, 1, 2).to("cpu")  # 首先在 CPU 上操作以节省 GPU 内存
#         # Resize
#         resized_images = F.resize(batch_images, [target_size, target_size])
#         # Normalize
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
#         normalized_images = (resized_images - mean) / std
#         normalized_images = normalized_images.to(device)  # 分批移动到目标设备

#         results.append(normalized_images)

#     return torch.cat(results, dim=0)

def process_sam2(obs_flat, boxes_list, sam2_predictor):
    """
    对批量帧中的目标框生成掩码

    """
    sam2_start = time.time()
    print(f"))))))))))){len(boxes_list)}")#160
    input_boxes_list = []
    sam2_masks = []
    for img, boxes in zip(obs_flat, boxes_list):
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        h, w = img.shape[:2]
        boxes = boxes * torch.tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        input_boxes_list.append(input_boxes)
        # print(boxes.shape)
        sam2_predictor.set_image(img)
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # print(masks)
        sam2_masks.append(masks)
    print(f"masks len is ((((({len(sam2_masks)}")#160
    sam2_end = time.time()
    print(f"sam2 time {sam2_end-sam2_start}")
    return sam2_masks, input_boxes_list
def process_sam2_batch(obs_flat, boxes_list, sam2_predictor, batch_size=16):
    """
    对批量帧中的目标框生成掩码，支持自定义批量大小。

    输入:
        obs_flat: torch.Tensor, 形状为 [total_frames, c, h, w] 的图像张量
        boxes_list: 每个图像对应的 box 列表
        sam2_predictor: SAM2 的预测器对象，支持批量输入
        batch_size: 每次处理的图像数量

    输出:
        sam2_masks: 生成的掩码列表
        input_boxes_list: 标准化后的框列表
    """
    sam2_start = time.time()
    total_frames = len(boxes_list)
    sam2_masks = []
    input_boxes_list = []

    for i in range(0, total_frames, batch_size):
        batch_obs = obs_flat[i:i + batch_size]
        print(f"batch obs shape is {batch_obs.shape}")
        batch_boxes = boxes_list[i:i + batch_size]


        # 转换为 numpy 格式
        # image_list = [img.permute(1, 2, 0).cpu().numpy().astype(np.uint8) for img in batch_obs]
        # 改到gpu
        # image_list = [img.permute(1,2,0) for img in batch_obs]
        image_list = batch_obs.permute(0,2,3,1)
        print(f"tensor image_list shape is {image_list.shape}")
        
        # 转换框坐标为像素并存储
        batch_input_boxes = []
        # 改到gpu
        # for img, boxes in zip(image_list, batch_boxes):
        #     h, w = img.shape[:2]
        #     # 改到gpu
        #     # boxes = boxes * torch.tensor([w, h, w, h])  # 像素坐标
        #     # input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        #     boxes = boxes * torch.tensor([w, h, w, h],device="cuda")  # 像素坐标
        #     input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        #     batch_input_boxes.append(input_boxes)
        
        # input_boxes_list.extend(batch_input_boxes)
        for img, boxes in zip(batch_obs, batch_boxes):
            # 从 batch_obs 中的每个图像中获取高度和宽度
            _, h, w = img.shape  # img.shape 为 (C, H, W)，我们只需要高度 (H) 和宽度 (W)
            
            # 保持所有操作在 GPU 上进行
            device = boxes.device  # 确保使用与 boxes 相同的设备
            scale_tensor = torch.tensor([w, h, w, h], device=device)  # 计算坐标的缩放系数
            
            # 将 cxcywh 格式转换为像素坐标的 xyxy 格式
            boxes = boxes * scale_tensor
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            
            # 将处理后的框直接添加到列表中
            batch_input_boxes.append(input_boxes)

        # 如果后续的函数仍需要批量框数据作为 numpy 数组，才进行最终的转换
        input_boxes_list.extend(batch_input_boxes)
        # 批量设置图像
        sam2_predictor.set_image_batch_cuda(image_list)

        # 批量预测
        batch_masks, _, _ = sam2_predictor.predict_batch_cuda(
            point_coords_batch=None,
            point_labels_batch=None,
            box_batch=batch_input_boxes,
            multimask_output=False,
        )

        sam2_masks.extend(batch_masks)

    print(f"Total masks generated: {len(sam2_masks)}")  # 打印生成的掩码总数
    sam2_end = time.time()
    print(f"sam2 time {sam2_end-sam2_start}")
    return sam2_masks, input_boxes_list

def crop_and_resize_boxes(obs_flat, boxes_list, target_size=(500, 500)):
    """
    对输入图片的边界框进行裁剪并缩放到统一大小，同时记录缩放比例。
    
    Parameters:
    - obs_flat: list of image tensors
    - boxes_list: list of bounding boxes for each image
    - target_size: tuple (width, height) indicating the desired output size for each crop

    Returns:
    - cropped_images: list of resized crops
    - resize_scales: list of scales for each crop (width_scale, height_scale)
    - crop_boxes: list of the original bounding box coordinates
    """
    crop_resize_start = time.time()
    print(f"Length of boxes list is {len(boxes_list)}")
    cropped_images = []
    # resize_scales = []
    crop_boxes = []
    num_of_crop_box = []
    # original_crop_h = []
    # original_crop_w = []

    for img_tensor, boxes in zip(obs_flat, boxes_list):
        # 如果 boxes 为空，跳过当前图片
        # print(f"boxes shape is {boxes.shape}")
        num_of_crop_box.append(len(boxes))
        # if len(boxes) == 0:
        #     print(f"No objects found in image, skipping...")
        #     continue

        img = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            # # 边界框验证
            # if x_min < 0 or y_min < 0 or x_max > img.shape[1] or y_max > img.shape[0]:
            #     print(f"Invalid box: {box}, Image shape: {img.shape}")
            #     continue

            cropped = img[y_min:y_max, x_min:x_max]
            # if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            #     print(f"Empty crop for box: {box}, skipping...")
            #     continue

            # 获取当前裁剪的宽高
            h, w = cropped.shape[:2]

            # 缩放比例
            width_scale = target_size[0] / w
            height_scale = target_size[1] / h

            # 缩放到指定大小
            cropped_resized = cv2.resize(cropped, target_size)

            # 保存结果
            cropped_images.append(cropped_resized)
            # resize_scales.append((width_scale, height_scale))
            crop_boxes.append((x_min, y_min, x_max, y_max))
            # original_crop_h.append(h)
            # original_crop_w.append(w)
    
    print(f"Length of cropped images is {len(cropped_images)}")
    print(f"hhhhhh{len(num_of_crop_box)}")
    print(num_of_crop_box)
    crop_resize_end = time.time()
    print(f"crop_resize time {crop_resize_end-crop_resize_start}")

    return cropped_images, crop_boxes, num_of_crop_box
import torch.nn.functional as F1
def crop_and_resize_boxes_cuda(obs_flat, boxes_list, target_size=(500, 500)):
    """
    对输入图片的边界框进行裁剪并缩放到统一大小，同时记录缩放比例。
    
    Parameters:
    - obs_flat: list of image tensors
    - boxes_list: list of bounding boxes for each image
    - target_size: tuple (width, height) indicating the desired output size for each crop

    Returns:
    - cropped_images: list of resized crops as tensors
    - crop_boxes: list of the original bounding box coordinates
    - num_of_crop_box: list of the number of crops for each image
    """
    crop_resize_start = time.time()
    print(f"Length of boxes list is {len(boxes_list)}")
    cropped_images = []
    crop_boxes = []
    num_of_crop_box = []

    # 将目标尺寸转换为张量
    target_size_tensor = torch.tensor(target_size, device="cuda")

    for img_tensor, boxes in zip(obs_flat, boxes_list):
        # 如果 boxes 为空，跳过当前图片
        num_of_crop_box.append(len(boxes))

        # 确保图像在 GPU 上
        img_tensor = img_tensor.to('cuda')

        for box in boxes:
            # 将边界框坐标转换为整数
            x_min, y_min, x_max, y_max = map(int, box)

            # 裁剪图像：img_tensor 的形状为 [C, H, W]
            cropped = img_tensor[:, y_min:y_max, x_min:x_max]
            if cropped.size(1) == 0 or cropped.size(2) == 0:
                # 跳过空裁剪
                print(f"Empty crop for box: {box}, skipping...")
                continue

            # 获取当前裁剪的宽高
            h, w = cropped.shape[1:]

            # 调整到目标尺寸
            cropped_resized = F1.interpolate(
                cropped.unsqueeze(0),  # 增加一个批次维度，形状变为 [1, C, H, W]
                size=target_size,      # 指定目标尺寸
                mode='bilinear',       # 使用双线性插值
                align_corners=False
            ).squeeze(0)  # 去除批次维度，形状变回 [C, target_height, target_width]

            # 保存结果
            cropped_images.append(cropped_resized)
            crop_boxes.append((x_min, y_min, x_max, y_max))

    print(f"Length of cropped images is {len(cropped_images)}")
    print(f"hhhhhh{len(num_of_crop_box)}")
    print(num_of_crop_box)
    crop_resize_end = time.time()
    print(f"crop_resize time {crop_resize_end-crop_resize_start}")

    return cropped_images, crop_boxes, num_of_crop_box
# def process_grad_cam(cropped_images, cam_model):
#     """
#     使用 Grad-CAM 处理裁剪后的图像
#     输入:
#         cropped_images: torch.Tensor，形状 [N, C, H, W]
#         cam_model: 用于 Grad-CAM 的模型
#     输出:
#         grayscale_cams: torch.Tensor，Grad-CAM 输出
#     """
#     # 如果输入是列表，将其转换为张量
#     if isinstance(cropped_images, list):
#         cropped_images = torch.stack([torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in cropped_images])

#     # 将张量移动到设备
#     device = cam_model.device if hasattr(cam_model, 'device') else "cuda"
#     cropped_images = cropped_images.to(device)

#     # 初始化 Grad-CAM 算法
#     target_layers = [cam_model.layer4]
#     cam_algorithm = KPCA_CAM(model=cam_model, target_layers=target_layers)

#     # 应用 Grad-CAM
#     grayscale_cams = cam_algorithm(input_tensor=cropped_images)
#     return grayscale_cams
def process_grad_cam(cropped_images, cam_model):
    """
    使用 Grad-CAM 处理裁剪后的图像
    输入:
        cropped_images: torch.Tensor，形状 [N, C, H, W]
        cam_model: 用于 Grad-CAM 的模型
    输出:
        grayscale_cams: torch.Tensor，Grad-CAM 输出
    """
    # 如果输入是列表，将其转换为张量
    if isinstance(cropped_images, list):
        cropped_images = torch.stack([torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in cropped_images])

    # 调整通道顺序（如果需要）
    if cropped_images.shape[1] != 3:
        cropped_images = cropped_images.permute(0, 3, 1, 2)  # 将通道放在第二维
        print("Corrected shape:", cropped_images.shape)

    # 将张量移动到设备
    device = cam_model.device if hasattr(cam_model, 'device') else "cuda"
    cropped_images = cropped_images.to(device)

    # 初始化 Grad-CAM 算法
    target_layers = [cam_model.layer4]
    cam_algorithm = KPCA_CAM(model=cam_model, target_layers=target_layers)
    # cam_algorithm.batch_size=4
    # 应用 Grad-CAM
    grayscale_cams = cam_algorithm(input_tensor=cropped_images)
    
    return grayscale_cams

def process_grad_cam_in_batches(cropped_images, cam_model, batch_size=4):
    print(f"Type cropped images is {type(cropped_images)}")
    cam_start = time.time()
    cam_results = []
    for i in range(0, len(cropped_images), batch_size):
        batch = cropped_images[i:i + batch_size]
        # 将 batch 转换为 Tensor
        device = cam_model.device if hasattr(cam_model, 'device') else "cuda"
        batch_tensor = torch.stack([torch.tensor(img).permute(2,0,1).float() / 255.0 for img in batch]).to(device)
        print(f"batch tensor shape {batch_tensor.shape}")
        # 调用 cam_model
        batch_cam_results = process_grad_cam(batch_tensor, cam_model)
        # cams_on_images = show_cam_on_image(batch_tensor, batch_cam_results[0], use_rgb=True)
        cam_results.extend(batch_cam_results)
        print(f"batch cam results {batch_cam_results.shape}")
        # cam_on_image_results.extend(cams_on_images)
    cam_end = time.time()
    print(f"cam time {cam_end-cam_start}")
    return cam_results

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
    cropped_images = cropped_images.permute(0,2,1,3).to(device)

    # 初始化 Grad-CAM 算法
    target_layers = [cam_model.layer4]
    cam_algorithm = KPCA_CAM(model=cam_model, target_layers=target_layers)

    # 应用 Grad-CAM
    grayscale_cams = cam_algorithm(input_tensor=cropped_images)
    
    return grayscale_cams

def process_grad_cam_in_batches_cuda(cropped_images, cam_model, batch_size=4):
    """
    分批处理 Grad-CAM
    输入:
        cropped_images: torch.Tensor, 形状 [N, C, H, W]
        cam_model: 用于 Grad-CAM 的模型
        batch_size: 每批处理的图像数量
    输出:
        cam_results: torch.Tensor, Grad-CAM 的输出
    """
    print(f"Type of cropped_images: {type(cropped_images)}")
    cam_start = time.time()
    cam_results = []

    # 确保图像数据在 GPU 上
    device = cam_model.device if hasattr(cam_model, 'device') else "cuda"
    cropped_images = cropped_images.to(device)

    # 分批处理图像
    for i in range(0, len(cropped_images), batch_size):
        batch = cropped_images[i:i + batch_size]  # [B, C, H, W] 批次
        print(f"Processing batch shape: {batch.shape}")

        # 调用 process_grad_cam
        batch_cam_results = process_grad_cam_cuda(batch, cam_model)

        if isinstance(batch_cam_results, np.ndarray):
            batch_cam_results = torch.from_numpy(batch_cam_results).to(device)
        # 将结果添加到 cam_results 中
        cam_results.append(batch_cam_results)
        print(f"Batch cam results shape: {batch_cam_results.shape}")

    # 将所有批次的结果连接成一个张量
    cam_results = torch.cat(cam_results, dim=0)

    cam_end = time.time()
    print(f"Grad-CAM processing time: {cam_end - cam_start}")

    return cam_results

def combine_cam_and_masks(obs_flat, grad_cams, masks, crop_boxes, num_of_crop_box):
    """
    Combine Grad-CAM results with masks for each image and restore them to their original positions.
    """
    print(f"{len(obs_flat), len(grad_cams), len(masks), len(crop_boxes)}")
    # 160,394,160,394
    final_results = []

    # 用于跟踪当前处理的索引位置
    start_idx = 0
    
    for img_idx, (img,mask) in enumerate(zip(obs_flat, masks)):
        # 当前图片对应的裁剪框数量
        n_crops = num_of_crop_box[img_idx]
        
        # 获取当前图片的 grad_cams、masks、crop_boxes、original_crop_h 和 original_crop_w
        img_grad_cams = grad_cams[start_idx:start_idx + n_crops]
        # img_masks = masks[start_idx:start_idx + n_crops]
        img_crop_boxes = crop_boxes[start_idx:start_idx + n_crops]
        # img_original_h = original_crop_h[start_idx:start_idx + n_crops]
        # img_original_w = original_crop_w[start_idx:start_idx + n_crops]
        
        # 更新索引位置
        start_idx += n_crops

        # 克隆原图以避免修改原始数据
        final_result = img.clone()
        final_result = final_result.permute(1, 2, 0).cpu().numpy()
        print(f"mask shape is {mask.shape}")

        for cam, crop_box in zip(img_grad_cams, img_crop_boxes):
            x_min, y_min, x_max, y_max = crop_box
            cropped_img = img[:, y_min:y_max, x_min:x_max]
            print(f"single img {img.shape} ")
            print(f"{x_min}, {x_max}, {y_min}, {y_max}")
            print(f"test cropped_img type {type(cropped_img)}")
            print(f"test cropped_img shape {cropped_img.shape}")
            print(f"cam0 type {type(cam[0])}")
            print(f"cam0 shape {cam[0].shape}")
            cropped_img = np.float32(cropped_img.permute(1, 2, 0).cpu().numpy()) / 255.0
            cam = cv2.resize(cam, (x_max - x_min,y_max - y_min))
            print(f"shape cam resized shape {cam.shape}")
            cam_on_image = show_cam_on_image(cropped_img, cam, use_rgb=True)
            resized_cam_result = cv2.resize(cam_on_image, (x_max - x_min, y_max - y_min))#no need to record h, w???
            # mask = mask.cpu().numpy()
            if mask.ndim == 4:
                current_mask = mask[0, 0, y_min:y_max, x_min:x_max].cpu().numpy().astype(np.uint8)
                # current_mask = mask[0, 0, y_min:y_max, x_min:x_max].to(torch.uint8)
            elif mask.ndim == 3:
                current_mask = mask[0, y_min:y_max, x_min:x_max].cpu().numpy().astype(np.uint8)
                # current_mask = mask[0, y_min:y_max, x_min:x_max].to(torch.uint8)
            transparent_region = np.zeros((y_max - y_min, x_max - x_min, 4), dtype=np.uint8)
            print(f"transparent region shape {transparent_region.shape}")
            print(f"current mask shape {current_mask.shape}")
            for c in range(3):  # 处理 R, G, B 三个通道
                transparent_region[:, :, c] = np.where(
                    current_mask == 1,  # 掩码区域
                    resized_cam_result[:, :, c],  # 使用 CAM 结果覆盖
                    final_result[y_min:y_max, x_min:x_max,c] # 保留原始背景
                )
                        # 设置 Alpha 通道，使得mask区域内不透明，其他区域透明
            transparent_region[:, :, 3] = current_mask * 255
            transparent_region = transparent_region.astype(np.float32)
            final_result[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                final_result[y_min:y_max, x_min:x_max], 0.7,
                transparent_region[:, :, :3], 0.3, 0
            )
            output_path = os.path.join(Path("outputs/grounded_sam2_local_demo"), "Test_Time.jpg")
            cv2.imwrite(output_path, final_result)
        final_results.append(final_result)
# Convert only if the element is a NumPy array
    final_results = [torch.from_numpy(result).float() if isinstance(result, np.ndarray) else result for result in final_results]


    return torch.stack(final_results, dim=0)


def combine_cam_and_masks_cuda(obs_flat, grad_cams, masks, crop_boxes, num_of_crop_box):
    """
    Combine Grad-CAM results with masks for each image and restore them to their original positions.
    """
    print(f"{len(obs_flat), len(grad_cams), len(masks), len(crop_boxes)}")
    final_results = []

    # 用于跟踪当前处理的索引位置
    start_idx = 0

    # 确保输入数据都在 GPU 上
    device = obs_flat.device if isinstance(obs_flat, torch.Tensor) else "cuda"
    # obs_flat = obs_flat.to(device)
    # grad_cams = grad_cams.to(device)
    # masks = masks.to(device)

    for img_idx, (img, mask) in enumerate(zip(obs_flat, masks)):
        # 当前图片对应的裁剪框数量
        n_crops = num_of_crop_box[img_idx]

        # 获取当前图片的 grad_cams、masks、crop_boxes
        img_grad_cams = grad_cams[start_idx:start_idx + n_crops]
        img_crop_boxes = crop_boxes[start_idx:start_idx + n_crops]

        # 更新索引位置
        start_idx += n_crops

        # 克隆原图以避免修改原始数据
        final_result = img.clone()

        for cam, crop_box in zip(img_grad_cams, img_crop_boxes):
            x_min, y_min, x_max, y_max = map(int, crop_box)

            # 获取裁剪的图像区域
            cropped_img = img[:, y_min:y_max, x_min:x_max]
            if cropped_img.size(1) == 0 or cropped_img.size(2) == 0:
                print(f"Empty crop for box: {crop_box}, skipping...")
                continue

            # 将图像和 CAM 移动到 CPU 并转换为 NumPy 格式
            cropped_img_np = cropped_img.permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
            
            cam_resized = F1.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(y_max - y_min, x_max - x_min), mode='bilinear', align_corners=False)
            cam_resized = cam_resized.squeeze(0).squeeze(0)
            cam_np = cam_resized.cpu().numpy()
            # 调用原始的 show_cam_on_image 函数生成叠加的 CAM 图像
            cam_on_image = show_cam_on_image(cropped_img_np, cam_np, use_rgb=True)

            # 将生成的图像再次转换为 GPU 上的张量
            cam_on_image_tensor = torch.from_numpy(cam_on_image).to(device).permute(2, 0, 1).float() / 255.0

            # 使用 mask 来将结果应用到 final_result 中
            if mask.ndim == 4:
                current_mask = mask[0, 0, y_min:y_max, x_min:x_max].to(torch.uint8)
            elif mask.ndim == 3:
                current_mask = mask[0, y_min:y_max, x_min:x_max].to(torch.uint8)

            # 创建一个新的透明区域
            transparent_region = torch.zeros((4, y_max - y_min, x_max - x_min), dtype=torch.float32, device=device)

            # 填充 R, G, B 通道
            for c in range(3):
                transparent_region[c, :, :] = torch.where(
                    current_mask == 1,
                    cam_on_image_tensor[c, :, :],
                    final_result[c, y_min:y_max, x_min:x_max]
                )

            # 设置 alpha 通道
            transparent_region[3, :, :] = current_mask * 255.0

            # 叠加到原图
            final_result[:, y_min:y_max, x_min:x_max] = final_result[:, y_min:y_max, x_min:x_max] * 0.7 + transparent_region[:3, :, :] * 0.3

        final_results.append(final_result)

    # 堆叠所有结果
    final_results = torch.stack(final_results, dim=0)

    return final_results





# Example usage:
if __name__ == "__main__":
    main()