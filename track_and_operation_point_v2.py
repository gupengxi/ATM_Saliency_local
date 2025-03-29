
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

from models.ooal import Net as model

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

import os
import torch
import supervision as sv
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# torch.set_float32_matmul_precision('high')


output_gif_path = "track_video.gif"
OUTPUT_DIR = Path("outputs/check_box")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)
# 假设 track_vid 是生成的视频帧
# ooal model related
parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='./dataset/')
parser.add_argument('--model_file', type=str, default="./ooal-models/unseen_best")
parser.add_argument('--save_path', type=str, default='./save_preds')
##  image
parser.add_argument('--divide', type=str, default='Seen')
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--resize_size', type=int, default=630)
#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)
args = parser.parse_args()
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")
args.class_names = UNSEEN_AFF
model = model(args, 768, 512).cuda()
model.eval()
state_dict = torch.load(args.model_file)['model_state_dict']
model.load_state_dict(state_dict, strict=False)

output_gif_path = "track_video.gif"
frames = []
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


def tracks_to_saliency_map_v2(tracks, img_size):
    """
    Generate a saliency map that emphasizes the trajectory regions and de-emphasizes the differences between trajectories.
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    img_size: (H, W), the size of the output image
    return: RGB saliency map (B, T, H, W, 3), where each pixel's color emphasizes trajectory regions
    """
    B, T, N, _ = tracks.shape
    binary_vid = tracks_to_binary_img(tracks, img_size=img_size).float()  # (B, T, 1, H, W)
    binary_vid[:, :, 0] = binary_vid[:, :, 1]
    binary_vid[:, :, 2] = binary_vid[:, :, 1]
    print(f"binary_vid shape is {binary_vid.shape}")
    
    # Sum across all time steps to highlight trajectory regions
    saliency_map = binary_vid.sum(dim=1)  # (B, 1, H, W)

    # Normalize to [0, 1] for consistency using torch.amax
    saliency_map = saliency_map / torch.amax(saliency_map, dim=(1, 2, 3), keepdim=True)

    # Apply Gaussian smoothing to emphasize trajectory regions
    gaussian_blur = transforms.GaussianBlur(kernel_size=55, sigma=3.0)
    saliency_map = torch.stack([gaussian_blur(frame) for frame in saliency_map])  # Apply blur to each frame
    
    # Scale back to 0-255 for visualization purposes
    saliency_map = (saliency_map * 255).clamp(0, 255)
    print(f"shape of saliency_map is {saliency_map.shape}") 
    # Convert to numpy for OpenCV processing (B, H, W)
    saliency_map_np = saliency_map.squeeze(1).cpu().numpy()  # Shape: (B, H, W)
    
    print(f"shape of saliency_map_np is {saliency_map_np.shape}")
    # Apply color map to each frame in the batch
    heatmaps = []
    for i in range(B):
        frame = saliency_map_np[i]
        
        # Ensure the frame is of type uint8 (8-bit unsigned integer)
        frame = np.uint8(frame)
        
        # Apply color map to the frame
        heatmap = cv2.applyColorMap(frame[0], colormap=cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        heatmap = np.float32(heatmap)  
        heatmaps.append(heatmap)
    
    # Stack the heatmaps to form the output (B, H, W, 3)
    heatmaps = np.stack(heatmaps, axis=0)  # Shape: (B, H, W, 3)
    
    return heatmaps

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
    
    ######修改：不复制三通道#############
    # 这里去除冗余通道，使 binary_vid 为三通道 (B, T, 3, H, W)
    # binary_vid[:, :, 0] = binary_vid[:, :, 1]
    # binary_vid[:, :, 2] = binary_vid[:, :, 1]

    # print(f"binary_vid shape is {binary_vid.shape}")
    ###########################

    
    # # Sum across all time steps (T) to emphasize trajectory regions
    saliency_map = binary_vid.sum(dim=1)  # (B, 3, H, W)
    ################3修改归一化方式###################3
    # # Normalize to [0, 1] for consistency
    # saliency_map = saliency_map / torch.amax(saliency_map, dim=(1, 2, 3), keepdim=True)
    # 归一化处理
    max_vals = torch.amax(saliency_map, dim=(1,2,3), keepdim=True)
    saliency_map = saliency_map / (max_vals + 1e-8)  # 避免除零
    #############################
    # Apply Gaussian smoothing
    gaussian_blur = transforms.GaussianBlur(kernel_size=33, sigma=5.0)
    ############修改：直接处理四维张量
    # saliency_map = torch.stack([gaussian_blur(frame) for frame in saliency_map])  # (B, 3, H, W)
    saliency_map = gaussian_blur(saliency_map) 
    #############
    # Scale back to 0-255 for visualization (still retaining grayscale)
    ###########修改
    # saliency_map = (saliency_map * 255)
    saliency_map = (saliency_map * 255).clamp(0, 255).byte()
    print(f"shape of saliency_map is {saliency_map.shape}")
    # Convert to numpy for further processing (B, H, W)
    saliency_map_np = saliency_map.sum(dim=1).cpu().numpy()  # Shape: (B, H, W)
    
    return saliency_map_np


def overlay_images(background, saliency_map, alpha=0.5, beta=0.5):
    """
    Overlay a saliency map onto the background image.
    Args:
        background (np.ndarray): The original observation frame (H, W, 3).
        saliency_map (np.ndarray): The saliency map (H, W, 3).
        alpha (float): Weight for the background image.
        beta (float): Weight for the saliency map.
    Returns:
        np.ndarray: The blended image.
    """
    # Ensure the images are in the correct format (uint8, [0, 255])
    background = background.astype(np.uint8)
    saliency_map = saliency_map.astype(np.uint8)

    # Blend the images together using weighted sum
    blended_image = cv2.addWeighted(background, alpha, saliency_map, beta, 0)
    return blended_image

def process_images(obs, model, crop_size, device='cuda'):
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
                ego_pred = model(img_t.cuda(), gt_aff=[args.class_names.index("open"),args.class_names.index("middle_drawer")])
                
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
def reshape_transform_local(tensor, height=64, width=64):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
def normalize_grayscale_image(img):
    img_min = torch.min(img)
    img_max = torch.max(img)
    return (img - img_min) / (img_max - img_min)


import torchvision.transforms.functional as F
def transform_img(image):
    # Resize (keep aspect ratio, and constrain max size)
    image_resized = F.resize(image, (800, 1333))  # or any other size
    
    # Normalize (mean and std as per ImageNet pre-trained models, for example)
    image_normalized = F.normalize(image_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image_normalized

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    grounding_model1 = load_model(
        model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        model_checkpoint_path="gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
        device=DEVICE
    )
    grounding_model2 = load_model(
        model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        model_checkpoint_path="gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
        device=DEVICE
    )
    # Set up the multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)

    output_gif_path = "track_video.gif"
    frames = []
    train_dataset = BCDataset(
        dataset_dir="data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo",   
        img_size=[128, 128],
        frame_stack=10,
        num_track_ts=16,
        num_track_ids=32,
        track_obs_fs=1,
        augment_track=False,
        extra_state_keys=["joint_states", "gripper_states"],
        cache_all=True,
        cache_image=True,
        aug_prob=0.9
    )
    train_loader = get_dataloader(train_dataset, mode="train", num_workers=1, batch_size=4)
    k = 0
    for obs, track_obs, track, task_emb, action, extra_states, track_ori ,vi_ori in tqdm(train_loader):
        # track = track.reshape(-1, 16, 64, 2)
        # track_vid = tracks_to_video(track, img_size=128)

        # track_vid_reshape = track_vid.view(4, 2, 10, 3, 128, 128)
        # boxes, confidences, labels = [], [], []
        # sample tracks
        sample_track, sample_vi = [], []
        gd_cams = []
        for b in range (4):
            sample_track_per_view, sample_vi_per_view = [], []
            for i in range(2):
                sample_track_per_time, sample_vi_per_time = [], []
                for t in range(10):
                    # track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                    image = obs[b, i, t]
                    img = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                    # original 
                    image_copy = image
                    image_copy = image_copy.permute(1,2,0).cpu().numpy()
                    sam2_predictor.set_image(img)
                    image = transform_img(image)
                    boxes, confidences, labels = predict(
                        model=grounding_model1,
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

                    scene_ndarray = scene_tensor.permute(1, 2, 0).cpu().numpy()
                    scene_ndarray = scene_ndarray.astype(np.uint8)

                    
                    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                    mask_annotator = sv.MaskAnnotator()
                    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)
                    
                    track_i_t, vi_i_t = sample_tracks_object_1(tracks=track_ori[b, i, t], vis=vi_ori[b, i, t], num_samples=128, uniform_ratio=0.25, object_boxes=boxes)
                    # track_i_t, vi_i_t = sample_tracks_object_1(tracks=track_ori[b, i, t], vis=vi_ori[b, i, t], num_samples=128, uniform_ratio=0.25)
                    sample_track_per_time.append(track_i_t.cuda())
                    sample_vi_per_time.append(vi_i_t)
                    
                    image_c = image

                    #generate gd cam
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        # transforms.Resize(256),
                        transforms.CenterCrop(256),
                        # transforms.ToTensor(),  # 将图像转换为张量
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
                    ])
                    im_for_gd = preprocess(image_c)
                    print(f"shape imforgd {im_for_gd.shape}")
                    im_for_gd = im_for_gd.unsqueeze(0)
                    # print(f"shape imforgd {im_for_gd.shape}")
                    target_layers = [grounding_model2.backbone[0].layers[0].blocks[-1].norm1]
                    cam_algorithm = KPCA_CAM(model=grounding_model2, target_layers=target_layers,reshape_transform=reshape_transform_local)
                    gd_cam = cam_algorithm(input_tensor=im_for_gd)
                    gd_cams.append(gd_cam)
                sample_track_per_view.append(torch.stack(sample_track_per_time, dim=0))
                sample_vi_per_view.append(torch.stack(sample_vi_per_time, dim=0))
            sample_track.append(torch.stack(sample_track_per_view, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_view, dim=0))
        track_obj = torch.stack(sample_track, dim=0)
        vi_obj = torch.stack(sample_vi, dim=0)

        gd_cams = torch.tensor(gd_cams)#80 1 256 256
        # 1. 移除通道维度（通过 squeeze）
        gd_cams = gd_cams.squeeze(1)  # 形状变为 [80, 256, 256]
        import torch.nn.functional as F1
        # 2. 调整空间尺寸（使用插值或池化）
        gd_cams = F1.interpolate(gd_cams.unsqueeze(1), size=(128, 128), mode='bilinear', align_corners=False)
        gd_cams = gd_cams.squeeze(1)
        print(f"shape gd cams {gd_cams.shape}")
        track_obj = track_obj.reshape(-1, 16, 128, 2)

        track_gray_maps = tracks_to_grayscale_maps(track_obj, img_size=128)

        ooal_images = process_images(obs, model, 128)
        track_gray_maps = torch.tensor(track_gray_maps)
        ooal_images = torch.tensor(ooal_images)


        # normalize method
        #########################################################
        track_gray_maps_norm = normalize_grayscale_image(track_gray_maps)
        # print(torch.max(track_gray_maps_norm))
        ooal_images_norm = normalize_grayscale_image(ooal_images)
        gd_cams_norm = normalize_grayscale_image(gd_cams)
        # print(torch.max(ooal_images_norm))
        norm_add = track_gray_maps_norm + ooal_images_norm + gd_cams_norm
        print(torch.max(norm_add))
        norm_add_norm = normalize_grayscale_image(norm_add)
        # print(torch.max(norm_add_norm))
        combined_images = norm_add_norm * 255
        combined_images = torch.clamp(combined_images, 0 , 255)
        combined_images = combined_images.to(torch.uint8)
        #########################################################


        print(f"shape combined images is {combined_images.shape}")
        combined_images_reshape = combined_images.view(4,2,10,128,128)


        for b1 in range(combined_images_reshape.shape[0]):
            for v1 in range(combined_images_reshape.shape[1]):

                for t in range(combined_images_reshape.shape[2]):  # 遍历时间步
                    # Get the current observation frame and the corresponding saliency map frame
                    obs_frame = obs[b1, v1, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H, W, C)
                    saliency_frame = combined_images_reshape[b1, v1, t, :, :]  # (H, W)
                    
                    # Overlay the saliency map onto the observation frame
                    blended_frame = overlay_images_v1(obs_frame, saliency_frame, alpha=0.1, beta=0.5)
                    blended_frame_tensor = torch.from_numpy(blended_frame)
                    saliency_frame = saliency_frame.unsqueeze(0)  # 变成 (1, H, W)

                    # 然后，如果需要将灰度图扩展到 RGB 形状 (3, H, W)，你可以重复它的通道
                    saliency_frame = saliency_frame.repeat(3, 1, 1)  # 变成 (3, H, W)

                    print(saliency_frame.shape)
                    print(blended_frame_tensor.shape)
                    blended_frame_tensor = blended_frame_tensor.permute(2, 0, 1) 

                    cat_frame = torch.cat([obs[b1,v1,t], saliency_frame, blended_frame_tensor], dim=2)

                    # Add to frames for GIF creation
                    # frames.append(Image.fromarray(blended_frame))
                    to_pil = transforms.ToPILImage()

                    pil_image = to_pil(cat_frame)

                    frames.append(pil_image)

        frames[0].save(
            f"track_video_{k}.gif", save_all=True, append_images=frames[1:], loop=0, duration=100
        )
        
        print(f"Saved GIF to {output_gif_path}")

        k += 1

if __name__ == '__main__':
    main()


 