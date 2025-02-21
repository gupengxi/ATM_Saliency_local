
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
# 假设 track_vid 是生成的视频帧
output_gif_path = "track_video.gif"
OUTPUT_DIR = Path("outputs/check_box")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)
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


def tracks_to_video_v2(tracks, img_size):
    """
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    return: (B, C, H, W)
    """
    B, T, N, _ = tracks.shape
    binary_vid = tracks_to_binary_img(tracks, img_size=img_size).float()  # b, t, c, h, w
    binary_vid[:, :, 0] = binary_vid[:, :, 1]
    binary_vid[:, :, 2] = binary_vid[:, :, 1]

    # Create a colormap from yellow to red
    cmap = np.linspace([1.0, 0.5, 0.0], [1.0, 0.0, 0.0], T)  # Interpolate from yellow (1, 1, 0) to red (1, 0, 0)
    binary_vid = binary_vid.clone()

    for l in range(T):
        # Apply the colormap to each frame
        binary_vid[:, l, 0] = binary_vid[:, l, 0] * cmap[l, 0] * 255  # Red channel
        binary_vid[:, l, 1] = binary_vid[:, l, 1] * cmap[l, 1] * 255  # Green channel
        binary_vid[:, l, 2] = binary_vid[:, l, 2] * cmap[l, 2] * 255  # Blue channel

    # Combine the frames into a single video
    track_vid = torch.sum(binary_vid, dim=1)
    track_vid[track_vid > 255] = 255  # Clip values to valid range
    return track_vid

def tracks_to_saliency_map(tracks, img_size):
    """
    Generate a saliency map that emphasizes the trajectory regions and de-emphasizes the differences between trajectories.
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    img_size: (H, W), the size of the output image
    return: (B, 1, H, W), saliency map for each video
    """
    B, T, N, _ = tracks.shape
    binary_vid = tracks_to_binary_img(tracks, img_size=img_size).float()  # (B, T, 1, H, W)
    binary_vid[:, :, 0] = binary_vid[:, :, 1]
    binary_vid[:, :, 2] = binary_vid[:, :, 1]
    # Sum across all time steps to highlight trajectory regions
    saliency_map = binary_vid.sum(dim=1)  # (B, 1, H, W)

    # Normalize to [0, 1] for consistency using torch.amax
    saliency_map = saliency_map / torch.amax(saliency_map, dim=(1, 2, 3), keepdim=True)

    # Apply Gaussian smoothing to emphasize trajectory regions
    gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=2.0)
    saliency_map = torch.stack([gaussian_blur(frame) for frame in saliency_map])  # Apply blur to each frame
    
    # Scale back to 0-255 for visualization purposes
    saliency_map = (saliency_map * 255).clamp(0, 255)
    print(f"shape is {saliency_map.shape}")

    
    return saliency_map
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
    gaussian_blur = transforms.GaussianBlur(kernel_size=33, sigma=5.0)
    saliency_map = torch.stack([gaussian_blur(frame) for frame in saliency_map])  # Apply blur to each frame
    print(saliency_map)
    # Scale back to 0-255 for visualization purposes
    saliency_map = (saliency_map * 255 *2).clamp(0, 255)
    print(f"shape of saliency_map is {saliency_map.shape}") 
    print(saliency_map)
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
    # saliency_map = saliency_map.cpu().numpy()  # Convert to NumPy array if it's a tensor
    saliency_map = saliency_map.astype(np.uint8)
    print(f"shape back is {background.shape}")
    print(f"shape saliency is {saliency_map.shape}")
    # Blend the images together using weighted sum
    blended_image = cv2.addWeighted(background, alpha, saliency_map, beta, 0)
    return blended_image

import torchvision.transforms.functional as F

def transform_img(image):
    # Resize (keep aspect ratio, and constrain max size)
    image_resized = F.resize(image, size=(800, 800))  # or any other size
    
    # Normalize (mean and std as per ImageNet pre-trained models, for example)
    image_normalized = F.normalize(image_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return image_normalized
def overlay_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Overlay detection boxes on the image.
    
    Args:
        image (np.ndarray): The input image (H, W, 3).
        boxes (np.ndarray): The bounding boxes (N, 4), where each box is (xmin, ymin, xmax, ymax).
        color (tuple): Color for the bounding boxes (default is green).
        thickness (int): Thickness of the bounding box lines (default is 2).
        
    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """
    image_with_boxes = image
    image_with_boxes = image_with_boxes.cpu().numpy()
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(img=image_with_boxes, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=thickness)
    return image_with_boxes
# import multiprocessing

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    grounding_model = load_model(
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
        for b in range (4):
            sample_track_per_view, sample_vi_per_view = [], []
            for i in range(2):
                sample_track_per_time, sample_vi_per_time = [], []
                for t in range(10):
                    # track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                    image = obs[b, i, t]
                    img = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    image_copy = image
                    image_copy = image_copy.permute(1,2,0).cpu().numpy()
                    print(f"shape set is {image_copy.shape}")
                    sam2_predictor.set_image(img)
                    print(f"image shape here is {image.shape}")
                    image = transform_img(image)
                    boxes, confidences, labels = predict(
                        model=grounding_model,
                        image=image,
                        caption="the gray bowl. the pink-white plate. the white end effector.",
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
                    # image_with_boxes = overlay_boxes_on_image(image, boxes)


                    # label_annotator = sv.LabelAnnotator()
                    # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), image_with_boxes)

                    
                    
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
                    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)
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
        track_vid = tracks_to_saliency_map_v2(track_obj, img_size=128)
        
        track_vid_reshape = np.reshape(track_vid, (4,2,10,128,128,3))


        for t in range(track_vid_reshape.shape[2]):
            obs_frame = obs[0, 0, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            saliency_frame = track_vid_reshape[0, 0, t, :, :, :]
            # saliency_frame = track_vid_reshape[0, 0, t, :, :, :].permute(1, 2, 0)
            # saliency_frame = np.transpose(track_vid_reshape[0, 0, t, :, :, :], (1, 2, 0))

            blended_frame = overlay_images(obs_frame, saliency_frame, alpha=0.3, beta=0.9)

            frames.append(Image.fromarray(blended_frame))

        frames[0].save(
            f"track_video_{k}.gif", save_all=True, append_images=frames[1:], loop=0, duration=100
        )

        print(f"Saved GIF to {output_gif_path}")
        k += 1

if __name__ == '__main__':
    main()
# multiprocessing.set_start_method('spawn', force=True)
# i = 0


# for obs, track_obs, track, task_emb, action, extra_states in tqdm(train_loader):
#     print(f"shape of obs is {obs.shape}")
#     print(f"shape of track_obs is {track_obs.shape}")
#     print(f"shape of track is {track.shape}")
#     track = track.reshape(-1, 16, 64, 2)
    
#     # track_vid = tracks_to_saliency_map_v2(track, img_size=128)
#     # track_vid = tracks_to_saliency_map(track, img_size=128)
#     track_vid = tracks_to_video(track, img_size=128)

#     print(f"shape of track vid is {track_vid.shape}")

#     track_vid_reshape = track_vid.view(4,2,10,3,128,128)#uncomment when using other methods
#     # track_vid_reshape = np.reshape(track_vid, (4,2,10,128,128,3))
#     print(f"shape of track_vid_reshape is {track_vid_reshape.shape}")
#     # gif
#     # for t in range(track_vid_reshape.shape[2]):  # 遍历时间步
#     #     # frame = track_vid_reshape[0, 0, t, :, :, :].permute(1,2,0).cpu().numpy().astype(np.uint8) #uncomment when using other methods
#     #     frame = track_vid_reshape[0, 0, t, :, :, :].astype(np.uint8)
#     #     frames.append(Image.fromarray(frame))

#     for t in range(track_vid_reshape.shape[2]):  # 遍历时间步
#         # Get the current observation frame and the corresponding saliency map frame
#         obs_frame = obs[0, 0, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H, W, C)
#         saliency_frame = track_vid_reshape[0, 0, t, :, :, :]  # (H, W, 3)
        
#         # Overlay the saliency map onto the observation frame
#         blended_frame = overlay_images(obs_frame, saliency_frame, alpha=0.1, beta=0.9)

#         # Add to frames for GIF creation
#         frames.append(Image.fromarray(blended_frame))

#     frames[0].save(
#         f"track_video_{i}.gif", save_all=True, append_images=frames[1:], loop=0, duration=100
#     )
    
#     print(f"Saved GIF to {output_gif_path}")
#     i+=1

 