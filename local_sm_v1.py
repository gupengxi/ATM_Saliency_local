import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import argparse
import signal
import sys
import torchvision.models as models


"""
Hyper parameters
"""

TEXT_PROMPT = "the gray bowl. the pink-white plate."
IMG_PATH = "data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/images/demo_0/agentview_0.png"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

def signal_handler(sig, frame):
    print("\nProcess interrupted by user. Exiting gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build Grounding DINO model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

image_source, image = load_image(IMG_PATH)
sam2_predictor.set_image(image_source)

# Predict bounding boxes with Grounding DINO
boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# Filter predictions
filtered_indices = [i for i, class_name in enumerate(labels) if class_name != ""]
boxes = boxes[filtered_indices]
confidences = [confidences[i] for i in filtered_indices]
labels = [labels[i] for i in filtered_indices]

# Process the box prompt for SAM2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

# Use autocast for GPU acceleration
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
    print(f"masks is {masks.shape}")

# Convert mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

class_names = labels
class_ids = np.array(list(range(len(class_names))))
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

# Visualize image with supervision
img = cv2.imread(IMG_PATH)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)
box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
# Argument parsing for CAM methods
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument('--image-path', type=str, default=IMG_PATH,
                        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen-smooth', action='store_true',
                        help='Reduce noise by taking the first principle component of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcamelementwise',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam'
                        ],
                        help='CAM method')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory to save the images')
    return parser.parse_args()

args = get_args()
methods = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
    "gradcamelementwise": GradCAMElementWise,
    'kpcacam': KPCA_CAM
}

# Apply CAM to each detected region
cam_model = models.resnet50(pretrained=True).to(DEVICE).eval()
cam_algorithm = methods[args.method]
target_layers = [cam_model.layer4]

final_image_with_all_cams = img.copy()

# Use the existing CAM context
with cam_algorithm(model=cam_model, target_layers=target_layers) as cam:
    targets = None
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
        grayscale_cam = cam(
            input_tensor=input_tensor, targets=targets,
            aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth
        )
        cam_on_image = show_cam_on_image(cropped_region, grayscale_cam[0], use_rgb=True)
        resized_cam_result = cv2.resize(cam_on_image, (x_max - x_min, y_max - y_min))
        print(resized_cam_result.shape)
        print("?????????????????????????????/")
        current_mask = masks[i, y_min:y_max, x_min:x_max].astype(np.uint8)  # 取得当前box的mask
        
        # 创建一个透明背景（RGBA），与裁剪区域相同大小
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

    # Save the final image with CAM results applied
    output_path = os.path.join(OUTPUT_DIR, "grounded_sam2_cam_all_results.jpg")
    cv2.imwrite(output_path, final_image_with_all_cams)
    
    print(final_image_with_all_cams.shape)
    print("????????????????????????????????????????")
    scale_factor = 1
    height, width = img.shape[:2]
    img = cv2.resize(img, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)
    # Apply CAM to the entire image
    input_tensor = preprocess_image(
        np.float32(img) / 255.0, mean=[0.485, 0.456, 0.406], std=[0.01, 0.01, 0.01]
    )

    # Generate CAM for the entire image
    grayscale_cam = cam(input_tensor=input_tensor, targets=None, 
                        aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)

    # Normalize to the range [0, 1] and create a mask for the lightest regions
    grayscale_cam = (grayscale_cam[0] - grayscale_cam[0].min()) / (grayscale_cam[0].max() - grayscale_cam[0].min())

    cam_overlay = show_cam_on_image(np.float32(img) / 255.0, grayscale_cam, use_rgb=True)
    # highlighted_overlay = cv2.bitwise_and(cam_overlay, cam_overlay, mask=lightest_mask)
    cam_overlay = cv2.resize(cam_overlay,(height,width))
    # Combine highlighted CAM overlay with the image processed with bounding boxes
    final_image = cv2.addWeighted(final_image_with_all_cams, 0.7, cam_overlay, 0.3, 0)

    # Save the final image
    output_path = os.path.join(OUTPUT_DIR, "combined_highlighted_cam_results_1.jpg")
    cv2.imwrite(output_path, final_image)
"""
Dump the results in standard format and save as json files
"""
if DUMP_JSON_RESULTS:
    # Convert mask into RLE format
    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    
    # Save results in standard format
    results = {
        "image_path": IMG_PATH,
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)
