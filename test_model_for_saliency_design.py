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


# 设置设备（CUDA 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
# model = mcr.load_mcr(ckpt_path="models/r3mdroid_resnet50.pth")
# model = mcr.load_mcr(ckpt_path="models/mcr_resnet50.pth")
# model = models.resnet50(pretrained=True).to(torch.device(device)).eval()
# load strongest vit/resnet models
# transform, model = load_vit()
# transform, model = load_resnet18()
# model,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_LARGE_NAME)
# model,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
model = load_model("grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py", "gdino_checkpoints/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = "data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/images/demo_0/agentview_0.png"
# TEXT_PROMPT = "the black bowl. the white plate."
TEXT_PROMPT = "circle"
print(model)
# 加载并预处理图像
img_path = 'images_test/demo_0/agentview_0.png'
# img_path = "data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/images/demo_0/agentview_0.png"
# img_path = "grounding_dino/.asset/cat_dog.jpeg"
img = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
# img_t = model_transforms(img)
# 定义预处理流程：调整大小、裁剪、归一化等
preprocess = transforms.Compose([
    transforms.Resize(1200),
    # transforms.Resize(256),
    transforms.CenterCrop(800),
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 对图像进行预处理
img_tensor = preprocess(img).unsqueeze(0)  
# img_tensor = img_t.unsqueeze(0)
with torch.no_grad(): 
    output = model(img_tensor, captions=[TEXT_PROMPT])  

print(output)

# target_layers = [model.backbone[0].layers[-1].blocks[-1].norm1]
# target_layers = [model.transformer.decoder.layers[-1].cross_attn]
target_layers = [model.transformer.decoder.layers[-1].norm3]
# target_layers = [model.transformer.decoder.norm]
# target_layers = [model.transformer.encoder.fusion_layers[-1]]
print(target_layers)
# target_layers = [model.transformer.decoder.ref_point_head.layers[1]]
# target_layers = [model.norm]
# target_layers = [model._model.layer4]
# cam_algorithm = GradCAM(model=model, target_layers=target_layers)
# ViT
# def reshape_transform(tensor, height=40, width=40):
#     result = tensor[:, 1 :  , :].reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result
# SwinT
# def reshape_transform(tensor, height=1, width=1):
#     result = tensor.reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result
# tRANSFORMER in Grounding Dino
# for [model.transformer.decoder.layers[-1].cross_attn]
# def reshape_transform(tensor, height=30, width=30):
#     """
#     Args:
#         tensor: 形状应为 [1, 900, 256]
#         height, width: 目标分辨率 (height * width 必须等于 900)
#     Returns:
#         [1, 256, height, width]
#     """
#     batch_size, num_queries, d_model = tensor.shape
#     assert num_queries == height * width, f"{num_queries} != {height * width}"
    
#     # 重塑为 [batch_size, height, width, d_model]
#     result = tensor.reshape(batch_size, height, width, d_model)
#     # 调整通道顺序为 [batch_size, d_model, height, width]
#     result = result.permute(0, 3, 1, 2)
#     return result
# for [model.transformer.decoder.layers[-1].norm3]
def reshape_transform(tensor, height=30, width=30):
    """
    Args:
        tensor: norm3 的输出，形状应为 [num_queries, batch_size, d_model] (如 [900, 1, 256])
        height, width: 目标空间分辨率（需满足 height * width = num_queries）
    Returns:
        重塑后的张量 [batch_size, d_model, height, width]
    """
    num_queries, batch_size, d_model = tensor.shape
    assert height * width == num_queries, f"{height*width} != {num_queries}"

    # 重塑为 [batch_size, height, width, d_model]
    result = tensor.permute(1, 0, 2).reshape(batch_size, height, width, d_model)
    # 调整通道顺序为 [batch_size, d_model, height, width]
    result = result.permute(0, 3, 1, 2)
    return result
# def reshape_transform(tensor, height=30, width=30):
#     """
#     Args:
#         tensor: Decoder cross-attention输出 [900, batch_size, 256]
#         height, width: 目标空间分辨率 (需满足 height * width = num_queries)
#     Returns:
#         [batch_size, 256, height, width]
#     """
#     batch_size = tensor.size(1)
#     d_model = tensor.size(2)
#     # 确保 num_queries 能平铺到 height * width
#     assert height * width == tensor.size(0), f"{height*width} != {tensor.size(0)}"
    
#     # 重塑为 [batch, height, width, d_model]
#     result = tensor.permute(1, 0, 2).reshape(batch_size, height, width, d_model)
#     # 调整通道顺序为 [batch, d_model, height, width]
#     result = result.permute(0, 3, 1, 2)
#     return result
# def reshape_transform(tensor, height=30, width=30):
#     # tensor shape: [batch_size, num_queries, embed_dim] (e.g., [1, 900, 256])
#     # 假设 num_queries = height * width (需满足 900 == 40*40=1600？不成立！需调整)
#     # 若 num_queries != height*width，需插值或截断
#     if tensor.size(1) != height * width:
#         # 方案1：截断多余的 queries
#         tensor = tensor[:, :height*width, :]
#         # 方案2：插值（需改为2D插值）
    
#     # 重塑为 [batch, height, width, dim]
#     result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
#     # 调整通道顺序为 [batch, dim, height, width]
#     result = result.permute(0, 3, 1, 2)  # 等同于 .transpose(2,3).transpose(1,2)
#     return result
# def reshape_transform(tensor):
#     # tensor shape: [1, num_queries, dim]
#     num_queries = tensor.size(1)
    
#     # 假设 ref_points 是归一化的坐标 [0,1] (需从模型获取)
#     ref_points = model.transformer.decoder.ref_point_head(tensor)  # 伪代码，需适配实际实现
#     original_height = 900
#     original_width = 256

#     # 生成空的特征图
#     feature_map = torch.zeros(1, tensor.size(2), original_height, original_width)
    
#     # 将查询特征插值到对应位置
#     for i in range(num_queries):
#         x, y = ref_points[0, i]  # 假设 ref_points 是 [1, num_queries, 2]
#         x_pixel = int(x * original_width)
#         y_pixel = int(y * original_height)
#         if 0 <= x_pixel < original_width and 0 <= y_pixel < original_height:
#             feature_map[0, :, y_pixel, x_pixel] = tensor[0, i, :]
    
#     return feature_map
cam_algorithm = KPCA_CAM(model=model, target_layers=target_layers,reshape_transform=reshape_transform)
# cam_algorithm = GradCAM(model=model, target_layers=target_layers,reshape_transform=reshape_transform)
# GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)


grayscale_cams = cam_algorithm(input_tensor=img_tensor) 
targets = None
grayscale_cam = grayscale_cams[0, :]
img = np.float32(img) / 255
img = cv2.resize(img,(800,800))
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


cam_output_path = os.path.join("/home/pengxi/Documents/ATM/output_test", 'test_cam.jpg')

cv2.imwrite(cam_output_path, cam_image)