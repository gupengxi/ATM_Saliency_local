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
model = load_model("grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", "gdino_checkpoints/groundingdino_swint_ogc.pth")
IMAGE_PATH = "data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/images/demo_0/agentview_0.png"
# TEXT_PROMPT = "the black bowl. the white plate."
TEXT_PROMPT = "circle"
print(model)
# 加载并预处理图像
# img_path = 'images_test/demo_0/agentview_0.png'
img_path = "images_test/demo_0/eye_in_hand_0.png"
img = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
# img_t = model_transforms(img)
# 定义预处理流程：调整大小、裁剪、归一化等
preprocess = transforms.Compose([
    transforms.Resize(650),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 对图像进行预处理
img_tensor = preprocess(img).unsqueeze(0)  
# img_tensor = img_t.unsqueeze(0)
with torch.no_grad(): 
    output = model(img_tensor, captions=[TEXT_PROMPT])  

print(output)

target_layers = [model.backbone[0].layers[-1].blocks[-1].norm1]
# target_layers = [model.norm]
# target_layers = [model._model.layer4]
# cam_algorithm = GradCAM(model=model, target_layers=target_layers)
# def reshape_transform(tensor, height=40, width=40):
#     result = tensor[:, 1 :  , :].reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result
def reshape_transform(tensor, height=21, width=21):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
cam_algorithm = KPCA_CAM(model=model, target_layers=target_layers,reshape_transform=reshape_transform)
# cam_algorithm = GradCAM(model=model, target_layers=target_layers,reshape_transform=reshape_transform)
# GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)


grayscale_cams = cam_algorithm(input_tensor=img_tensor) 
targets = None
grayscale_cam = grayscale_cams[0, :]
img = np.float32(img) / 255
img = cv2.resize(img,(650,650))
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


cam_output_path = os.path.join("/home/pengxi/Documents/ATM/output_test", 'test_cam.jpg')

cv2.imwrite(cam_output_path, cam_image)