import os
import argparse
from tqdm import tqdm

import cv2
import torch
import numpy as np
from models.ooal import Net as model

from ooal_utils.utils.viz import viz_pred_test
from ooal_utils.utils.util import set_seed, process_gt, normalize_map
from ooal_utils.utils.evaluation import cal_kl, cal_sim, cal_nss
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, FullGrad, EigenCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, GradCAMElementWise, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np
import cv2
import os
from PIL import Image
from ooal_data.data.agd20k_ego import TestData, SEEN_AFF, UNSEEN_AFF
parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='./dataset/')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./save_preds')
##  image
parser.add_argument('--divide', type=str, default='Seen')
parser.add_argument('--crop_size', type=int, default=630)
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
print(model)
img_path = 'images_test/demo_0/agentview_0.png'
# img_path = "download.png"
img = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
preprocess = transforms.Compose([
    transforms.Resize(630),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),  # 将图像转换为张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])
img_t = preprocess(img).unsqueeze(0)
# 定义预处理流程：调整大小、裁剪、归一化等
# UNSEEN_AFF = ["carry", "catch", "cut", "cut_with", 'drink_with',
#              "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
#              "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
#              "swing", "take_photo", "throw", "type_on", "wash"]
ego_pred = model(img_t.cuda(), gt_aff=[args.class_names.index("open"),args.class_names.index("middle_drawer")])
ego_pred = np.array(ego_pred.squeeze().data.cpu())
ego_pred = normalize_map(ego_pred, args.crop_size)
print(ego_pred)
from ooal_utils.utils.util import overlay_mask
import matplotlib.pyplot as plt
import cv2


def viz_pred_test_modified(args, image, ego_pred, img_name, epoch=None):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    # gt = Image.fromarray(GT_mask)
    # gt_result = overlay_mask(img, gt, alpha=0.5)
    # aff_str = aff_list[aff_label.item()]

    os.makedirs(os.path.join(args.save_path, 'viz_gray'), exist_ok=True)
    gray_name = os.path.join(args.save_path, 'viz_gray', img_name + '.jpg')
    cv2.imwrite(gray_name, ego_pred * 255)

    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(img, ego_pred, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()

    ax[0].imshow(ego_pred)
    # ax[0].set_title(aff_str)
    # ax[1].imshow(gt_result)
    # ax[1].set_title('GT')

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "iter" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()

viz_pred_test_modified(args, img_t, ego_pred, img_name="test_img")