# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
from torch.utils.data import DataLoader
from util.loader import Fusion_dataset
from unet.unetv2 import Unet as FusionNet ##abla
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from unet.unet_model import UNet as UNet_vi
from unet.unet_ir import UNet as UNet_ir #abla

def tensor2img(img, is_norm=True):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  if is_norm:
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)

def save_img_single(img, name, size, is_norm=True):
  img = tensor2img(img, is_norm=True)
  img = Image.fromarray(img[:,:,0])
  img = img.resize(size)
  img.save(name)

# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main(data_dir, save_dir, fusion_model_path):

    fusionmodel = FusionNet(False)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel.load_state_dict(torch.load(fusion_model_path)['model'])
    fusionmodel = fusionmodel.to(device)
    fusionmodel.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    model_ir = UNet_ir(3, 3)
    checkpoint = torch.load(args.model_ir, map_location='cpu')
    model_ir.load_state_dict(checkpoint['model'], strict=True)
    model_ir.to(device)
    model_ir.eval()

    model_vi = UNet_vi(3, 3)
    checkpoint = torch.load(args.model_vi, map_location='cpu')
    model_vi.load_state_dict(checkpoint['model'], strict=True)
    model_vi.to(device)
    model_vi.eval()

    test_dataset = Fusion_dataset(data_dir, 'test_M3FD_irvi', transform_test)
    test_loader = DataLoader(dataset=test_dataset,
        batch_size=args.batch_size,shuffle=False,
        num_workers=args.num_workers,pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_ir, img_vis, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            _, _, H, W = img_vis.shape
            size = (W, H)
            _, fea_vis = model_vi(img_vis)
            _, fea_ir = model_ir(img_ir, fea_vis)

            logits, _ = fusionmodel(fea_ir, fea_vis, img_ir, img_vis)

            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(logits[k, ::], save_path, size)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MMA-UNet with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./output_dir/checkpoint-100.pth') #460
    ## dataset
    parser.add_argument('--data_dir', '-data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./M3FD')#M3FD MSRS
    parser.add_argument('--model_vi', default='./saved_checkpoint/checkpoint-400-vi.pth', type=str,
                        help='GPUs used for training')
    parser.add_argument('--model_ir', default='./saved_checkpoint/checkpoint-400-ir.pth', type=str,
                        help='GPUs used for training') ##abla
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    main(data_dir=args.data_dir, save_dir=args.save_dir, fusion_model_path=args.model_path)
