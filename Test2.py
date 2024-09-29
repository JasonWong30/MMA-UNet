# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import torch
from torch.utils.data import DataLoader
from util.loader import Fusion_dataset
from tqdm import tqdm
import torch.utils.data
import torch.nn.functional
import time
import numpy as np
import torchvision.transforms as transforms
from unet.unetv2 import Unet as net
from unet.unet_model import UNet as UNet_vi
from unet.unet_ir import UNet as UNet_ir

def main(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model = net()
    model.eval()
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=True)

    print('fusionmodel load done!')
    test_dataset = Fusion_dataset(args.data_dir, 'test_MSRS_irvi', transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,
        shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=False)
    time_list = []
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (ir, vis, name) in enumerate(test_bar):
            start = time.time()
            vis = vis.to(device)
            ir = ir.to(device)

            # _, _, h, w = vis.shape
            model_ir = eval('UNet_ir')(3, 3)
            model_ir.to(device)
            checkpoint = torch.load(args.model_ir, map_location='cpu')
            model_ir.load_state_dict(checkpoint['model'], strict=True)

            model_vi = eval('UNet_vi')(3, 3)
            model_vi.to(device)
            checkpoint = torch.load(args.model_vi, map_location='cpu')
            model_vi.load_state_dict(checkpoint['model'], strict=True)

            _, fea_vis = model_vi(vis)
            _, fea_ir = model_ir(ir, fea_vis)
            #fea_ir, fea_vis, ir, vis
            fused_img, _  = model(fea_ir, fea_vis, ir, vis)
            ones = torch.ones_like(fused_img)
            zeros = torch.zeros_like(fused_img)
            fused_img = torch.where(fused_img > ones, ones, fused_img)
            fused_img = torch.where(fused_img < zeros, zeros, fused_img)

            fused_image = fused_img.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            # size = (w, h)

            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )

            # fused_image = np.uint8(255.0 * fused_image)
            end = time.time()
            time_list.append(end - start)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                # image = cv2.resize(image, size)
                image = np.uint8(255.0 * image)
                save_path = os.path.join(args.save_dir, name[k])
                image.save(save_path)

    print(time_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MMA-UNet with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./output_dir/checkpoint-100.pth')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./UNet/')
    parser.add_argument('--batch_size', '-B', type=int, default=5)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)

    parser.add_argument('--model_vi', default='./saved_checkpoint/checkpoint-400-vi.pth', type=str, help='GPUs used for training')
    parser.add_argument('--model_ir', default='./saved_checkpoint/checkpoint-400-ir.pth', type=str, help='GPUs used for training')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    main(args)


