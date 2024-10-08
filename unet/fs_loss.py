import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import ssim
# Parts of these codes are from: https://github.com/Linfeng-Tang/SeAFusion

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis
        # x_in_max = torch.max(image_y, image_ir)
        # loss_ints = F.l1_loss(generate_img, x_in_max)
        loss_ssim = ssim(generate_img, image_y, 11) + ssim(generate_img, image_ir, 11)
        image_a = (image_y+image_ir)/2
        loss_in = self.mse_criterion(generate_img, image_a) # + self.mse_criterion(generate_img, image_ir)

        # loss = color_angle_loss + loss_in
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)

        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)
        # loss_grad = F.l1_loss(generate_img_grad, y_grad) +  F.l1_loss(generate_img_grad, ir_grad)
        #
        return loss_ssim, loss_in, torch.square(loss_grad)
