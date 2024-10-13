import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM

def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153), 
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)], 
                     dtype=np.uint8) 
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    #label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy

def generate_label(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))
                
    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)	

    return label_batch

def generate_label_plain(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        #pred = pred.reshape((1, 512, 512))
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        label_batch.append(p.numpy())
                
    label_batch = np.array(label_batch)

    return label_batch

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def ms_ssim_loss(input, target):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Ensure input and target have the same spatial dimensions
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    
    # Apply softmax to get class probabilities
    input = F.softmax(input, dim=1)  # shape: [batch_size, num_classes, height, width]

    # You could use one-hot encoding to convert target to one-hot for multiple classes
    target_one_hot = F.one_hot(target, num_classes=c).permute(0, 3, 1, 2).float()  # shape: [batch_size, num_classes, height, width]

    # Use MS-SSIM as the loss function (using probabilities directly without argmax)
    ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=c)
    loss = 1 - ms_ssim_module(input, target_one_hot)
    
    return loss

def dice_loss_fn(input, target, smooth=1e-5):
    input = torch.sigmoid(input)  # 如果是多类别分割，使用 softmax
    target = target.float()       # 确保 target 是浮点数
    input_flat = input.view(-1)   # 将输入展平
    target_flat = target.view(-1) # 将目标展平

    intersection = (input_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

    return 1 - dice_score

# Dice Loss for multi-class segmentation
def dice_loss_fn(input, target, smooth=1e-5):
    # 使用 softmax 将 input 转换为每个类别的概率分布
    input = F.softmax(input, dim=1)

    # 提取类别数量
    num_classes = input.size(1)

    # 初始化总的 Dice Loss
    dice_loss = 0

    for i in range(num_classes):
        # 提取第 i 类的预测和目标
        input_flat = input[:, i, :, :].contiguous().view(-1)
        target_flat = (target == i).float().contiguous().view(-1)

        # 计算 Dice 系数
        intersection = (input_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        
        # 累加每个类别的 Dice Loss
        dice_loss += 1 - dice_score

    # 取每个类别 Dice Loss 的平均值
    return dice_loss / num_classes

# 交叉熵 + Dice Loss 组合
def cross_entropy_dice_loss(input, target, weight=None, size_average=True, dice_weight=0.5, ce_weight=0.5):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # 如果输入和目标尺寸不一致，使用插值调整输入尺寸
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    # 交叉熵损失
    ce_loss = F.cross_entropy(input, target, weight=weight, ignore_index=250)

    # Dice Loss
    dice_loss = dice_loss_fn(input, target)

    # 加权组合损失
    combined_loss = ce_weight * ce_loss + dice_weight * dice_loss
    return combined_loss
