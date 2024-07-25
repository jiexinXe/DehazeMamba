import argparse
from torch.autograd import Variable
import os
import random
import numpy as np
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch.cuda.amp import GradScaler, autocast

from model import MambaSSM  # 引入 MambaSSM 模型
from utils import *  # 导入工具函数
import dataloader

# 全局变量和参数
dtype = torch.float16  # 切换为 float16
USE_GPU = True
EPOCH = 12
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4  # 梯度累积步数
print_every = int(50 / BATCH_SIZE * 64)
Load_model = False
NAME = 'MambaDehaze'

NET_LR = 1e-4
FC_LR = 1e-4
OPTIMIZER = 'adam'
LR_DECAY_EPOCH = [] if OPTIMIZER == 'adam' else [15, 30]
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
DECAY_RATE = 0.1
RANDOM_SEED = 8273

START_EPOCH = 0

loader_train = None
loader_val = None

# 设备
def set_device():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('using device:', device)
    return device

# 训练函数
def train(model, optimizer, criterion, device, epochs=1, start=0):
    global loader_train, loader_val
    model = model.to(device=device)
    scaler = GradScaler()  # 使用 PyTorch 的混合精度训练，防止显存不够

    if not os.path.isdir(NAME + '_save'):
        os.mkdir(NAME + '_save')

    for e in range(start, epochs):
        print('epoch: %d' % e)

        losses = AverageMeter()
        batch_time = AverageMeter()

        if e in LR_DECAY_EPOCH:
            adjust_learning_rate(optimizer, decay_rate=DECAY_RATE)

        end_time = time.time()
        optimizer.zero_grad()  # 初始化优化器梯度
        for t, (img_original, img_haze) in enumerate(loader_train):
            model.train()  # 设置模型为训练模式
            img_original = img_original.to(device=device).half()
            img_haze = img_haze.to(device=device).half()

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            with autocast():
                output = model(img_haze)
                loss = criterion(output, img_original)
                loss = loss / ACCUMULATION_STEPS  # 梯度累积步数

            scaler.scale(loss).backward()

            if (t + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses.update(loss.item() * ACCUMULATION_STEPS)  # 累积的损失值乘回累积步数

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if t % print_every == 0:
                print('Train: [%d/%d]\t'
                      'Time %.3f (%.3f)\t'
                      'Loss %.4f (%.4f)'
                      % (t, len(loader_train), batch_time.val, batch_time.avg, losses.val, losses.avg))

        test_epoch(model, criterion, loader_val, device, e, epochs)

        save_model_optimizer_history(model, optimizer, filepath=NAME + '_save' + '/epoch%d.pth' % (e), device=device)

# 将张量转换为图像并显示
def tensor_showImg(a):
    a = a.cpu()
    image_PIL = transforms.ToPILImage()(a)
    image_PIL.show()

# 测试函数
def test_epoch(model, criterion, loader_val, device, epoch, end_epoch, verbo=True):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    total = 0
    correct = 0
    end_time = time.time()
    for batch_idx, (img_original, img_haze) in enumerate(loader_val):
        img_original = img_original.to(device=device, dtype=dtype)
        img_haze = img_haze.to(device=device, dtype=dtype)
        img_original, img_haze = Variable(img_original), Variable(img_haze)

        with autocast():
            output = model(img_haze)
            output = output.to(device=device)

            loss = criterion(output, img_original)

        losses.update(loss.cpu().data.numpy())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_idx % 20 == 0 and verbo == True:
            print('Test: [%d/%d]\t'
                  'Time %.3f (%.3f)\t'
                  'Loss %.4f (%.4f)' % (batch_idx, len(loader_val),
                                       batch_time.val, batch_time.avg,
                                       losses.val, losses.avg))

    print('Test: [%d/%d]' % (epoch, end_epoch))

# 检查参数是否属于全连接层
def is_fc(para_name):
    split_name = para_name.split('.')
    if len(split_name) < 3:
        return False
    if split_name[-3] == 'classifier':
        return True
    else:
        return False

# 设置网络的学习率
def net_lr(model, fc_lr, lr):
    params = []
    for keys, param_value in model.named_parameters():
        if (is_fc(keys)):
            params += [{'params': [param_value], 'lr': fc_lr}]
        else:
            params += [{'params': [param_value], 'lr': lr}]
    return params

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 训练主函数
def train_main(args):
    global loader_train, loader_val

    # 加载训练和验证数据集
    train_dataset = dataloader.dehazing_loader(args.original_pic_root, args.haze_pic_root)
    val_dataset = dataloader.dehazing_loader(args.original_pic_root, args.haze_pic_root, mode="val")
    loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = set_device()
    setup_seed(RANDOM_SEED)  # 设置随机种子

    model = MambaSSM()  # 使用Mamba模型
    # model = nn.DataParallel(model)  # 多GPU训练

    criterion = nn.MSELoss()

    params = net_lr(model, FC_LR, NET_LR)

    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999), weight_decay=0, eps=1e-08)
    else:
        optimizer = torch.optim.SGD(params, momentum=MOMENTUM, nesterov=True,
                                    weight_decay=WEIGHT_DECAY)

    print(model)
    start_epoch = 0
    if Load_model:
        start_epoch = 25
        filepath = 'load_model_path'
        model = load_model(model, filepath, device=device)
        model = model.to(device=device)
        optimizer = load_optimizer(optimizer, filepath, device=device)

    train(model, optimizer, criterion, device=device, epochs=EPOCH, start=start_epoch)

if __name__ == '__main__':
    print("RANDOM SEED:", RANDOM_SEED)

    parser = argparse.ArgumentParser(description='Dehaze image with Mamba')
    parser.add_argument('--original_pic_root', type=str, help='path of train image npy', default='data/images/')
    parser.add_argument('--haze_pic_root', type=str, help='path of test image npy', default='data/data/')

    args = parser.parse_args()

    train_main(args)
