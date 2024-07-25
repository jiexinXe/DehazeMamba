# import torch
# import torchvision.models as models
#
# # 调整学习率
# def adjust_learning_rate(optimizer, decay_rate=0.9):
#     # 遍历优化器的参数组，并将学习率乘以衰减率
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = param_group['lr'] * decay_rate
#         print('learning rate: %.2e' % param_group['lr'])
#
# # 计算和存储平均值
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self, max_count=100):
#         # 初始化计数器，最大计数默认为100
#         self.reset(max_count)
#
#     def reset(self, max_count):
#         # 重置所有参数
#         self.val = 0
#         self.avg = 0
#         self.data_container = []
#         self.max_count = max_count
#
#     def update(self, val):
#         # 更新当前值和平均值
#         self.val = val
#         if len(self.data_container) < self.max_count:
#             # 如果容器未满，则直接添加新值
#             self.data_container.append(val)
#             self.avg = sum(self.data_container) * 1.0 / len(self.data_container)
#         else:
#             # 如果容器已满，则移除最早的值并添加新值
#             self.data_container.pop(0)
#             self.data_container.append(val)
#             self.avg = sum(self.data_container) * 1.0 / self.max_count
#
# # 保存模型和优化器状态的函数
# def save_model_optimizer_history(model, optimizer, filepath, device):
#     state = {
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     torch.save(state, filepath)
#
# # 加载模型状态
# def load_model(model, filepath, device):
#     print("loading model")
#     state_dict = torch.load(filepath, map_location=device)['state_dict']
#     model.load_state_dict(state_dict)
#     return model
#
# # 加载优化器状态
# def load_optimizer(optimizer, filepath, device):
#     print("loading optimizer")
#     state = torch.load(filepath)
#     optimizer.load_state_dict(state['optimizer'])
#     return optimizer
#
# # 获取VGG16模型框架的函数
# def getVgg_frame(pretrained=True):
#     if pretrained == False:
#         model = models.vgg16(pretrained=False, num_classes=101)
#     else:
#         model_np = models.vgg16(pretrained=False, num_classes=101)
#         model_p = models.vgg16(pretrained=True)
#
#         pretrained_dict = model_p.state_dict()
#         unload_model_dict = model_np.state_dict()
#         # 过滤掉形状不匹配的参数
#         load_model = {k: v for k, v in pretrained_dict.items() if
#                       (k in unload_model_dict and pretrained_dict[k].shape == unload_model_dict[k].shape)}
#
#         print('load_model:')
#         for dict_inx, (k, v) in enumerate(load_model.items()):
#             print(dict_inx, k, v.shape)
#         unload_model_dict.update(load_model)
#         model_np.load_state_dict(unload_model_dict)
#         model = model_np
#
#     return model
#
# # 加载预训练模型的函数
# def load_pretrained_model(model_path):
#     model = getVgg_frame(pretrained=False)
#     pretrained_dict = torch.load(model_path)
#     unload_model_dict = model.state_dict()
#
#     print('unload_model_dict:')
#     for dict_inx, (k, v) in enumerate(unload_model_dict.items()):
#         print(dict_inx, k, v.shape)
#
#     load_model = {}
#     for k, v in pretrained_dict.items():
#         load_k = k.split('.')[1:]
#         load_k = '.'.join(load_k)
#         if pretrained_dict[k].shape == unload_model_dict[load_k].shape:
#             load_model[load_k] = pretrained_dict[k]
#     print('len of load_model:', len(load_model))
#     print('pretrained_dict:')
#     for dict_inx, (k, v) in enumerate(pretrained_dict.items()):
#         print(dict_inx, k, v.shape)
#     unload_model_dict.update(load_model)
#     model.load_state_dict(unload_model_dict)
#
#     return model

import torch
import torchvision.models as models

# 调整学习率
def adjust_learning_rate(optimizer, decay_rate=0.9):
    # 遍历优化器的参数组，并将学习率乘以衰减率
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print('learning rate: %.2e' % param_group['lr'])

# 计算和存储平均值
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, max_count=100):
        # 初始化计数器，最大计数默认为100
        self.reset(max_count)

    def reset(self, max_count):
        # 重置所有参数
        self.val = 0
        self.avg = 0
        self.data_container = []
        self.max_count = max_count

    def update(self, val):
        # 更新当前值和平均值
        self.val = val
        if len(self.data_container) < self.max_count:
            # 如果容器未满，则直接添加新值
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / len(self.data_container)
        else:
            # 如果容器已满，则移除最早的值并添加新值
            self.data_container.pop(0)
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / self.max_count

# 保存模型和优化器状态的函数
def save_model_optimizer_history(model, optimizer, filepath, device):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)

# 加载模型状态的函数
def load_model(model, filepath, device):
    print("loading model")
    state_dict = torch.load(filepath, map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    return model

# 加载优化器状态的函数
def load_optimizer(optimizer, filepath, device):
    print("loading optimizer")
    state = torch.load(filepath)
    optimizer.load_state_dict(state['optimizer'])
    return optimizer

# 获取VGG16模型框架的函数（如果需要）
def getVgg_frame(pretrained=True):
    if pretrained == False:
        model = models.vgg16(pretrained=False, num_classes=101)
    else:
        model_np = models.vgg16(pretrained=False, num_classes=101)
        model_p = models.vgg16(pretrained=True)

        pretrained_dict = model_p.state_dict()
        unload_model_dict = model_np.state_dict()
        # 过滤掉形状不匹配的参数
        load_model = {k: v for k, v in pretrained_dict.items() if
                      (k in unload_model_dict and pretrained_dict[k].shape == unload_model_dict[k].shape)}

        print('load_model:')
        for dict_inx, (k, v) in enumerate(load_model.items()):
            print(dict_inx, k, v.shape)
        unload_model_dict.update(load_model)
        model_np.load_state_dict(unload_model_dict)
        model = model_np

    return model

# 加载预训练模型的函数（如果需要）
def load_pretrained_model(model_path):
    model = getVgg_frame(pretrained=False)
    pretrained_dict = torch.load(model_path)
    unload_model_dict = model.state_dict()

    print('unload_model_dict:')
    for dict_inx, (k, v) in enumerate(unload_model_dict.items()):
        print(dict_inx, k, v.shape)

    load_model = {}
    for k, v in pretrained_dict.items():
        load_k = k.split('.')[1:]
        load_k = '.'.join(load_k)
        if pretrained_dict[k].shape == unload_model_dict[load_k].shape:
            load_model[load_k] = pretrained_dict[k]
    print('len of load_model:', len(load_model))
    print('pretrained_dict:')
    for dict_inx, (k, v) in enumerate(pretrained_dict.items()):
        print(dict_inx, k, v.shape)
    unload_model_dict.update(load_model)
    model.load_state_dict(unload_model_dict)

    return model
