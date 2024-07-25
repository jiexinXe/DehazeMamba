import os
import torch
from model import MambaSSM  # 更新导入MambaSSM
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import time

# 测试单张图像的去雾
def test_on_img(state_dict_path, haze_img_path):
    state_dict = torch.load(state_dict_path)['state_dict']
    model = MambaSSM()  # 使用MambaSSM
    model.load_state_dict(state_dict)
    model.eval()  # 设置模型为评估模式

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    haze_img = Image.open(haze_img_path).convert('RGB')

    # 保存原始图像尺寸
    original_size = haze_img.size

    # 调整图像大小到训练时的尺寸
    transform = transforms.Compose([
        transforms.Resize((480, 640), interpolation=Image.LANCZOS),
        transforms.ToTensor()
    ])
    haze_img_resized = transform(haze_img)
    haze_img_resized = haze_img_resized.unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        result_img = model(haze_img_resized)
    end_time = time.time()

    elapsed_time = end_time - start_time

    result_img = result_img.squeeze(0).cpu()

    return result_img, elapsed_time, original_size


if __name__ == '__main__':
    state_dict_path = 'MambaDehaze_save/epoch11.pth'
    input_dir = '../../dataset/Haze4K/test/haze'
    output_dir = 'dataset_output/Haze4K'
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            haze_img_path = os.path.join(input_dir, img_name)

            result, elapsed_time, original_size = test_on_img(state_dict_path, haze_img_path)

            # 调整结果图像的尺寸以匹配原始图像
            result_pil = transforms.ToPILImage()(result)
            result_resized = result_pil.resize(original_size, Image.LANCZOS)
            result_resized = transforms.ToTensor()(result_resized)

            output_img_path = os.path.join(output_dir, img_name)
            torchvision.utils.save_image(result_resized, output_img_path)

            print(f'Processed {img_name} - Elapsed time: {elapsed_time:.4f} seconds')
