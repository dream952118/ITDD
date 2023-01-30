from __future__ import print_function, division

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


plt.ion()   # interactive mode

# 模型存儲路徑
model_save_path = './output/first_10/model_final.pth'

# ------------------------ 加載數據 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定義預訓練變換
preprocess_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class_names = [
	'I-Others','E-M1-Al Residue',
	'I-Dust','T-ITO1-Hole','T-M2-Fiber','I-Sand Defect',
	'I-Glass Scratch','E-AS-Residue','I-Oil Like','T-M1-Fiber',
	'E-ITO1-Hole','P-ITO1-Residue','E-AS-BPADJ','I-Laser Repair',
	'P-AS-NO','T-ITO1-Residue','I-M2-Crack','E-M2-PR Residue',
	'E-M2-Short','T-Brush defect','T-AS-Residue','T-AS-Particle Small',
	'E-M2-Residue','T-M2-Particle','P-M2-Residue','P-AS-Residue',
	'P-M1-Residue','T-M1-Particle','P-M2-Short','T-AS-SiN Hole',
	'P-AS-BPADJ','P-M2-Open','P-M1-Short',
]  # 這個順序很重要，要和訓練時候的類名順序一致

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------ 載入模型並且訓練 --------------------------- #
model = torch.load(model_save_path)
model.eval()
# print(model)

image_PIL = Image.open('./datasets/VOC2007/JPEGImages/I-Others/I-Others_PU90H13U00-09_3_NGID003500190.jpg')
# 
image_tensor = preprocess_transform(image_PIL)
# 以下語句等效於 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 沒有這句話會報錯
image_tensor = image_tensor.to(device)

out = model(image_tensor)
# 得到預測結果，並且從大到小排序
_, indices = torch.sort(out, descending=True)
# 返回每個預測值的百分數
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:5]])