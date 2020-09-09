import torch
import cv2
from config import DefaultConfig
from models.mobilenetv3 import MobileNetV3_Small
from torchvision import transforms
import numpy as np


def cvImgToTensor(cvImg):
    image = cvImg.copy()
    height, width, channel = image.shape
    ratio = 224/min(height, width)
    image = cv2.resize(image, None, fx=ratio, fy=ratio)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if image is not None:
        image = image[:, :, (2, 1, 0)]
        image = transform(image)
        image.unsqueeze_(0)

    return image

DataSetInfo = torch.load(DefaultConfig.DataSetInfoPath)
index_to_class = DataSetInfo['index_to_class']
index_to_group = DataSetInfo['index_to_group']
MyModel = MobileNetV3_Small(DataSetInfo["class_num"])
device = torch.device('cuda') if DefaultConfig.InferWithGPU else torch.device('cpu')
MyModel.load_state_dict(torch.load(DefaultConfig.CkptPath,map_location=torch.device('cpu'))['state_dict'])
if DefaultConfig.InferWithGPU:
    MyModel.cuda()
else:
    MyModel.cpu()
MyModel.eval()



def inference(cvImg):
    image = cvImgToTensor(cvImg)
    image = image.to(device=device)
    result = MyModel(image)
    _, predicted = torch.max(result, 1)
    predicted = predicted.item()
    print('物体类别：', index_to_class[predicted])
    print('该物体属于', index_to_group[predicted])


if __name__ == '__main__':
    import time
    image = cv2.imread('images/test.jpg')
    t0 = time.time()
    inference(image)
    print(time.time()-t0)
    print(DataSetInfo['group_count']['train'])