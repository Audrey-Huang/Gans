from fgsm import attack
import os
from PIL import Image
from cleverhans.model import Model
from InceptionModel import InceptionModel
dataset_dir="./ccpd_rotate"
image_filenames=[os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]

def recovert(image):
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(256, 256, 256))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)
    image_data=background.resize((299,299))#缩放
    return image_data

if __name__ == '__main__':
    a=InceptionModel(Model)
    A=attack(a)
    for filename in image_filenames:
        attackimg=A.run(filename)
        print(attackimg)