import torch
from os import path, listdir
import argparse
import ResNet
import torch.nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from math import sqrt, floor, ceil

parser = argparse.ArgumentParser(description="A demo for the trained model")
parser.add_argument("--exp-id", default='1', help="id of the experiment")
args = parser.parse_args()

model_dir = path.join(path.dirname(__file__), "model", args.exp_id)
paths = listdir(model_dir)
model_name = ""
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

for tmp in paths:
    if tmp.rfind("ResNet") != -1:
        model_path = path.join(model_dir, tmp)
        model_name = tmp.split('.')[0].lower()
        break

print(model_path, model_name)
model = ResNet.__dict__[model_name]()
model.load_state_dict(torch.load(model_path, map_location = 'cuda:0'))


a = torch.rand(3,4,4,5)
_a, m = torch.max(a, 0)


def predict(img):
    img_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    output = model(img)
    _, predicted = torch.max(output, dim=1)
    return classes[predicted]

if __name__ == "__main__":
    test_dir = path.join(path.dirname(__file__), 'test')
    test_imgs = listdir(test_dir)
    for img_name in test_imgs:
        if img_name.rfind('result') != -1:
            continue
        img_path = path.join(test_dir, img_name)
        img = Image.open(img_path)
        result = "\n".join([img_name, predict(img)])
        plt_img = plt.imread(img_path)
        plt.subplot(floor(sqrt(len(test_imgs))), ceil(sqrt(len(test_imgs))), test_imgs.index(img_name) + 1)
        plt.imshow(plt_img)
        plt.title(result)
        plt.tight_layout()

    plt.savefig(path.join(test_dir, 'result{}.jpg'.format(args.exp_id)))    
    plt.show()
    