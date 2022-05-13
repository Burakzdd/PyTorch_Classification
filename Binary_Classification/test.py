from cv2 import COLOR_BAYER_BG2GRAY
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import cv2
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model_path = "./model.pth"
model = torchvision.models.resnet34(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_filters = model.fc.in_features
model.fc = torch.nn.Linear(in_features=num_filters, out_features=1)

model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

images_path = "/home/burakzdd/Desktop/work/torch_classification/datasets/test/"
images = os.listdir(images_path)


for img in images:
    image = Image.open(images_path+img)
    image = image.convert("RGB")
    image = transform(image)
    image = image.to(torch.float)
    image = image.to(device)
    image = torch.unsqueeze(image, 0)
    img = cv2.imread(images_path+img)
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).cpu().detach().numpy()[0][0]
        
    if probability < 0.5:
        image = cv2.putText(img, 'CAT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        image = cv2.putText(img, 'DOG', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("result",img)
    cv2.waitKey()
    
cv2.destroyAllWindows()
