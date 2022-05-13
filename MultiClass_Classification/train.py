import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        #transforms.colorJitter(brightnes=.5, hue=.3) # görüntünün parlaklıki doygunluk, ve diğer özelliklerini rastgele olarak değiştirir.
        #transforms.GaussianBlur(kernel_size(5,9), sigma=(0.1, 5)) # görüntü üzerinde gaussv bulanıklığı dönüşümü gerçekleştirir.
        #transforms.RandomPerspective(distortion_scale=0.6,p=1.0) #görüntü üzerinde rastgele olarak perspektif dönüşümü gerçekleştirir.
        #transforms.RandomRotation(degrees=(0,180)) #görüntüyü belirlenen açılar arasındaki rastgele bir açıyla döndürür.
        #transforms.RandomAffine(degress=(30,70),translate=(0.1,0.3),scale=(0.5,0.75)) #bir görüntü üzserinde rastgele afin uzayında bir dönüşüm gerçekleştrir.
        #transforms.RandomCrop(size=(128,128)) bir görüntüyü rastgele bir konumda kırpar.
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,  0.5))
    ]
)
batch_size = 8
epochs = 10
class_number = 3
train_folder = "/home/burakzdd/Desktop/work/torch_classification/datasets/tas_kagit_makas/"

train_data = torchvision.datasets.ImageFolder(root = train_folder, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)

model = torchvision.models.resnet34(pretrained=True)

for param in model.parameters():
    param.requires_grad = False 

num_filters = model.fc.in_features
model.fc = torch.nn.Linear(in_features= num_filters, out_features= class_number)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    running_loss = []
    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    
    for batch_index, (data, target) in loop:
        image, target = data.to(device), target.to(device) 
        
        output = model(image)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
        mean_loss = sum(running_loss) / len(running_loss)
        
        loop.set_description(f'[{epoch+1} /{epochs}]')
        loop.set_postfix(bacth_loss=loss.item(), mean_loss=mean_loss,lr=optimizer.param_groups[0]["lr"])

    torch.save(model.state_dict(), "./model_multiclass.pth")
