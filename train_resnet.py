from data.LungHist700 import LungHist700
from models.CHIEF import CHIEF
from models.ctran import ctranspath

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.models.resnet import resnet50, ResNet50_Weights

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 7

trnsfrms_val = ResNet50_Weights.DEFAULT.transforms()


train_ds = LungHist700("/home/mali2/datasets/LungHist700/data/images", is_train=True, transform=trnsfrms_val)
train_loader = DataLoader(train_ds, batch_size=4)

val_ds = LungHist700("/home/mali2/datasets/LungHist700/data/images", is_train=False, transform=trnsfrms_val)
val_loader = DataLoader(val_ds, batch_size=4)

td = torch.load("./model_weight/resnet50-11ad3fa6.pth", map_location=device, weights_only=True)
td = {k: v for k, v in td.items() if not k.startswith('fc.')}

model = resnet50(num_classes = num_classes).to(device)

model.load_state_dict(td, strict=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

num_epochs = 200

def run_val():
    val_loss = 0.0
    num_samples = 0
    num_batches = 0
    num_correct = 0

    model.eval()

    for _, (image, label) in enumerate(val_loader):
        num_samples += label.shape[0]
        num_batches += 1

        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, num_classes=num_classes).type(torch.float).to(device)

            logits = model(image)

            val_loss += criterion(logits, label_one_hot).cpu().item()
        
            num_correct += torch.sum(label == F.softmax(logits, dim=1).argmax(dim=1))
    
    return val_loss/num_batches, num_correct/num_samples


for name, param in model.named_parameters():
    if name.startswith("bn."):
        param.requires_grad = False

best_acc = 0

for e in range(num_epochs):
    average_loss = 0.0
    num_batches = 0

    model.train()

    for b, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = F.one_hot(label, num_classes=num_classes).type(torch.float).to(device)

        optimizer.zero_grad()

        logits = model(image)
        
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        average_loss += loss.detach().cpu().item()
        num_batches += 1

        if (b + 1) % 25 == 0:
            print(f"Epoch: {e + 1}, Batch: {num_batches} | Loss: {average_loss/num_batches}")

    val_loss, val_acc = run_val()
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "./model_weight/resnet50_lunghist700.pth")

    print()
    print(f"Epoch: {e + 1} | Train Loss: {average_loss/num_batches:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
   
    print("-" * 20)    