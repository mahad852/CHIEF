from data.LungHist700 import LungHist700
from models.CHIEF import CHIEF
from models.ctran import ctranspath

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision import transforms

from PIL import Image

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = LungHist700("/home/mali2/datasets/LungHist700/data/images", is_train=True, transform=trnsfrms_val)
train_loader = DataLoader(train_ds, batch_size=1)

val_ds = LungHist700("/home/mali2/datasets/LungHist700/data/images", is_train=False, transform=trnsfrms_val)
val_loader = DataLoader(val_ds, batch_size=1)

model_embed = ctranspath()
model_embed.head = nn.Identity()

td = torch.load('./model_weight/CHIEF_CTransPath.pth', weights_only=True)
model_embed.load_state_dict(td['model'], strict=True)

model_embed = model_embed.to(device)

model = CHIEF(size_arg="small", dropout=True, n_classes=7)
model = model.to(device)

td = torch.load('./model_weight/CHIEF_finetune.pth', map_location=device, weights_only=True)

model.load_state_dict(td, strict=False)

anatomical=6

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

num_epochs = 200

def run_val():
    val_loss = 0.0
    num_samples = 0
    num_correct = 0

    model.eval()
    model_embed.eval()

    for _, (image, label) in enumerate(val_loader):
        num_samples += label.shape[0]

        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, num_classes=7).type(torch.float).to(device)

            patch_feature_emb = model_embed(image)

            x,tmp_z = patch_feature_emb, anatomical

            result = model(x, torch.tensor([tmp_z]))

            val_loss += criterion(result["bag_logits"], label_one_hot).cpu().item()
        
            num_correct += torch.sum(label == F.softmax(result["bag_logits"], dim=1).argmax(dim=1))
    
    print("Correct:", num_correct, "Total:", num_samples)

    return val_loss/num_samples, num_correct/num_samples


for param in model_embed.parameters():
    param.requires_grad = False

best_acc = 0

for e in range(num_epochs):
    average_loss = 0.0
    num_batches = len(train_ds)

    model.train()
    model_embed.eval()

    for b, (image, label) in enumerate(train_loader):
        image = image.to(device)

        with torch.no_grad():
            patch_feature_emb = model_embed(image)

        label = F.one_hot(label, num_classes=7).type(torch.float).to(device)

        optimizer.zero_grad()

        x,tmp_z = patch_feature_emb,anatomical
        result = model(x, torch.tensor([tmp_z]))
        wsi_feature_emb = result['WSI_feature']  ###[1,768]

        loss = criterion(result["bag_logits"], label)

        loss.backward()
        optimizer.step()

        average_loss += loss.detach().cpu().item()

        if (b + 1) % 50 == 0:
            print(f"Epoch: {e + 1}, Batch: {b + 1} | Loss: {average_loss/(b + 1)}")

    val_loss, val_acc = run_val()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "./model_weight/chief_lunghist700.pth")
        torch.save(model_embed.state_dict(), "./model_weight/chief_embed_lunghist700.pth")

    print()
    print(f"Epoch: {e + 1} | Train Loss: {average_loss/num_batches:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
   
    print("-" * 20)    

    # break