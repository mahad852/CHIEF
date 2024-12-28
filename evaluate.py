from data.LungHist700 import LungHist700
from models.CHIEF import CHIEF
from models.ctran import ctranspath

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision import transforms

from torchvision.models.resnet import resnet50, ResNet50_Weights

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="name of the model that you want to use")
args = parser.parse_args()

model_name = args.model_name

num_classes = 7
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_transforms():
    if model_name == "resnet50":
        return ResNet50_Weights.DEFAULT.transforms()
    elif model_name == "chief":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = mean, std = std)
            ]
        )
    else:
        raise NotImplementedError()

def get_model():
    if model_name == "resnet50":
        td = torch.load("./model_weight/resnet50_lunghist700.pth", map_location=device, weights_only=True)
        model = resnet50(num_classes=num_classes).to(device)
        model.load_state_dict(td, strict=True)

        return model, None
    elif model_name == "chief":
        model_embed = ctranspath()
        model_embed.head = nn.Identity()
        td = torch.load('./model_weight/CHIEF_CTransPath.pth', weights_only=True)
        model_embed.load_state_dict(td['model'], strict=True)
        model_embed = model_embed.to(device)

        model = CHIEF(size_arg="small", dropout=True, n_classes=7)
        model = model.to(device)
        td = torch.load('./model_weight/chief_lunghist700.pth', map_location=device, weights_only=True)
        model.load_state_dict(td, strict=True)

        for param in model.parameters():
            param.requires_grad = False

        for param in model_embed.parameters():
            param.requires_grad = False

        return model, model_embed
    else:
        raise NotImplementedError()
    


model, model_embed = get_model()
trnsfrms = get_transforms()

val_ds = LungHist700("/home/mali2/datasets/LungHist700/data/images", is_train=False, transform=trnsfrms)
val_loader = DataLoader(val_ds, batch_size=1)


def run_inference_resnet50(image: torch.Tensor) -> torch.Tensor:
    return model(image)

def run_inference_chief(image: torch.Tensor) -> torch.Tensor:
    anatomical = 6

    with torch.no_grad():
        patch_feature_emb = model_embed(image)
        x, tmp_z = patch_feature_emb, anatomical

        result = model(x, torch.tensor([tmp_z]))
    
    return result["bag_logits"]

def run_inference(image: torch.Tensor) -> torch.Tensor:
    if model_name == "chief":
        return run_inference_chief(image)
    elif model_name == "resnet50":
        return run_inference_resnet50(image)
    else:
        raise NotImplementedError()

val_loss = 0.0
num_samples = 0
num_correct = 0
num_batches = 0


classes = {
    0: "aca_bd", 
    1: "aca_md", 
    2: "aca_pd", 
    3: "nor", 
    4: "scc_bd",
    5: "scc_md",
    6: "scc_pd"
}

model.eval()

if model_name == "chief":
    model_embed.eval()

results_overall = []
results_subclass = []

for _, (image, label) in enumerate(val_loader):
    with torch.no_grad():
        image = image.to(device)
        
        logits = run_inference(image)

        preds = logits.argmax(dim = 1).cpu()
        for i in range(preds.shape[0]):
            p, l = preds[i].item(), label[i].cpu().item()

            results_overall.append((classes[l].split("_")[0], classes[p].split("_")[0]))
            results_subclass.append((classes[l], classes[p]))


def precision(results, attr):
    tp, fp = 0, 0

    for (label, pred) in results:
        if pred != attr:
            continue

        if label == attr:
            tp += 1
        else:
            fp += 1
    
    return tp/(tp + fp) if tp + fp > 0 else 0.0

def recall(results, attr):
    tp, fn = 0, 0

    for (label, pred) in results:
        if label != attr:
            continue

        if pred == attr:
            tp += 1
        else:
            fn += 1
    
    return tp/(tp + fn) if tp + fn > 0 else 0.0

def get_total(results, attr):
    total = 0
    for label, _ in results:
        if label == attr:
            total += 1
    return total

def acc(results):
    total = len(results)
    correct=  0
    
    for (label, pred) in results:
        if label == pred:
            correct += 1
    
    print("Correct:", correct, "Total:", total)
    
    return correct/total

overall_classes = ["aca", "scc", "nor"]
sub_classes = ["aca_bd", "aca_md", "aca_pd", "nor", "scc_bd", "scc_md", "scc_pd"]

print("*" * 5, "Overall Stats", "*" * 5)
print("Accuracy:", acc(results_overall))
for oc in overall_classes:
    print(f"{oc:6} | \t Precision: {precision(results_overall, oc):0,.4f} | \t Recall: {recall(results_overall, oc):0,.4f} | \t Total: {get_total(results_overall, oc)}")

print("-" * 30)

print("*" * 5, "Subclass Stats", "*" * 5)
print("Accuracy:", acc(results_subclass))
for sc in sub_classes:
    print(f"{sc:6} | \t Precision: {precision(results_subclass, sc):0,.4f} | \t Recall: {recall(results_subclass, sc):0,.4f} | \t Total: {get_total(results_subclass, sc)}")