from torch.utils.data import Dataset
import os
import random
from PIL import Image

class LungHist700(Dataset):
    def __init__(self, root: str, is_train = True, transform = None):
        self.root = root
        self.transform = transform

        self.classes = {
            0: "aca_bd", 
            1: "aca_md", 
            2: "aca_pd", 
            3: "nor", 
            4: "scc_bd",
            5: "scc_md",
            6: "scc_pd"
        }
                        
        self.images_and_labels = self.get_images_and_labels()
        random.Random(42).shuffle(self.images_and_labels)

        if is_train:
            self.images_and_labels = self.images_and_labels[:int(len(self) * 0.70)]
        else:
            self.images_and_labels = self.images_and_labels[int(len(self) * 0.70):]

    def get_images_and_labels(self):
        images_and_labels = []

        for class_idx in self.classes.keys():
            sub_dir = os.path.join(self.root, self.classes[class_idx])
            for image_name in os.listdir(sub_dir):
                images_and_labels.append((os.path.join(sub_dir, image_name), class_idx))
                
        return images_and_labels
    
    def __len__(self) -> int:
        return len(self.images_and_labels)

    def __getitem__(self, index):
        image, label = self.images_and_labels[index]
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)
        
        return image, label