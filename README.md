# CHIEF - Clinical Histopathology Imaging Evaluation Foundation Model
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

### A Pathology Foundation Model for Cancer Diagnosis and Prognosis Prediction

Wang X^, Zhao J^, Marostica E, Yuan W, Jin J, Zhang J, Li R, Tang H, Wang K, Li Y, Wang F, Peng Y, Zhu J, Zhang J, Jackson CR, Zhang J, Dillon D, Lin NU, Sholl L, Denize T, Meredith D, Ligon KL, Signoretti S, Ogino S, Golden JA, Nasrallah MP, Han X, Yang S<sup>+</sup>, Yu KH<sup>+</sup>.


Nature (2024).
https://www.nature.com/articles/s41586-024-07894-z

*Lead Contact: Kun-Hsing Yu, M.D., Ph.D.*

#### ABSTRACT 
*Histopathology image evaluation is indispensable for cancer diagnoses and subtype classification. Standard artificial intelligence (AI) methods for histopathology image analyses have focused on optimizing specialized models for each diagnostic task. Although such methods have achieved some success, they often have limited generalizability to images generated by different digitization protocols or samples collected from different populations. To address this challenge, we devised the Clinical Histopathology Imaging Evaluation Foundation (CHIEF) model, a general-purpose weakly supervised machine learning framework to extract pathology imaging features for systematic cancer evaluation. CHIEF leverages two complementary pretraining methods to extract diverse pathology representations: unsupervised pretraining for tile-level feature identification and weakly supervised pretraining for whole-slide pattern recognition. We developed CHIEF using 60,530 whole-slide mimages (WSIs) spanning 19 distinct anatomical sites. Through pretraining on 44 terabytes of high-resolution pathology imaging datasets, CHIEF extracted microscopic representations useful for cancer cell detection, tumor origin identification, molecular profile characterization, and prognostic prediction. We successfully validated CHIEF using 19,491 whole-slide images from 32 independent slide sets collected from 24 hospitals and cohorts internationally. Overall, CHIEF outperformed the state-of-the-art deep learning methods by up to 36.1%, showing its ability to address domain shifts observed in samples from diverse populations and processed by different slide preparation methods. CHIEF provides a generalizable foundation for efficient digital pathology evaluation for cancer patients.*

![Github-Cover](https://github.com/hms-dbmi/CHIEF/assets/31292151/442391e2-3706-4337-ae9a-69c2cc24222e)

Docker images (model weights) are available at https://hub.docker.com/r/chiefcontainer/chief/


## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce V100 x 32GB)
* Python (Python 3.8.10),torch==1.8.1+cu111,
torchvision==0.9.1+cu111, h5py==3.6.0, matplotlib==3.5.2, numpy==1.22.3, opencv-python==4.5.5.64, openslide-python==1.3.0, pandas==1.4.2, Pillow==10.0.0, scikit-image==0.21.0
scikit-learn==1.2.2,scikit-survival==0.21.0, scipy==1.8.0, tensorboardX==2.6.1, tensorboard==2.8.0.

Install the modified [timm](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing) library
```
pip install timm-0.5.4.tar
```

### Installation Guide for Linux (using anaconda)
1. Installation anaconda(https://www.anaconda.com/distribution/)
```
2. sudo apt-get install openslide-tools
```
```
3. pip install requirements.txt
```


```
git clone https://github.com/hms-dbmi/CHIEF.git
cd CHIEF
```

Downloading Pre-trained models
Request access to the model [weights](https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV?usp=sharing). The docker images are already included and do not need to be downloaded.
## Creating model
### Patch-level model(CHIEF-Ctranspath)
using the commands below:
````
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.ctran import ctranspath

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)


model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./model_weight/CHIEF_CTransPath.pth')
model.load_state_dict(td['model'], strict=True)
model.eval()

````
### Running Inference
Get patch features

````
image = Image.open("./exsample/exsample.tif")
image = trnsfrms_val(image).unsqueeze(dim=0)
with torch.no_grad():
    patch_feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,768]
    print(patch_feature_emb.size())

````
Here's an example.

````
python3 Get_CHIEF_patch_feature.py
````
### WSI-level model(CHIEF)
There are already some extracted features for the patch images, please [weights](https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV?usp=sharing) them first.Put it under `./Downstream/Tumor_origin/src/feature`. The docker images are already included and do not need to be downloaded.

````
import torch, torchvision
import torch.nn as nn
from models.CHIEF import CHIEF


model = CHIEF(size_arg="small", dropout=True, n_classes=2)

td = torch.load(r'./model_weight/CHIEF_pretraining.pth')
model.load_state_dict(td, strict=True)
model.eval()

````
### Running Inference
Get WSI-level features


````

full_path = r'./Downstream/Tumor_origin/src/feature/tcga/TCGA-LN-A8I1-01Z-00-DX1.F2C4FBC3-1FFA-45E9-9483-C3F1B2B7EF2D.pt'

features = torch.load(full_path, map_location=torch.device('cpu'))
anatomical=13
with torch.no_grad():
    x,tmp_z = features,anatomical
    result = model(x, torch.tensor([tmp_z]))
    wsi_feature_emb = result['WSI_feature']  ###[1,512]
    print(wsi_feature_emb.size())

````

Here's an example.

````
python3 Get_CHIEF_WSI_level_feature.py
````

Batch WSI image extraction

````
python3 Get_CHIEF_WSI_level_feature_batch.py
````


### Finetune  model

Here is exsample:
````
cd ./Downstream/Tumor_origin/src

Training/Val/Test Splits is here(./Downstream/Tumor_origin/src/csv)

CUDA_VISIBLE_DEVICES=0 python3 train_valid_test.py --classification_type='tumor_origin' --exec_mode='train' --exp_name='tcga_only_7_1_2'
````

### Evaluation

#### Reproducibility
To reproduce the results in our paper,  please download the [feature](). The docker images are already included and do not need to be downloaded. The source data for Extended Data Figure 5 can be found [here](https://www.dropbox.com/scl/fo/y6zv790iw0bozmz63qdq1/AO5SNj8PCqFL5ecNypQswFs?rlkey=htiaghgwaymo4ksfrvwmpwkbc&st=0v8r1myk&dl=0).
##### 1. Cancer_Cell_Detection


````
CUDA_VISIBLE_DEVICES=0 python3 classification_eval.py --config_path configs/colon.yaml --dataset_name Dataset_PT
````

````
CUDA_VISIBLE_DEVICES=0 python3 classification_eval.py --config_path configs/breast.yaml --dataset_name DROID_breast
````

##### 2. Tumor origin Classification

````
CUDA_VISIBLE_DEVICES=0 python3 train_valid_test.py --classification_type='tumor_origin' --exec_mode='eval' --exp_name='tcga_only_7_1_2' --split_name='test' 
````
##### 3. Biomaker

````
CUDA_VISIBLE_DEVICES=0 python3 classification_eval.py --config_path configs/IDH_lgg.yaml --dataset_name muv_lgg
````
##### 4. Survial
Below we provide a quick example using a subset of cases for RCC survival task.

```shell
cd ./Downstream/Survial

run inference.ipynb
```


```shell
docker pull chiefcontainer/chief:v1
docker run --rm -it --entrypoint /bin/bash chiefcontainer/chief:v1
```
You will see a CHIEF folder under "root".

Attention-based heatmaps can be viewed at https://yulab.hms.harvard.edu/projects/CHIEF/CHIEF.htm

## Reference and Acknowledgements
We thank the authors and developers for their contribution as below.
* [SupContrast: Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast)
* [CLAM](https://github.com/mahmoodlab/CLAM)
* [multimodal-cancer-origin-prediction](https://github.com/mahmoodlab/multimodal-cancer-origin-prediction)


## Issues
- Please open new threads or address all questions to xiyue.wang.scu@gmail.com or Kun-Hsing_Yu@hms.harvard.edu

## License
CHIEF is made available under the GPLv3 License and is available for non-commercial academic purposes. 

