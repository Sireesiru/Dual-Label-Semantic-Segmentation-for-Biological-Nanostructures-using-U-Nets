########### Reading input Images and masks processing block #######################

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class CocoBacteriaDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=True):
        self.coco = COCO(annotation_file)  # Load COCO dataset annotations
        self.image_dir = image_dir  # Directory containing images
        self.transform = transform  # Transformations to apply
        self.image_ids = self.coco.getImgIds()  # List of image IDs in the dataset 
        # Get category IDs for OM and IM.
        self.cat_ids = self.coco.getCatIds(catNms=['OM', 'IM'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_metadata['file_name'])
        image = Image.open(img_path).convert("L")  # Load as grayscale

        # Convert image to tensor
        image = transforms.ToTensor()(image)  # Convert the image to tensor.ToTensor() is not simple conversion to tensor. It does PIL to Tensor + channel reordering + 0–255→0–1 scaling

        # Load multilabel segmentation masks
        h, w = img_metadata['height'], img_metadata['width']
        # Create a mask with 2 channels: [2, H, W]
        multi_mask = np.zeros((2, h, w), dtype=np.uint8)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        for ann in annotations:
            cat_id = ann['category_id']
            channel = 0 if cat_id == self.cat_ids[0] else 1
            segmentation = ann['segmentation']
            for poly in segmentation:
                poly = np.array(poly).reshape((int(len(poly) / 2), 2)).astype(np.int32)
                cv2.fillPoly(multi_mask[channel], [poly], color=1)
        
        # Convert to tensor [2, H, W]
        mask = torch.tensor(multi_mask, dtype=torch.float32)
        raw_image, raw_mask = image.clone(), mask.clone()

        if self.transform:
            # Note: 320x320 resize should be consistent
            #image = TF.resize(image, (640, 640))
            #mask = TF.resize(mask, (640, 640))

            angle, translations, scale, shear = transforms.RandomAffine.get_params(
                degrees=(-180, 180), 
                translate=(0.1, 0.1), 
                scale_ranges=(0.9, 1.1), 
                shears=(-5, 5), 
                img_size=(640, 640))
            
            image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=shear)
            mask = TF.affine(mask, angle=angle, translate=translations, scale=scale, shear=shear)
            
        return raw_image, raw_mask, image, mask


#Data Loader-Load the train, Test and validation datasets

train_annotation_file = "/home/cloud/old-volume/home/cloud/bacteria-thickness_additional.v4i.coco/train/_annotations.coco.json"
train_image_dir = "/home/cloud/old-volume/home/cloud/bacteria-thickness_additional.v4i.coco/train"
train_dataset = CocoBacteriaDataset(train_annotation_file, train_image_dir, transform=True)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


valid_annotation_file = "/home/cloud/old-volume/home/cloud/bacteria-thickness_additional.v4i.coco/valid/_annotations.coco.json"
valid_image_dir = "/home/cloud/old-volume/home/cloud/bacteria-thickness_additional.v4i.coco/valid"
valid_dataset = CocoBacteriaDataset(valid_annotation_file, valid_image_dir, transform=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)


test_annotation_file = "/home/cloud/old-volume/home/cloud/bacteria-thickness_additional.v4i.coco/test/_annotations.coco.json"
test_image_dir = "/home/cloud/old-volume/home/cloud/bacteria-thickness_additional.v4i.coco/test/"
test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)