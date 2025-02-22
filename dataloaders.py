from pycocotools.coco import COCO
import os
from PIL import Image
from torch import as_tensor,float32,int64
from torch.utils.data import Dataset
class CocoDataset(Dataset):
    def __init__(self, json_path, img_dir, transforms):
        self.coco = COCO(json_path)
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = list(self.coco.imgs.keys())
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # load image
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        
        # converting coco format (x,y,h,w) to (xmin, ymin, xmax, ymax) because that's what the model expects
        boxes = []
        labels = []
        areas=[]
        iscrowd=[]
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])
        
        boxes = as_tensor(boxes, dtype=float32)
        labels = as_tensor(labels, dtype=int64)
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, {"boxes": boxes, "labels": labels, "image_id": img_id,"area":torch.tensor(areas),"iscrowd":torch.tensor(iscrowd)}