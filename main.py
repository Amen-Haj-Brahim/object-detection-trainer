import yaml
import time

from src.models import faster_rcnn_resnet50_fpn
from src.trainer import train
from src.tester import test
from src.utils import utils
from src.dataloaders.ms_coco_dataloader import CocoDataset

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def main():    
    # import hyperparams
    with open("config.yaml") as f:
        config=yaml.safe_load(f)
        print(config)
        
    # load datasets
    transform=transforms.Compose([transforms.ToTensor()])
    
    train_dataset = CocoDataset(img_dir="data/train/", transforms=transform)
    valid_dataset = CocoDataset(img_dir="data/valid/", transforms=transform)
    test_dataset = CocoDataset(img_dir="data/test/", transforms=transform)

    train_loader = DataLoader(train_dataset,config["train_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset,config["test_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset,config["valid_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
    
    classes=utils.get_categories("data/train/_annotations.coco.json")
    
    model=faster_rcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(classes)

    # run trainer
    train(model,train_loader,valid_loader,config["epochs"],config["lr"],config["print_freq"])
    
    # save model with timestamp when training loop ends
    model_name=str(round(time.time()))
    torch.save(model.state_dict(), "models/"+model_name+".pt")
    
    # run tester on all batches of the test dataset
    test(model,model_name,test_loader,"data/test","data/train/_annotations.coco.json")
    
if __name__ == "__main__":
    main()