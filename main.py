import yaml
import time

from src.models import faster_rcnn_resnet50_fpn
from src.trainer import train
from src.tester import test
from src.utils import utils

import torch
from torchvision.datasets import CocoDetection
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
    
    train_dataset = CocoDetection(root="data/train/", annFile="data/train/_annotations.coco.json", transform=transform)
    valid_dataset = CocoDetection(root="data/valid/", annFile="data/valid/_annotations.coco.json", transform=transform)
    test_dataset = CocoDetection(root="data/test/", annFile="data/test/_annotations.coco.json", transform=transform)

    train_loader = DataLoader(train_dataset,config["train_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset,config["test_valid_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset,config["test_valid_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
    
    classes=utils.get_categories()
    
    model=faster_rcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(classes)

    # run trainer
    train(model,train_loader,valid_loader,config["lr"],config["print_freq"])
    
    # save model with timestamp when training loop ends
    torch.save(model.state_dict(), "/models/"+str(round(time.time()))+".pt")
    
    # run tester on all batches of the test dataset
    # test(model,test_loader)
    
if __name__ == "__main__":
    main()