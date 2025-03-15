import time
import yaml

from helper.engine import train_one_epoch, evaluate

import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader


def train(model,train_loader, valid_loader,epochs,lr,print_freq):  

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  # trainable params are only the unfrozen ones
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
  
  # lr scheduler to try decaying lr
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,gamma=0.1)
  
  # set to train mode
  model.train()
  
  # using helper functions from the torch vision github repo, shoutout torchvision fr
  for epoch in range(epochs):
    
    # training for one epoch
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=print_freq)
    
    # update lr
    lr_scheduler.step()
    
    # eval on valid_set
    evaluate(model, valid_loader, device=device)
    

if __name__ == "__main__":
  
  from models.faster_rcnn_resnet50_fpn import fasterrcnn_resnet50_fpn
  from helper.engine import train_one_epoch, evaluate
  import utils.utils
  
  with open("config.yaml") as f:
      config=yaml.safe_load(f)
      print(config)

  # this is assuming your data is augmented so i will only apply ToTensor()
  transform=transforms.Compose([transforms.ToTensor()])
  
  # load datasets  
  train_dataset = CocoDetection(root="data/train/", annFile="data/train/_annotations.coco.json", transform=transform)
  valid_dataset = CocoDetection(root="data/valid/", annFile="data/valid/_annotations.coco.json", transform=transform)
  test_dataset = CocoDetection(root="data/test/", annFile="data/test/_annotations.coco.json", transform=transform)

  train_loader = DataLoader(train_dataset,config["train_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
  test_loader = DataLoader(test_dataset,config["test_valid_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
  valid_loader = DataLoader(valid_dataset,config["test_valid_batch_size"],collate_fn=lambda x: tuple(zip(*x)))
  
  classes=utils.utils.get_categories()

  model=fasterrcnn_resnet50_fpn(utils.utils.get_categories())
  
  # run trainer
  train(model,train_loader,valid_loader,config["epochs"],config["lr"],config["print_freq"])
  
  # save model with timestamp when training loop ends
  torch.save(model.state_dict(), "../models/"+str(round(time.time()))+".pt")