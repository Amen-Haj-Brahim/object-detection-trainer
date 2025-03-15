import time
import yaml

from helper.engine import train_one_epoch, evaluate

import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
# from dataloaders.ms_coco_dataloader import CocoDataset

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
    train_one_epoch(model, optimizer, train_loader, device, epoch,epochs, print_freq=print_freq)
    
    # update lr
    lr_scheduler.step()
    
    # eval on valid_set
    evaluate(model, valid_loader, device=device,print_freq=print_freq)