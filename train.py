import torch
import dataloaders.ms_coco_dataloader
from helper.engine import train_one_epoch, evaluate
from models.faster_rcnn_resnet50_fpn import fasterrcnn_resnet50_fpn
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import dataloaders
def train(model,train_set, valid_set,epochs,lr,print_freq):  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  # trainable params are only the unfrozen ones
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
  # lr scheduler to try decaying lr
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,gamma=0.1)
  # set to train mode
  model.train()
  # using helper functions from the torch vision github repo, thanks torchvision !
  for epoch in range(epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, train_set, device, epoch, print_freq=print_freq)
    # update lr
    lr_scheduler.step()
    # eval on valid_set
    evaluate(model, valid_set, device=device)
  
  torch.save(model.state_dict(), "logs/my_model.pt")



if __name__ == "__main__":
  #------------------------------argument parser--------------------------------#
  parser = argparse.ArgumentParser(description='Object Detection Trainer')

  parser.add_argument('--epochs', type=int, default=5,help='number of epochs: 5')
  
  # parser.add_argument('--data-format', type=str,help='your data format (for now only coco is implemented)')

  parser.add_argument('--train-batch-size', type=int, default=4,help='train batch size: 4')
  
  parser.add_argument('--test-valid-batch-size', type=int, default=8,help='test and valid batch size: 8')
  
  parser.add_argument('--lr', type=float, default=0.0001,help='learning rate: 0.0001')
  
  parser.add_argument('--print-freq', type=int, default=5,help='print model metrics every N batches: 5')
    
  args = parser.parse_args()
  #------------------------------data loader------------------------------------#
  # this is assuming your data is augmented so i will only apply ToTensor()
  transform=transforms.Compose([
      transforms.ToTensor(),
      ])
  
  train_dataset = dataloaders.ms_coco_dataloader('train',transform=transform)
  valid_dataset= dataloaders.ms_coco_dataloader('valid',transform=transform)
  test_dataset = dataloaders.ms_coco_dataloader('test',transform=transform)

  train_loader = DataLoader(train_dataset,args.train_batch_size)
  test_loader = DataLoader(test_dataset,args.test_valid_batch_size)
  test_loader = DataLoader(test_dataset,args.test__valid_batch_size)

  model= fasterrcnn_resnet50_fpn(pretrained=True)

  train(model,train_loader,test_loader,args.epochs,args.lr,args.print_freq)