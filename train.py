import torch
from helper.engine import train_one_epoch, evaluate
from models.faster_rcnn_resnet50_fpn import fasterrcnn_resnet50_fpn
def train():
  model= fasterrcnn_resnet50_fpn()
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  # trainable params are only the unfrozen ones
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, 
                                                    weight_decay=0.0005)
  # lr scheduler to try decaying lr
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,gamma=0.1)
                    
  
  # training for 10 epochs
  num_epochs = 1
  # using helper functions from the torch vision github repo, thanks torchvision !
  for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, train_set, device, epoch, print_freq=5)
    # update lr
    lr_scheduler.step()
    # eval on valid_set
    evaluate(model, valid_set, device=device)
    
    
    
    
if __name__ == "__main__":
  train()