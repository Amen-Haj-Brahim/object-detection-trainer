from helper.engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter

import torch
# from dataloaders.ms_coco_dataloader import CocoDataset

def get_validation_loss(model, valid_loader, device):
    model.train()
    total_loss = 0

    with torch.no_grad():
        for images, targets in valid_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets) 
            print(loss_dict)
            total_loss += sum(loss_dict.values()).item()

    avg_loss = total_loss / len(valid_loader)
    model.eval()
    return avg_loss



def train(model,train_loader, valid_loader,epochs,lr,print_freq):  
  
  # tensorboard and metrics for logging
  logger = SummaryWriter("logs/")
  coco_metrics = [
    "AP@IoU=0.50:0.95",
    "AP@IoU=0.50",
    "AP@IoU=0.75",
    "AP small",
    "AP med",
    "AP large",
    "AR 1 det",
    "AR 10 det",
    "AR 100 detections", 
    "AR small",
    "AR med",
    "AR large"]
  
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
    
    # training for one epoch and log losses
    train_loss=train_one_epoch(model, optimizer, train_loader, device, epoch,epochs, print_freq=print_freq)
    print(train_loss,"\n")
    logger.add_scalar("train_loss", train_loss.meters["loss"].global_avg, epoch)
    # update lr
    lr_scheduler.step()
    
    # eval and log on valid_set
    val_metrics=evaluate(model, valid_loader, device=device,print_freq=print_freq)
    val_loss=get_validation_loss(model, valid_loader, device)
    coco_stats = val_metrics.coco_eval["bbox"].stats 
    
    # loop through em and log
    for i, metric_name in enumerate(coco_metrics):
        logger.add_scalar(metric_name,coco_stats[i],epoch)

    logger.add_scalar("val_loss",val_loss, epoch)
  
  logger.close()