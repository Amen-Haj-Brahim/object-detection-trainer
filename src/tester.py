import os
import time
import torch
import matplotlib.pyplot as plt
from torchvision.ops import nms
import cv2 as cv
import argparse
from utils.utils import get_categories
def draw_boxes(image, boxes, labels, confs,classes):
  
    for i, box in enumerate(boxes):
        # casting to int because for some reason it wouldn't work otherwise
        x1, y1, x2, y2 = map(int, box)
         # draw bbox
        cv.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        # write class and confidence score
        label = classes[labels[i]]
        if confs is not None:
            label += str(round(confs[i],2))
        cv.putText(image,label,(x1, y1-5),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    
    return image


def test(model,model_name,test_loader,test_dir):
  
  if not os.path.exists(test_dir+model_name):
    os.makedirs(test_dir+model_name,exist_ok=True)
  
  model.eval()
  c=1
  classes=get_categories("../data/train/_annotations.coco.json")

  # loop through batches
  for i, (images, targets) in enumerate(test_loader):    
    # move images and labels+annotations to gpu
    images = [img.to("cuda" if torch.cuda.is_available() else "cpu") for img in images]

    # run em through the model
    with torch.no_grad():
        preds = model(images)
    
    # loop through samples in the batch
    for j, image in enumerate(images):
      
        print("batch : "+str(i)+"img: "+str(c)+"/"+str(len(test_loader.dataset)))
      
        # pillow loads the images in (c,h,w) format so i gotta transpose it to (h,w,c)
        image = image.cpu().numpy().transpose(1, 2, 0)  
        image = (image * 255).astype("uint8") 

        # get bboxes and labels
        real_bboxes = targets[j]["boxes"].cpu().numpy()
        real_labels = targets[j]["labels"].cpu().numpy()

        # get preds
        pred_boxes = preds[j]["boxes"].cpu().numpy()
        pred_labels = preds[j]["labels"].cpu().numpy()
        pred_conf = preds[j]["scores"].cpu().numpy()

        # Apply non max supression to remove overlapping boxes
        # change iou threshold if you want you do you
        keep = nms(torch.tensor(pred_boxes), torch.tensor(pred_conf), iou_threshold=0.3)
        pred_bboxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_conf = pred_conf[keep]

        # draw bboxes
        real_img = draw_boxes(image.copy(), real_bboxes, real_labels,None,classes)
        pred_img = draw_boxes(image.copy(), pred_bboxes, pred_labels, pred_conf,classes)

        plt.figure(figsize=(10,10))

        # plot preds and real
        plt.subplot(1, 2, 1)
        plt.imshow(real_img)
        plt.title("ground truth")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_img)
        plt.title("predictions")
        plt.axis("off")


        plt.savefig(test_dir+model_name+"/test_"+str(i)+"_"+str(j)+".png")
        plt.close()
        c+=1

  print("--------> done testing check /tests/"+model_name+" <--------")
  
  
  

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from utils.utils import get_categories
    from models.faster_rcnn_resnet50_fpn import fasterrcnn_resnet50_fpn
    from dataloaders.ms_coco_dataloader import CocoDataset
    transform=transforms.Compose([
      transforms.ToTensor(),
      ])
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, default="",help='model file name')
    
    args = parser.parse_args()

    if args.model_name=="":
        print("you forgot to pass the model name do it as follows : python tester.py --model-name your_model_name")
        exit()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=fasterrcnn_resnet50_fpn(get_categories("../data/train/_annotations.coco.json"))

    model.load_state_dict(torch.load("../models/"+args.model_name+".pt"))
    model.to(device)

    test_dataset = CocoDataset('../data/test', transforms=transform)
    test_loader = DataLoader(test_dataset,4,shuffle=True,collate_fn=lambda x: tuple(zip(*x)))
    
    test(model,args.model_name,test_loader,"../tests/")