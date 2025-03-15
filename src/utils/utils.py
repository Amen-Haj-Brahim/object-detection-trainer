import json
import cv2 as cv

def get_categories(path):
  with open(path, 'r') as f:
    json_data=json.load(f)
    categories = [c["name"] for c in json_data["categories"]]
    return categories

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