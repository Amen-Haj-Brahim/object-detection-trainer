import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def fasterrcnn_resnet50_fpn(classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)

    return model