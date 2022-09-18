'''
3번!
'''

# 1. 패키지 불러오기
from torch import nn
# from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

# 2. 코드 작성
def get_model() -> nn.Module:   # resnet 101, resnet 152
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=2)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names = ['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=14,
    sampling_ratio=2
    )
    
    model = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=15,
        box_roi_pool=roi_pooler,
        keypoint_roi_pool=keypoint_roi_pooler
    )

    return model