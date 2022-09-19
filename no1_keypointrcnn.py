### PACKAGES ###
import os
from typing import Tuple, List, Sequence, Callable, Dict

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

import albumentations as A
from albumentations.pytorch import ToTensorV2


### DATA CUSTOM ###
class KeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_df: pd.DataFrame,
        transforms: Sequence[Callable]=None
    ) -> None:
        self.image_dir = image_dir
        self.df = label_df
        self.transforms = transforms

    def __len__(self) -> int:
        return self.df.shape[0]  # the number of rows
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_id = self.df.iloc[index, 0]
        labels = np.array([1])
        keypoints = self.df.iloc[index, 1:].values.reshape(-1, 2).astype(np.int64)

        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

        image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.COLOR_BGR2RGB)

        targets ={
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets['image']
        image = image / 255.0

        targets = {
            'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
            'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
            'keypoints': torch.as_tensor(
                np.concatenate([targets['keypoints'], np.ones((15, 1))], axis=1)[np.newaxis], dtype=torch.float32
            )
        }

        return image, targets


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


### Data Transform & Train-Test-Split ###
def load_data(train_img_path, train_key_path):
    transforms = A.Compose([
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )

    total_df = pd.read_csv(train_key_path)
    train_key, valid_key = train_test_split(total_df[:10000], test_size=0.2, random_state=42)

    trainset = KeypointDataset(train_img_path, train_key, transforms)
    validset = KeypointDataset(train_img_path, valid_key, transforms)
    
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    return train_loader, valid_loader


### BRING MODEL ###
def get_model() -> nn.Module:
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=2)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
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


### TRAIN ###
def train(model, train_loader, optimizer, epoch, device='cuda'):
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        # data, target 값 DEVICE에 할당
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()             # optimizer gradient 값 초기화
        losses = model(images, targets)   # calculate loss

        loss = losses['loss_keypoint']    # keypoint loss
        loss.backward()                   # loss back propagation
        optimizer.step()                  # parameter update

        if (batch_idx+1) % 200 == 0:
            print(f'|epoch: {epoch} | batch: {batch_idx+1} / {len(train_loader)}')


def evaluate(model, test_loader, device='cuda'):
    model.train()
    test_loss = 0   # test_loss 초기화

    with torch.no_grad():
        for images, targets in test_loader:
            # data, target 값 DEVICE에 할당
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)               # validation loss
            test_loss += float(losses['loss_keypoint'])   # sum of all loss

    test_loss /= len(test_loader.dataset)                 # 평균 loss
    return test_loss


def train_model(train_loader, val_loader, num_epochs=30, device='cuda'):
    model = get_model()
    model.to(device)

    best_loss = 999999      # initialize best loss
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, num_epochs+1):
        since = time.time()
        train(model, train_loader, optimizer, epoch, device)
        train_loss = evaluate(model, train_loader)
        val_loss = evaluate(model, val_loader)

        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(model, '../models/RCNN_ep'+str(epoch)+'_'+str(best_loss)+'.pt')
            print('Best Model Saved, Loss: ', val_loss)

        time_elapsed = time.time()-since
        print()
        print('------------- epoch {} ----------------'.format(epoch))
        print('Train Keypoint Loss: {:.4f}, Val Keypoint Loss: {:.4f}'.format(train_loss, val_loss))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print()


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_image_path = 'C:/Users/admin/Desktop/CAT/RAW/ARCH'
    train_key_path   = 'annotations_1.csv'

    train_loader, valid_loader = load_data(train_image_path, train_key_path)
    train_model(train_loader, valid_loader, num_epochs=5, device=DEVICE)

    '''
    default: epoch - 30,
             device - cuda
    '''

if __name__ == '__main__':
    main()