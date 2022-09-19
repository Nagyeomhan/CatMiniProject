### PACKAGES ###
import os
import cv2
import random
import transform
import torch
import torch.nn
import torch.utils
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from transform import get_affine_transform, affine_transform


class SingleModelConfig:
  def __init__(self,
               
               input_size: List[int] = [384, 288],
               kpd: float = 4.0,
               epochs: int = 15,
               sigma: float = 3.0,
               num_joints: int = 15,
               batch_size: int = 16,
               random_seed: int = 2021,
               test_ratio: float = 0.1,
               learning_rate: float = 1e-3,
               save_folder: str = '',
               # main_dir: str = main_dir,
               loss_type: str = 'MSE',
               target_type: str = 'gaussian',
               post_processing: str = 'dark',
               debug: bool = False,
               shift: bool = False,
               init_training: bool = False,
              
    ):

    self.save_folder = save_folder
    if not os.path.exists(self.save_folder) and self.save_folder != '':
      os.makedirs(self.save_folder, exist_ok=True)

    self.epochs = epochs
    self.seed = random_seed
    self.lr = learning_rate
    self.loss_type = loss_type
    self.num_joints = num_joints
    self.batch_size = batch_size
    self.test_ratio = test_ratio
    self.init_training = init_training

    self.kpd = kpd

    self.sigma = sigma
    self.shift = shift
    self.debug = debug
    
    self.target_type = target_type
    self.image_size = np.array(input_size)
    self.output_size = self.image_size//4
    self.post_processing = post_processing

    self.joints_name = {
          0: 'nose', 1: 'middle_forehead', 2: 'lip_tail', 3: 'middle_lower_lip', 4: 'neck',
          5: 'right_foreleg_start', 6: 'left_foreleg_start', 7: 'right_foreleg_ankle', 8: 'left_foreleg_ankle',
          9: 'right_femur', 10: 'left_femer', 11: 'right_hindleg_ankle', 12: 'left_hindleg_ankle',
          13: 'tail_start', 14: 'tail_end', 
    }

    self.joint_pair = [
          (0, 1),  (0,3), (2,3), (3, 4), (4, 5),
          (4, 6), (5, 7), (6, 8), (4, 13), (13, 9), 
          (13, 10), (13,14), (9, 11), (10,12)
    ]

    self.flip_pair = [
          (9, 10), (7,8), (5, 6), (11, 12)
    ]

    self.joints_weight = np.array(
            [
                1.3,           # 코
                1.3,           # 이마 중앙 
                1.3,           # 입꼬리(입끝)
                1.3,           # 아래 입술 중앙
                1.3,           # 목
                1.3, 1.3,      # 앞다리 시작
                1.3, 1.3,      # 앞다리 발목
                1.3, 1.3,      # 대퇴골
                1.3, 1.3,      # 뒷다리 발목
                1.3,           # 꼬리 시작
                1.3,           # 꼬리 끝
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))
    
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1,  self.num_joints + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    self.joint_colors = {k: colors[k] for k in range(self.num_joints)}


class AnimalKeypoint(Dataset):
    def __init__(
        self,
        cfg: SingleModelConfig,
        image_dir: str, 
        label_df: pd.DataFrame, 
        transforms: Sequence = None,
        mode: str = 'train'
    ) -> None:
        self.image_dir = image_dir
        self.df = label_df
        self.transforms = transforms
        self.mode = mode
        self.kpd = cfg.kpd
        self.debug = cfg.debug
        self.shift = cfg.shift
        self.num_joints = cfg.num_joints
        self.flip_pairs = cfg.flip_pair
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.output_size
        self.sigma = cfg.sigma
        self.target_type = cfg.target_type
        self.joints_weight = cfg.joints_weight

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        image_id = self.df.iloc[index, 0]
        labels = np.array([1])
        keypoints = self.df.iloc[index, 1:].values.reshape(-1, 2).astype(np.float32)
        keypoints = np.concatenate([keypoints, np.ones((15, 1))], axis=1)

        # define bbox
        xmin = np.min(keypoints[:, 0])
        xmax = np.max(keypoints[:, 0])
        width = xmax - xmin if xmax > xmin else 20
        center = (xmin + xmax)/2.
        xmin = int(center - width/2.*1.2)
        xmax = int(center + width/2.*1.2)

        ymin = np.min(keypoints[:, 1])
        ymax = np.max(keypoints[:, 1])
        height = ymax - ymin if ymax > ymin else 20
        center = (ymin + ymax)/2.
        ymin = int(center - height/2.*1.2)
        ymax = int(center + height/2.*1.2)
        

        x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin
        aspect_ratio = self.image_size[1] / self.image_size[0]
        centre = np.array([x+w*.5, y+h*.5])
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        
        scale = np.array([w, h]) * 1.25
        rotation = 0

        
        image = cv2.imread(os.path.join(self.image_dir, image_id))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)