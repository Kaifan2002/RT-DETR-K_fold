import torch  
from torch.utils.data import Subset  
from sklearn.model_selection import KFold  
from .coco_dataset import CocoDetection  
from ...core import register  
  
@register()  
class KFoldCocoDetection:  
    def __init__(self, img_folder, ann_file, transforms, return_masks=False,   
                 remap_mscoco_category=False, n_splits=5, fold=0, random_seed=42):  
        self.img_folder = img_folder  
        self.ann_file = ann_file  
        self.transforms = transforms  
        self.return_masks = return_masks  
        self.remap_mscoco_category = remap_mscoco_category  
        self.n_splits = n_splits  
        self.fold = fold  
        self.random_seed = random_seed  
          
        # 创建完整数据集  
        self.full_dataset = CocoDetection(  
            img_folder, ann_file, transforms, return_masks, remap_mscoco_category  
        )  
          
        # 生成K折索引  
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)  
        self.indices = list(self.kfold.split(range(len(self.full_dataset))))  
          
        # 设置当前fold的训练和验证索引  
        self.train_indices, self.val_indices = self.indices[fold]  
          
        # 创建训练和验证子集  
        self.train_dataset = Subset(self.full_dataset, self.train_indices)  
        self.val_dataset = Subset(self.full_dataset, self.val_indices)  
      
    def get_train_dataset(self):  
        return self.train_dataset  
      
    def get_val_dataset(self):  
        return self.val_dataset