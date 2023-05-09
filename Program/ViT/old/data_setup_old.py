import os
import json
from typing import Callable, Optional, Tuple
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from typing import Any, Callable, Optional
from PIL import Image

class CustomDataSet(CocoDetection):
    def __init__(self, 
                 root: str, 
                 annFile: str, 
                 transform: Callable[..., Any] | None = None, 
                 target_transform: Callable[..., Any] | None = None, 
                 transforms: Callable[..., Any] | None = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
    
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __repr__(self) -> str:
        f_str = 'Dataset ' + self.__class__.__name__ + '\n'
        f_str += f'     Number of datapoints: {self.__len__()}'
        f_str += f'     Root Location: {self.root}'
        temp_str =  '     Transform (if any): '
        f_str += f'{temp_str} {self.transform.__repr__()}'.replace('\n', '\n' + ' ' * len(temp_str))
        temp_str =  '     Target Transforms (if any): '
        f_str += f'{temp_str} {self.transform.__repr__()}'.replace('\n', '\n' + ' ' * len(temp_str))
        return f_str
    