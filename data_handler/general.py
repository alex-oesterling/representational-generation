import torch
from data_handler.dataset_factory import GenericDataset
import os
from PIL import Image
import numpy as np
    
class General(GenericDataset):

    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)
        self.datapath = self.args.dataset_path

        if self.datapath is None:
            raise ValueError(f"Dataset path is not provided")
        
        self.check_path_validation()
        self.transform = transform
        self.processor = processor

        self.query_dataset = self.args.query_dataset
        self.target_model = self.args.target_model
        self.target_profession = self.args.target_profession

        self.filenames = os.listdir(self.datapath)
        self.filenames = [f for f in self.filenames if f.endswith('.png')]
        self.filenames = np.array(self.filenames)
        filenames_id = [int(f.split('.')[0]) for f in self.filenames]
        filenames_id = np.argsort(filenames_id)
        self.filenames = self.filenames[filenames_id]

        self.profession_set = [self.args.target_profession]
        print('The number of generated samples : ', len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        imagepath = os.path.join(self.datapath, filename)
        image = Image.open(imagepath).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        return image, 0

    def check_path_validation(self):
        folders = self.datapath.split('/')
        target_model = folders[-2]
        trainer = folders[1]
        profession = folders[-1].split('_')[0]

        if not os.path.isdir(self.datapath):
            raise ValueError(f"Dataset path is not valid")

        if self.args.target_profession != profession:
            raise ValueError(f"The profession ({profession}) in the data path and the target profession ({self.args.target_profession}) are not matching")

        if self.args.target_model != target_model:
            raise ValueError(f"The model ({target_model}) in the data path and the target model ({self.args.target_model}) are not matching")
        
        if self.args.trainer != trainer:
            raise ValueError(f"The trainer ({trainer}) in the data path and the trainer ({self.args.trainer}) are not matching")

        if self.args.p_ver == 'v2':
            if 'noadj' not in self.datapath:
                raise ValueError(f"The data path should contain 'noadj' for the p_ver v2")

        args_group_name = "".join([_g[0] for _g in self.args.group])        
        if self.args.trainer != 'scratch':
            group_name = folders[2]
            if args_group_name != group_name:
                raise ValueError(f"The group ({group_name}) in the data path and the group ({args_group_name}) are not matching")
        
