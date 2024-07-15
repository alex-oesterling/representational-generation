import torch
import os
from data_handler.dataset_factory import GenericDataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import torchvision
import pandas as pd
    
class FairFace(GenericDataset):
    
    def __init__(self, transform=None, processor=None, **kwargs):
    # transform=torchvision.transforms.ToTensor(), embedding_model=None, binarize_age=True):
        GenericDataset.__init__(self, **kwargs)
        self.datapath = os.path.join(self.root, 'tmp_fairface')
        self.processor = processor
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.labeltags = [
            "gender",
            "age",
        ]

        if self.split=='train':
            df = pd.read_csv(self.datapath + "/fairface_label_train.csv")
        else:
            df = pd.read_csv(self.datapath + "/fairface_label_val.csv")

        self.race_to_idx = {}
        for i, race in enumerate(df.race.unique()):
            self.labeltags.append(race)
            self.race_to_idx[race] = i

        self.gender_to_idx = {
            'Male': 0,
            'Female': 1
        }

        self.age_to_idx = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8
        }

        one_hot = torch.nn.functional.one_hot(torch.tensor([self.race_to_idx[race] for race in df.race])).numpy()
        gender_idx = [self.gender_to_idx[gen] for gen in df.gender]
        age_idx = [self.age_to_idx[age] for age in df.age]

        if self.args.binarize_age:
            age_idx = [int(ag>4) for ag in age_idx]
        ## labels is [gender_binary, age_categorical, race_one_hot]

        self.labels = []
        for i in range(len(gender_idx)):
            self.labels.append([gender_idx[i], age_idx[i]] + list(one_hot[i]))

        self.labels = torch.tensor(self.labels)

        # construct_path = lambda x: os.path.join(self.datapath, x)
        self.img_paths = df.file.to_list()

        print(self.labeltags)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if self.embedding_model is not None:
        #     path = self.img_paths[idx]
        #     image_id = path.split(".")[0]
        #     embeddingpath = os.path.join(self.datapath, self.embedding_model, image_id+".pt")
        #     return torch.load(embeddingpath), self.labels[idx]
        
        path = os.path.join(self.datapath, self.img_paths[idx])
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        
        return image, self.labels[idx]
