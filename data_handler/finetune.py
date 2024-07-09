import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from data_handler.dataset_factory import GenericDataset

class Finetune(GenericDataset):
    def __init__(self, args, transform=None, processor=None, split='val'):
        super().__init__(args=args, split=split)
        self.filepath = os.path.join(self.root, 'finetune')
        self.transform = transform
        self.processor = processor
        print("FINETUNE")
        print(os.path.join(self.filepath, "finetune_label_train.csv"))
        if split == 'train':
            df = pd.read_csv(os.path.join(self.filepath, "finetune_label_train.csv"))
        else:
            df = pd.read_csv(os.path.join(self.filepath, "finetune_label_val.csv"))

        self.image_paths = df['file'].tolist()
        self.labels = df[['gender', 'race']].values

        self.gender_to_idx = {'Male': 0, 'Female': 1}
        self.race_to_idx = {race: idx for idx, race in enumerate(df['race'].unique())}

        self.labels = [
            [self.gender_to_idx[row['gender']], self.race_to_idx[row['race']]]
            for _, row in df.iterrows()
        ]

        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.filepath, self.image_paths[idx])
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# python main.py --refer-dataset finetune --query-dataset finetune --vision-encoder CLIP --retrieve --target-model SD_14 --group gender race --pool-size 1.0 --k 20 --functionclass linear --cutting_planes 50 --n_rhos 30

# python main.py --refer-dataset fairface --query-dataset finetune --vision-encoder CLIP --retrieve --target-model SD_14 --group gender race --pool-size 1.0 --k 20 --functionclass linear --cutting_planes 50 --n_rhos 30

# /n/home07/iislasluz/representational-generation/datasets