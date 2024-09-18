import os
from data_handler.dataset_factory import GenericDataset
from PIL import Image
import glob

class CustomDataset(GenericDataset):
    name = 'openface'
    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)        
        self.dataset_path = os.path.join('datasets/custom_mpr_datasets', "_".join(self.args.target_concept.split(" ")))

        self.filepaths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.filepaths += glob.glob(os.path.join(self.dataset_path, ext))
        print(self.filepaths)

        self.processor = processor
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx])

        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(image, return_tensors="pt")
        return image, idx

    def __len__(self):
        return len(self.filepaths)

    def _check_integrity(self):
        # I dont know how to check the integrity of Huggingface datasets so I'm leaving this
        return True