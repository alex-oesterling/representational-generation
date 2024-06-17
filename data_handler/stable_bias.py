import torch
from data_handler.dataset_factory import GenericDataset
from PIL import Image
from datasets import load_dataset

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/image_processing_blip.py
class StableBiasIdentity(GenericDataset):
    race_set = ['African-American',
                'American_Indian',
                'Black',
                'Caucasian',
                'East_Asian',
                'First_Nations',
                'Hispanic',
                'Indigenous_American',
                'Latino',
                'Latinx',
                'Multiracial',
                'Native_American',
                'Pacific_Islander',
                'South_Asian',
                'Southeast_Asian',
                'White',
                'no_ethnicity_specified']
    gender_set = ['man', 'woman', 'non-binary', 'no_gender_specified']

    def __init__(self, transform=None, processor=None):
        # self.dataset = dataset
        path = '/n/holylabs/LABS/calmon_labs/Lab/datasets/stable_bias'
        self.dataset = load_dataset('tti-bias/identities', split='train')
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # image = Image.open(data['image']).convert("RGB")
        image = data['image']
        
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        race = self.race_set.index(data["ethnicity"])
        gender = self.gender_set.index(data["gender"])

        return image, race, gender
    

class StableBiasProfession(GenericDataset):

    def __init__(self, transform=None, processor=None):
        # self.dataset = dataset
        path = '/n/holylabs/LABS/calmon_labs/Lab/datasets/stable_bias'
        self.dataset = load_dataset('tti-bias/professions', split='train')
        self.profession_set = self.dataset['profession']
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # image = Image.open(data['image']).convert("RGB")
        image = data['image']
        
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        profession = self.profession_set.index(data["profession"])
        return image, profession