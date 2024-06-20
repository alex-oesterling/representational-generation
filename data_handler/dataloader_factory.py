
from data_handler.dataset_factory import DatasetFactory

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF


import random 

from transformers import BlipProcessor, BlipModel, BlipForConditionalGeneration, BlipForQuestionAnswering


class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(dataname, args):
        # Make a transform function
        processor = None
        transform = None
        test_transform = None

        if args.vision_encoder == 'BLIP':
            from transformers import BlipProcessor
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        elif args.vision_encoder == 'CLIP':
            import clip
            _, transform = clip.load("ViT-B/32", device= 'cpu')

        else:
            # For CelebA
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            transform = transforms.Compose(
                    [transforms.Resize((256,256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)]
                )
            test_transform = transforms.Compose(
                    [transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)] 
                )

        test_dataset = DatasetFactory.get_dataset(args, dataname, transform, processor, split='test')

        if args.train:       # If training, use the training dataset
            train_dataset = DatasetFactory.get_dataset(args, dataname, transform, processor, split='train')            
            # val_dataset = DatasetFactory.get_dataset(argrs, dataname, transform, processor, split='val')

        def _init_fn(worker_id):
            np.random.seed(int(args.seed))
            
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, worker_init_fn=_init_fn, pin_memory=True)

        if args.train:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)
            # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)
        
            # return train_dataloader, val_dataloader, test_dataloader
            return train_dataloader, test_dataloader
        
        return test_dataloader
