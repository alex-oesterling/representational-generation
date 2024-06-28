import importlib
import torch.utils.data as data
from collections import defaultdict

dataset_dict = {'stable_bias_i' : ['data_handler.stable_bias','StableBiasIdentity'],
                'stable_bias_p' : ['data_handler.stable_bias','StableBiasProfession'],
                'stable_bias_large' : ['data_handler.stable_bias_large','StableBiasProfession'],
                'fairface' : ['data_handler.fairface','FairFace']
               }

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(args, name, transform, processor, split='test'):
        
        if name not in dataset_dict.keys():
            raise Exception('Not allowed dataset')
        
        module = importlib.import_module(dataset_dict[name][0])
        class_ = getattr(module, dataset_dict[name][1])
        
        return class_(args=args, transform=transform, processor=processor, split=split)

class GenericDataset(data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.root = f'/n/holylabs/LABS/calmon_lab/Lab/datasets/'