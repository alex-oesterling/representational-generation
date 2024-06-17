import importlib
import torch.utils.data as data
from collections import defaultdict

dataset_dict = {'stable_bias_i' : ['data_handler.stable_bias','StableBiasIdentity'],
                'stable_bias_p' : ['data_handler.stable_bias','StableBiasProfession'],
               }

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform, processor):
        root = f'/n/holylabs/LABS/calmon_lab/Labs/datasets/{name}'
         
        if name not in dataset_dict.keys():
            raise Exception('Not allowed dataset')
        
        module = importlib.import_module(dataset_dict[name][0])
        class_ = getattr(module, dataset_dict[name][1])
        
        return class_(transform, processor)

class GenericDataset(data.Dataset):
    def __init__(self):
        pass