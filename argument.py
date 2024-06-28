import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    
    # For coupmting MPR
    parser.add_argument('--refer-dataset', type=str, default='stable_bias_i')
    parser.add_argument('--query-dataset', type=str, default='stable_bias_p')
    parser.add_argument('--vision-encoder', type=str, default='BLIP', 
                        choices = ['BLIP', 'CLIP', 'PATHS'])
    parser.add_argument('--target-profession', type=str, default='all')
    parser.add_argument('--target-model', type=str, default='SD_14', #required=True, 
                        choices = ['SD_14','SD_2', 'DallE'])
    
    # For retrieving the dataset
    parser.add_argument('--retrieve', default=False, action='store_true', help='retrieve the dataset')
    parser.add_argument('--retriever', type=str, default='mapr', choices=['mapr', 'random','knn', 'random_ratio'])
    parser.add_argument('--functionclass', type=str, default='linear', choices=['linear','dt','nn','l2'], help='functionclass for mapr')
    parser.add_argument('--max-depth', type=int, default=2, help='max depth for decision tree')
    parser.add_argument('--ratio', type=float, default=1, help='ratio for random_ratio retriever')
    parser.add_argument('--k', type=int, default=20, help='the number of retrieved sample')

    # Hyperparameters used for each dataset
    parser.add_argument('--binarize-age', default=True, action='store_false', help='it is used for fairface only')                    

    # Hyperparameters for training
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64)

    # Info for result file names 
    parser.add_argument('--date', type=str, default='061824', help='date when to save')
    parser.add_argument('--save-dir', type=str, default='results/', help='directory to save the results')

    args = parser.parse_args()
    return args
