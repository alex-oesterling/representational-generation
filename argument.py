import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    
    # For coupmting MPR
    parser.add_argument('--refer-dataset', type=str, default='fairface', choices=['fairface', 'stable_bias_i'])
    parser.add_argument('--query-dataset', type=str, default='CLIP')
    parser.add_argument('--dataset-path', type=str, default=None, help='it is only used when query-dataset is general')
    parser.add_argument('--vision-encoder', type=str, default='CLIP', 
                        choices = ['BLIP', 'CLIP', 'PATHS'])
    parser.add_argument('--target-profession', type=str, default='all')
    parser.add_argument('--target-model', type=str, default='SD_14', #required=True, 
                        choices = ['SD_14','SD_2', 'DallE'])
    parser.add_argument('--group', type=str, nargs='+', default=['gender','age','race'])    
    
    # For retrieving the dataset
    parser.add_argument('--retrieve', default=False, action='store_true', help='retrieve the dataset')
    parser.add_argument('--retriever', type=str, default='mapr', choices=['mapr', 'random','knn', 'random_ratio'])
    parser.add_argument('--functionclass', type=str, default='linear', choices=['linear','dt','nn','l2'], help='functionclass for mapr')
    parser.add_argument('--max-depth', type=int, default=2, help='max depth for decision tree')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio for random_ratio retriever')
    parser.add_argument('--pool-size', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=20, help='the number of retrieved sample')


    # mapr hyperparameters
    parser.add_argument('--cutting_planes', type=int, default=50, help='the number of cutting planes in LP')
    parser.add_argument('--n_rhos', type=int, default=30, help='the number of constraint problems')

    # Hyperparameters used for each dataset
    parser.add_argument('--binarize-age', default=True, action='store_false', help='it is used for fairface only')                    

    # Hyperparameters for training
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--trainer', default='uce', type=str, help='choose the trainer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--method', type=str, default=None, help='method to mitigate unfairness')

    # Info for result file names 
    parser.add_argument('--date', type=str, default='default', help='date when to save')
    parser.add_argument('--save-dir', type=str, default='results/', help='directory to save the results')

    args = parser.parse_args()
    return args
