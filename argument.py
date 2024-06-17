import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    
    # For coupmting MPR
    parser.add_argument('--refer-dataset', type=str, default='stable_bias_i')
    parser.add_argument('--query-dataset', type=str, default='stable_bias_p')
    parser.add_argument('--vision-encoder', type=str, default='BLIP')

    parser.add_argument('--target-model', type=str, default='sd1.4', #required=True, 
                        choices = ['sd1.4', 'sd1.5', 'sd2.0', 'dalle2'])

    # Hyperparameters for training
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()
    return args
