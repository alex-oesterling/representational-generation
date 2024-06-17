

import torch
import numpy as np

from argument import get_args
import data_handler
import networks

from torch.utils.data import DataLoader

from utils import set_seed, getMPR, feature_extraction

import trainer
import utils
import networks

# import wandb
# wandb.login()

def eval(args_):
    
    refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)  
    query_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.query_dataset, args=args)
    
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder)  
    
    print('extract embeddings from the reference distribution')
    refer_embedding = feature_extraction(vision_encoder, refer_loader, args)
    print('extract embeddings from the query distribution')
    query_embedding = feature_extraction(vision_encoder, query_loader, args)

    indices = np.ones(query_embedding.shape[0])

    MPR = getMPR(indices, query_embedding.cpu().numpy(), query_embedding.shape[0], curation_set=refer_embedding.cpu().numpy())

    print(f'MPR: {MPR}')

def train(args_):
    train_loader, val_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)

    # wandb.init(
    #         project='fair_continual',
    #         entity='fair_continual',
    #         name=args.date,
    #         settings=wandb.Settings(start_method="fork")
    # )


    # if args.save_model:
    #     args_, save_dir, log_name = check_dirs(args_)
    # else:
    #     save_dir = None
    #     log_name = None

    # wandb.config.update(args_)

    # Get the required model

    # Train the model

    # wandb.finish()


if __name__ == '__main__':

    args = get_args()    

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    set_seed(args.seed)


    if args.train:
        train(args)
    else:
        eval(args)
