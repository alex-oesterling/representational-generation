

import torch
import numpy as np
import datetime
from argument import get_args
import pickle
import wandb

import sys
import os
os.environ["WANDB_MODE"]="offline"

import data_handler
import networks
import trainer
import retriever
from utils import set_seed, check_log_dir

def main(args):
    # train_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)

    dm = networks.ModelFactory.get_model(modelname=args.target_model, train=args.train)

    _trainer = trainer.TrainerFactory.get_trainer(trainername=args.trainer, model=dm, args=args)
    _trainer.train()

    model =_trainer.model
    # save model
    save_dir = f'trained_models/{args.trainer}'
    check_log_dir(save_dir)

    groupname = [_g[0] for _g in args.trainer_group]
    groupname = "".join(groupname)
    filename = f'{args.date}_{groupname}'
    torch.save(model, os.path.join(save_dir, f'{filename}.pt'))
    
    # Get the required model

    # Train the model


if __name__ == '__main__':

    print(" ".join(sys.argv))

    # check gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Print out additional information when using CUDA
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()

    args = get_args()    

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    set_seed(args.seed)

    now = datetime.datetime.now()
    # Format as 'ddmmyyHMS'
    formatted_time = now.strftime('%H%M')
    if args.date == 'default':
        args.date = now.strftime('%m%d%y')
    args.time = formatted_time

    run = wandb.init(
            project='mpr_generative',
            entity='sangwonjung-Harvard University',
            name=args.date+'_'+formatted_time,
            settings=wandb.Settings(start_method="fork")
    )
    print('wandb mode : ',run.settings.mode)
    
    wandb.config.update(args)

    main(args)
    
    wandb.finish()
