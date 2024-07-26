from diffusers import DiffusionPipeline
import torch
import argparse
import numpy as np
from utils import set_seed, check_log_dir
import networks
import os
def main():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='SD_14')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--trainer', type=str, default='scratch')
    parser.add_argument('--professions', type=str, nargs='+', default=['firefighter','CEO','musician'])
    parser.add_argument('--n-generations', type=int, default=10000)
    parser.add_argument('--n-gen-per-iter', type=int, default=10)
    parser.add_argument('--no-adjective', default=False, action='store_true')

    args = parser.parse_args()

    group_idx ={
        'gender' : 0,
        'age' : 1,
        'race' : 2
    }
    group_dic = {
        'gender' : ['male', 'female'],
        'age' : ['young', 'old'],
        'race' : ['East Asian', 'White', 'Latino_Hispanic', 'Southeast Asian', 'Black', 'Indian', 'Middle Eastern']
    }
    
    # for group in args.group:
    #     if group not in group_dic.keys():
    #         raise ValueError(f"Group {group} not considered")

    args = parser.parse_args()

    set_seed(args.seed)

    gen = torch.Generator(device='cuda')
    gen.manual_seed(2)

    if args.model_path is None:
        model = networks.ModelFactory.get_model(modelname=args.model, train=False)
    else:
        if args.trainer not in args.model_path:
            raise ValueError(f"Model name and path are not matching")
        model = torch.load(args.model_path)
        model = model.to(torch.float16)

    model = model.to("cuda")

    if not args.no_adjective:
        file = open("adjectives.txt", "r") 
        data = file.read() 
        adjectives = data.replace('\n', ' ').split(" ") 
        adjectives = [adj+" "  if adj != '' else adj for adj in adjectives]
        adjectives = np.array(adjectives)
        print(adjectives)
        file.close()

    base_path = f'/n/holyscratch01/calmon_lab/Lab/datasets/{args.trainer}'
    if args.trainer != 'scratch':
        # group_name = "".join([g[0].upper() for g in args.group])
        group_name = args.model_path.split("_")[-1].split(".")[0]
        base_path = os.path.join(base_path, group_name)
    base_path = os.path.join(base_path, args.model)
    check_log_dir(base_path)

    for profession in args.professions:
        print(f"Generating images for {profession}")
        path = os.path.join(base_path, profession) if not args.no_adjective else os.path.join(base_path, profession+'_noadj')
        check_log_dir(path)
        img_num = 21000
        while img_num < args.n_generations:
            if img_num % 1000 == 0:
                print(f"Generated {img_num} images")
                
            if not args.no_adjective:
                adj_idx = np.random.choice(a=adjectives.size)
                adjective = adjectives[adj_idx]
                images = model(prompt=f"Photo portrait of a {adjective}{profession}", num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
                for j, image in enumerate(images):
                    image.save(f"{path}/{img_num}.png")
                    img_num += 1
            else:
                images = model(prompt=f"Photo portrait of a {profession}", num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
                for j, image in enumerate(images):
                    image.save(f"{path}/{img_num}.png")
                    img_num += 1

if __name__ == "__main__":
    main()
