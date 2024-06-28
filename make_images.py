from diffusers import DiffusionPipeline
import torch
import argparse
from utils import set_seed, check_log_dir
import networks
import os
def main():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='SD_14')
    parser.add_argument('--profession', type=str, default='SD_14')

    args = parser.parse_args()

    set_seed(args.seed)

    model = networks.ModelFactory.get_model(modelname=args.model)
    model = model.to("cuda")

    file = open("adjectives.txt", "r") 
    data = file.read() 
    adjectives = data.replace('\n', ' ').split(" ") 
    print(adjectives)
    file.close()

    base_path = f'/n/holyscratch01/calmon_lab/Lab/datasets/{args.model}'
    check_log_dir(base_path)

    professions = [args.profession]
    img_num = 0
    for profession in professions:
        path = os.path.join(base_path, profession)
        check_log_dir(path)
        for adjective in adjectives:
            print(f"Generating images for {adjective} {profession}")
            for i in range(100):
                images = model(prompt=f"Photo portrait of a {adjective} {profession}", num_images_per_prompt=10).images
                for j, image in enumerate(images):
                    image.save(f"{path}/{img_num}.png")
                    img_num += 1
    
    ## save

if __name__ == "__main__":
    main()