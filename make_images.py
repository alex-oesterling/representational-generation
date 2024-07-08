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
    parser.add_argument('--profession', type=str, default='firefighter')

    args = parser.parse_args()

    set_seed(args.seed)

    gen = torch.Generator(device='cuda')
    gen.manual_seed(1)

    model = networks.ModelFactory.get_model(modelname=args.model)
    model = model.to("cuda")

    file = open("adjectives.txt", "r") 
    data = file.read() 
    adjectives = data.replace('\n', ' ').split(" ") 
    print(adjectives)
    file.close()

    base_path = f'/n/holyscratch01/calmon_lab/Lab/datasets/{args.model}'
    check_log_dir(base_path)

    # professions = [args.profession]
    professions = ['firefighter','CEO','musician']
    # professions=['CEO']
    for profession in professions:
        path = os.path.join(base_path, profession, 'no_adjective')
        check_log_dir(path)
        img_num = 0        
        # for adjective in adjectives:
        #     if profession == 'firefighter' and adjective in ['ambitious','assertive','committed','compassionate']:
        #         img_num += 1000
        #         continue
        #     print(f"Generating images for {adjective} {profession}")
        #     # if img_num<15000:
        #         # img_num += 1000
        #         # continue
        #     for i in range(100):
        #         images = model(prompt=f"Photo portrait of a {adjective} {profession}", num_images_per_prompt=10, gen=gen).images
        #         for j, image in enumerate(images):
        #             image.save(f"{path}/{img_num}.png")
        #             img_num += 1

        print(f"Generating images for {profession}")
        # if img_num<15000:
            # img_num += 1000
            # continue
        for i in range(2100):
            images = model(prompt=f"Photo portrait of a {profession}", num_images_per_prompt=10, gen=gen).images
            for j, image in enumerate(images):
                image.save(f"{path}/{img_num}.png")
                img_num += 1

    
    ## save

if __name__ == "__main__":
    main()
