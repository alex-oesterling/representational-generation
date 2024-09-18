#!/bin/bash
#SBATCH -p seas_gpu 	  # Partition to submit to
#SBATCH -t 0-04:00    # Runtime in D-HH:MM
#SBATCH --gres=gpu:1 # nvidia_a100-sxm4-80gb:1
#SBATCH --mem=32000   # memory (per node)
#SBATCH -o ./slurm_out/slurm-%j.out	# File to which STDOUT will be written

module load python/3
## ITI-GEN installation:
# git clone https://github.com/humansensinglab/ITI-GEN.git
# cd ITI-GEN
# conda env create --name iti-gen --file=environment.yml
source activate iti-gen


concepts=("doctor" "firefighter")

## Construct soft prompts
for concept in ${concepts[@]};
do
    prompt="A photo of a ${concept}"
    echo "$prompt"
    python prepend.py --prompt="a headshot of a person" --attr-list='Male,Skin_tone,Age' --load-model-epoch=19 --prepended-prompt="$prompt" --data-path='/n/holyscratch01/calmon_lab/Users/aoesterling/itigen_data' --ckpt-path='/n/holyscratch01/calmon_lab/Lab/trained_models/itigen/ckpts'
done

## Generate images
## requires the following:

# cd models
# git clone https://github.com/CompVis/stable-diffusion.git
# mv stable-diffusion sd
# mkdir -p sd/models/ldm/stable-diffusion-v1/
# ln -s /n/holyscratch01/calmon_lab/Lab/trained_models/itigen/model.ckpt sd/models/ldm/stable-diffusion-v1/model.ckpt
# cd sd
# pip install -e .
# cd ../..

## this is for gender, skintone, age, if you want to do just gender, replace a_headshot_of_a_person_Male_Skin_tone_Age with a_headshot_of_a_person_Male
for concept in ${concepts[@]};
do
    prompt="A photo of a ${concept}"
    promptpath="prepend_prompt_embedding_${prompt// /_}"
    echo $promptpath
    python generation.py --config='models/sd/configs/stable-diffusion/v1-inference.yaml' --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' --plms --attr-list='Male,Skin_tone,Age' --outdir="../datasets/itigen/GSTA/${concept}" --prompt-path="/n/holyscratch01/calmon_lab/Lab/trained_models/itigen/ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/${promptpath}/basis_final_embed_19.pt" --n_iter=1 --skip_grid --n_samples=5
done