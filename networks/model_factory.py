import torch


import torch


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(modelname):
        if modelname == 'BLIP':
            from transformers import BlipForQuestionAnswering
            network = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        elif modelname == 'CLIP':
            import clip
            network, _ = clip.load("ViT-B/32", device= 'cpu')
        elif modelname == 'SD_14':
            from diffusers import StableDiffusionPipeline
            network = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        elif modelname == 'SD_2':
            from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
            model_id ="stabilityai/stable-diffusion-2-1-base"
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            network = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        else:
            raise NotImplementedError

        return network
