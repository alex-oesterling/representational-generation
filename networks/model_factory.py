import torch


import torch


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(modelname, train=True):
        if modelname == 'BLIP':
            from transformers import BlipForQuestionAnswering
            network = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        elif modelname == 'CLIP':
            import clip
            network, _ = clip.load("ViT-B/32", device= 'cpu')
        elif modelname == 'SD_14' or modelname == 'SD_15':
            from diffusers import StableDiffusionPipeline
            if train:
                network = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")#, torch_dtype=torch.float16)
            else:
                network = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        elif modelname == 'SD_2':
            from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
            model_id ="stabilityai/stable-diffusion-2-1-base"
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            network = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        elif modelname == 'SDXL':
            from diffusers import AutoPipelineForText2Image
            network = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, 
                                                                variant="fp16", use_safetensors=True)
        else:
            raise NotImplementedError

        return network
