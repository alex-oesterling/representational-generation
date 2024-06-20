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
        else:
            raise NotImplementedError

        return network
