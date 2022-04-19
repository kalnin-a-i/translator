from statistics import mode
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODEL_CHEKPOINT = 'Helsinki-NLP/opus-mt-en-ru'

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHEKPOINT)

def get_model(path_to_tuned=None):
    '''
    Get pretraned MarianMT model

    Args:
        path_to_tuned(str, optinal) - path to previosly fine-tuned model
    
    Return:
        MarianMT class object with pre-trained or fine-tuned weigths
    '''
    #define model class object
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHEKPOINT)

    #load tuned model if necessary
    if path_to_tuned is not None:
        model.load_state_dict(torch.load(path_to_tuned))

    return model

if __name__ == '__main__':
    print(get_model())