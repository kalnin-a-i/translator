from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODEL_CHEKPOINT = 'Helsinki-NLP/opus-mt-en-ru'

def get_model(path_to_tuned=None):
    
    #define model class object
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHEKPOINT)

    #load tuned model if necessary
    if path_to_tuned is not None:
        model.load_state_dict(torch.load(path_to_tuned))
    
    #define model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHEKPOINT)

    return model, tokenizer

if __name__ == '__main__':
    print(get_model())