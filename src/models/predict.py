from transformers import pipeline 

class Translator():
    def __init__(self, model, tokenizer):
        self.generator = pipeline(task='translation', model=model, tokenizer=tokenizer)

    def translate(self, inputs : str or list):
        outputs = self.generator(inputs)
        result = [item['translation_text'] for item in outputs]
        return result

if __name__ == '__main__':
    from ..models.model import get_model
    model, tokenizer = get_model()
    translator  = Translator(model, tokenizer)
    print(translator.translate('How are you?'))