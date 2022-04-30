import bs4
from urllib.request import urlopen
from transformers import pipeline

class HtmlTranslator():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = pipeline(task='translation', model=model, tokenizer=tokenizer)
    
    def translate_sentence(self, sentences):
        translated = self.generator(sentences)
        return translated[0]['translation_text']

    def get_main(self, soup):
        #find main section of docs
        main = soup.find('section', id='doc_center_content')
        return main

    def translate_html(self, html):
        soup = bs4.BeautifulSoup(html, 'html.parser')
        main = self.get_main(soup)
        print(main)
        print(main.find_all('h'))

        for elem in main.find_all('p'):
            new_elem = soup.new_tag('p')
            translated_text = self.translate_sentence(elem.get_text())
            new_elem.string = translated_text
            elem.replace_with(new_elem) 
        
        return soup

def dump_to_file(soup):
    with open("output.html", "w", encoding='utf-8') as file:
        file.write(str(soup))


if __name__ == '__main__':
    from ..models.model import get_model
    model, tokenizer = get_model('src\Helsinki_cp_(1)14.pth')
    translator = HtmlTranslator(model, tokenizer)
    url = 'https://www.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html'
    translated = translator.translate_html(html = urlopen(url))
    dump_to_file(translated)

    
