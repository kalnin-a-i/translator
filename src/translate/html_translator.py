import bs4
from urllib.request import urlopen
from transformers import pipeline

class HtmlTranslator():
    def __init__(self, model, tokenizer):
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model#.to(self.device)
        self.tokenizer = tokenizer
        self.generator = pipeline(task='translation', model=model, tokenizer=tokenizer, device=0)
    
    def translate_sentence(self, sentences):
        translated = self.generator(sentences)
        return translated[0]['translation_text']

    def get_main(self, soup):
        #find main section of docs
        main = soup.find('section', id='doc_center_content')
        return main
        
    def replace_nbsp(srelf, soup):
        texts = soup.find_all(text=True)
        for t in texts:
            newtext = t.replace("&nbsp;", "")
            t.replace_with(newtext)
        return soup

    def translate_html(self, html):
        soup = bs4.BeautifulSoup(html, 'html.parser')
        main = self.get_main(soup)

        #translate paragraphs
        for elem in main.find_all('p'):
            new_elem = soup.new_tag('p')
            if (elem.find_parent('div', class_='bibliomixed') or 
                elem.find_previous_sibling('h2', text='See Also') or 
                elem.find_previous_sibling('h3', text='References')):
                continue
            translated_text = self.translate_sentence(elem.get_text())
            new_elem.string = translated_text
            elem.replace_with(new_elem) 

        #translate headers
        for header in ['h1','h2', 'h3']:
            for elem in main.find_all(header):
                new_elem = soup.new_tag(header)
                translated_text = self.translate_sentence(elem.get_text())
                new_elem.string = translated_text
                elem.replace_with(new_elem)

        soup = self.replace_nbsp(soup)
        return soup

def dump_to_file(soup):
    with open("output.html", "w", encoding='utf-8') as file:
        file.write(str(soup))


if __name__ == '__main__':
    from ..models.model import get_model
    import torch
    model, tokenizer = get_model('src/fresh_model.pth')
    translator = HtmlTranslator(model, tokenizer)
    url = 'https://www.mathworks.com/help/deeplearning/gs/create-simple-sequence-classification-network.html'
    translated = translator.translate_html(html = urlopen(url))
    dump_to_file(translated)

    
