# translator
Model for matlab docs translation

## Prepare Envinronment
```
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113 && pip3 install -r requirments.txt
```
## Train model
```
python3 train_script.py
```
To see training parameters

```
python3 train_script.py --help
```
### Run with Docker
```
docker build -t translator:latest .
```
```
docker run translator:latest pyhton3 train_script.py
```
### HTML translation
To translate html file your need to call translate_html method of class HtmlTranslator located at src/translate/html_translator
