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
docker run translator:latest <command>
```
