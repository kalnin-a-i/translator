FROM python:3

WORKDIR C:\Users\User\Desktop\docs_translate

COPY . ./
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --no-cache-dir -r requirements.txt


CMD [ "python", "train_script.py" ]