FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

WORKDIR C:\Users\User\Desktop\docs_translate

COPY . ./
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --no-cache-dir -r requirements.txt


CMD [ "python", "train_script.py" ]
