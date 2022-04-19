from src.models.train import get_train_opt, get_hyp, train
from src.models.model import get_model
from transformers import AutoTokenizer
MODEL_CHEKPOINT = 'Helsinki-NLP/opus-mt-en-ru'
model = get_model()
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHEKPOINT)
opt = get_train_opt()
hyp = get_hyp()
train(model, tokenizer, opt, hyp)