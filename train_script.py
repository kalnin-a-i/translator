from src.models.train import get_train_opt, train
from src.models.model import get_model

if __name__ == "__main__":
    model, tokenizer = get_model()
    opt = get_train_opt()
    model = train(model, tokenizer, opt)
    