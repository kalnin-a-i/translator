from src.models.train import get_train_opt, train
from src.models.model import get_model
import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())
    model, tokenizer = get_model()
    print(next(model.parameters()).device)
    opt = get_train_opt()
    model = train(model, tokenizer, opt)
