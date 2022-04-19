import torch
import os
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
from accelerate import Accelerator
import argparse
from datasets import load_metric
from ..data.dataset import TranslationDataset
from .eval import evaluate

def get_train_opt():
    '''
    Parse training options from command line

    Returns:
        opt(argparse.Namespace) - argparse.Namespace object with train options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weigths')
    parser.add_argument('--noval', type=bool, default=False, help='validation only after last epoch')
    parser.add_argument('--save_path', type=str, default='models/models/', help='path there to save tuned model')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'], help='device for training')
    
    parser.add_argument('--wandb', type=bool, default=True, help='Use or not Weigths and baises')
   
    parser.add_argument('--data_path', type=str, default='src/data/matlab/', help='path to train data file')
    opt = parser.parse_args()
    return opt

def get_hyp():
    '''
    Parse training hyperparameters from command line

    Returns:
        opt(argparse.Namespace) - argparse.Namespace object with train parameters
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='AdamW', choices=['AdamW', 'SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default='2e-5', help='leraning rate')
    parser.add_argument('--sheduler', type=bool, default=False, help='use linear lr sheduler')    
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of train epochs')
    
    hyp = parser.parse_args()
    return hyp


def train(model, tokenizer, opt, hyp):
    
    #define device
    device = torch.device(opt.device)
    if opt.device == 'cpu':
        print('Using cpu for training, are you sure?')

    # preprocess data
    train_dataset = TranslationDataset(os.path.join(opt.data_path, 'train.csv'))
    val_dataset = TranslationDataset(os.path.join(opt.data_path, 'test.csv'))

    # define dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=hyp.bs,
    )

    test_loader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=hyp.bs,
    )
    # define optimizer
    if hyp.opt == 'AdamW':
        optimizer = AdamW(model.parameters(), hyp.lr) # add other params
    elif hyp.opt == 'Adam':
        optimizer = Adam(model.parameters(), hyp.lr) # add other params
    elif hyp.opt == 'SGD':
        optimizer = SGD(model.parameters(), hyp.lr) # add other params

    # define lr sheduler 
    if hyp.sheduler:
        pass # add sheduler

    # define accelrator
    accelerator = Accelerator()
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    # define metric
    metric = load_metric('sacrebleu')
    print(val_dataset)
    # train loop 
    for epoch in range(hyp.epochs):
        
        # Forward and  backward pass
        model.train()
        for batch in tqdm(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
        
        #eval step if need
        if not opt.noval:
            bleu = evaluate(model, accelerator, test_loader, metric, tokenizer)
        
    return model

if __name__ == '__main__':
    opt = get_train_opt()
    print(opt)