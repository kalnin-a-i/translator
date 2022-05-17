import torch
import os
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import LinearLR, ConstantLR, ExponentialLR
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
from accelerate import Accelerator
import argparse
from datasets import load_metric
from ..data.dataset import TranslationDataset
from .eval import evaluate
import wandb


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
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'], help='device for training')

    parser.add_argument('--opt', type=str, default='AdamW', choices=['AdamW', 'SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default='2e-5', help='leraning rate') 
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of train epochs')
    parser.add_argument('--sheduler', type=str, default='linear', choices=['linear', 'constant', 'exponential'])

    parser.add_argument('--wandb_key', type=str, default='f9a8d11e45667377c03389e476e8cf67740cee7f')
    parser.add_argument('--data_path', type=str, default='src/data/matlab/', help='path to train data file')
    opt = parser.parse_args()
    return opt


def train(model, tokenizer, opt):
    # preprocess data
    train_dataset = TranslationDataset(os.path.join(opt.data_path, 'train.csv'), tokenizer)
    val_dataset = TranslationDataset(os.path.join(opt.data_path, 'test.csv'), tokenizer)

    # define dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=opt.bs,
    )

    test_loader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=opt.bs,
    )
    # define optimizer
    if opt.opt == 'AdamW':
        optimizer = AdamW(model.parameters(), opt.lr) # add other params
    elif opt.opt == 'Adam':
        optimizer = Adam(model.parameters(), opt.lr) # add other params
    elif opt.opt == 'SGD':
        optimizer = SGD(model.parameters(), opt.lr) # add other params

    # define lr sheduler
    if opt.sheduler == 'linear':
        sheduler = LinearLR(optimizer)
    elif opt.sheduler == 'exponenntial':
        sheduler = ExponentialLR(optimizer)
    elif opt.sheduler == 'constant':
        sheduler == ConstantLR(optimizer)


    # define accelrator
    #define device
    if opt.device == 'cpu':
        print('Using cpu for training, are you sure?')
        accelerator = Accelerator(cpu=True)
    else:
        accelerator = Accelerator()
    
    #prepare accelerator
    model, optimizer, train_loader, test_loader, sheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, sheduler
    )
    # define metric
    metric = load_metric('sacrebleu')

    #wandb 
    #login and init
    wandb.login(key=opt.wandb_key)
    
    wandb.init(
        project='translator',
        )
    wandb.config.update(opt)

    # train loop 
    for epoch in range(opt.epochs):
        
        # Forward and  backward pass
        model.train()
        for batch in tqdm(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
        
        #eval step if need
        if not opt.noval:
            bleu = evaluate(model, accelerator, test_loader, metric, tokenizer)
            wandb.log(
                {
                    'loss' : loss,
                    'bleu' : bleu,
                }
            )
            print(f'Epoch {epoch} finished bleu : {bleu}, loss : {loss}')
        else:
            wandb.log(
                {
                    'loss' : loss,  
                }
            )
            print(f'Epoch {epoch}  loss{loss}')
        
        sheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'Helsinki_cp_{epoch}.pth'))

    if opt.noval:
        bleu = evaluate(model, accelerator, test_loader, metric, tokenizer)
    print(f'Training finished bleu : {bleu}, loss : {loss}')
            
    return model

if __name__ == '__main__':
    opt = get_train_opt()
    print(opt)
