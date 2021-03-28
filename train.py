import logging
import math
import os
import random
import sys
import time
from typing import Tuple
import torch
from torch import cuda
from transformers import AdamW
from torch import Tensor as T
from torch import nn
from dataloader import LcqmcDataset, TaipeiQADataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from models import BiEncoder
from loss import CosineContrastiveLoss, BinaryClassficationLoss
from torch import optim

# logger = logging.getLogger()
# # setup_logger(logger)


def train(model, optimizer, train_loader, val_loader, epochs):
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'--- using device : {device} ---')

    model = model.to(device)
    model.train()


    # criterion = CosineContrastiveLoss()
    # criterion = BinaryClassficationLoss()

    best_loss = 9999.9999
    best_acc = 0.0
    
    for epoch in range(epochs):
        running_loss_val = 0.0
        running_correct_count = 0.0
        running_count = 0.0
        running_acc = 0.0
        for batch_index , data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            ids, mask, token_type_ids = data['ids'].to(device), data['mask'].to(device), data['token_type_ids'].to(device)
            labels = data['label'].to(device)
             
            outputs = model(ids, mask, token_type_ids , labels = labels)
            loss, logits = outputs[:2]
            loss.sum().backward()
            optimizer.step()
            
            # compute the loss
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute acc
            predicts = torch.argmax(logits, -1)
            n_correct = torch.eq(predicts, labels).sum().item()
            running_correct_count += n_correct
            running_count += labels.size()[0]
            running_acc = running_correct_count / running_count

            print(f"\r epoch:{epoch+1} batch:{batch_index+1} train_loss:{running_loss_val:.2f} train_acc:{running_acc:.2f}", end='')        
       
        # validate
        print('\n ---valid---')

        running_loss_val = 0.0
        running_correct_count = 0.0
        running_count = 0.0
        running_acc = 0.0

        for batch_index, data in enumerate(val_loader):
            model.eval()
            ids, mask, token_type_ids = data['ids'].to(device), data['mask'].to(device), data['token_type_ids'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(ids, mask, token_type_ids , labels = labels)
            loss, logits = outputs[:2]

            # compute the loss
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)
            

            # compute acc
            predicts = torch.argmax(logits, -1)
            n_correct = torch.eq(predicts, labels).sum().item()
            running_correct_count += n_correct
            running_count += labels.size()[0]
            running_acc = running_correct_count / running_count

            # log
            print(f"\r valid: batch:{batch_index+1} val_loss:{running_loss_val:.2f} val_acc:{running_acc:.2f}", end='')        

        print('')
        # break

        # saving
        if running_acc > best_acc:
            print('---saving best acc---')
            best_acc = running_acc
            CHECKPOINT_NAME = 'outputs/TaipeiQA/model_best_acc.pt'
            torch.save(model.state_dict(), CHECKPOINT_NAME)

        if running_loss_val < best_loss:
            print('---saving best loss---')
            best_loss = running_loss_val
            CHECKPOINT_NAME = 'outputs/TaipeiQA/model_best_loss.pt'
            torch.save(model.state_dict(), CHECKPOINT_NAME)



        # if (epoch + 1) % 5 == 0:
        #     print('---saving...---')
        #     CHECKPOINT_NAME = 'outputs/TaipeiQA/model' + str(epoch + 1) + '.pt' 
        #     torch.save(model.state_dict(), CHECKPOINT_NAME)

            
    



if __name__ == '__main__':
    # logger.info("Sys.argv: %s", sys.argv)

    ### params setting ### 
    mode = 'train' 
    # path = 'data/lcqmc/'
    path = 'data/TaipeiQA/'
    pretrained_bert_model = 'hfl/chinese-bert-wwm'
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)
    max_len = 128
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 32
    num_workers = 0
    epochs = 30
    lr = 1e-05
    weight_decay = 1e-05
    # taipeiQA label nums
    num_labels = 149
    ### params setting ### 


    train_params = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': num_workers
    }
    val_params = {
        'batch_size': val_batch_size,
        'shuffle': True,
        'num_workers': num_workers
    }
    test_params = {
        'batch_size': test_batch_size,
        'shuffle': True,
        'num_workers': num_workers
    }


    # create dataloader
    train_dataset = TaipeiQADataset('train', path, max_len, tokenizer)
    train_loader = DataLoader(train_dataset, **train_params)

    val_dataset = TaipeiQADataset('dev', path, max_len, tokenizer)
    val_loader = DataLoader(val_dataset, **val_params)

    test_dataset = TaipeiQADataset('test', path, max_len, tokenizer)
    test_loader = DataLoader(test_dataset, **test_params)

    print(f'train data size : {len(train_dataset)}')
    print(f'dev data size : {len(val_dataset)}')
    print(f'test data size : {len(test_dataset)}')

    model = BertForSequenceClassification.from_pretrained(pretrained_bert_model,num_labels = num_labels)
    # optimizer = AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()

    train(
        model,
        optimizer,
        train_loader,
        val_loader,
        epochs
    )



