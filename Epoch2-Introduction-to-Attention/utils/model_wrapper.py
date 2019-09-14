import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

class ModelWrapper(object):
    
    def train(self, model, data_loader, 
              device='cuda:0', num_epoch=10, learning_rate=1e-3, early_stopping_rounds=5):
        
        # initialization
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        early_stopping_ctr = 0
        best_dev_loss = np.float('inf')
        
        # training loop
        for epoch in range(num_epoch):
            # training and validation
            train_loss, train_acc = self._train_loop(model, data_loader, optimizer)
            dev_loss, dev_acc = self._eval_loop(model, data_loader)
            dev_loss = np.mean(dev_loss)
            
            print("Epoch {}: train-loss: {:.4f}\t train-acc: {:.4f}\t dev-loss: {:.4f}\t dev-acc: {:.4f}".format(
                epoch+1, train_loss, train_acc, dev_loss, dev_acc))
            
            # early stopping
            if best_dev_loss > dev_loss:
                best_dev_loss = dev_loss
                early_stopping_ctr = 0
            else:
                early_stopping_ctr += 1
            
            if early_stopping_ctr >= early_stopping_rounds:
                print("Eopch {}: Early stop.".format(epoch+1))
                break
            
    def _train_loop(self, model, data_loader, optimizer):
        rec_loss, rec_acc = [], []
        model.train()
        
        for x, y in tqdm(data_loader):
            optimizer.zero_grad()
            logits, loss, attention_weights = model(x, y)
            loss.backward()
            optimizer.step()
            
            # logging
            acc = self._calculate_accuracy(logits, y)
            rec_loss.append(loss.cpu().detach().numpy())
            rec_acc.append(acc)
            
        return np.mean(rec_loss), np.mean(rec_acc)
    
    def _eval_loop(self, model, data_loader):
        rec_loss, rec_acc = [], []
        model.eval()
        
        with torch.no_grad():
            for x, y in tqdm(data_loader):
                logits, loss, attention_weights = model(x, y)
                
                # logging
                acc = self._calculate_accuracy(logits, y)
                rec_loss.append(loss.cpu().numpy())
                rec_acc.append(acc)
        return np.mean(rec_loss), np.mean(rec_acc)       
    
    def _calculate_accuracy(self, logits, labels):
        logits = torch.nn.functional.sigmoid(logits)
        acc = ((logits > 0.5) == labels.byte()).float().mean().cpu().numpy()
        return acc        