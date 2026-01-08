
from tokenizer import *
import numpy as np
from tqdm import tqdm
import torch
from PRLmodel import *
from dataloader import *
from loss import *

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PassRLTrainer:
    def __init__(
            self, 
            t1=KBDPasswordTokenizer(), 
            t2=TransTokenizer(), 
            epoch=50, 
            batch_size=4, 
            hidden_size=128, 
            embed_size=200, 
            num_layers=3,
            max_len=16, 
            dropout=0.4, 
            lr=0.001,
            device=DEFAULT_DEVICE,
            weight=0.2):
        self.t1 = t1
        self.t2 = t2
        self.pad_token = self.t2.pad_token_id
        self.model = PRLModel(
            len(self.t1), 
            len(self.t2), 
            pad_token=self.t2.pad_token_id, 
            start_token=self.t2.start_token_id, 
            end_token=self.t2.end_token_id,
            hidden_size=hidden_size, 
            embed_size=embed_size, 
            num_layers=num_layers, 
            maxlen=max_len, 
            dropout=dropout,
            device=device)
        self.max_len = max_len
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.model.set_mode("train")
        self.device = device
        self.model = self.model.to(self.device)
        self.weight = weight
        pass
    
    def train(self, data_path, model_save):
        dataset = prl_dataloader(
            data_path, 
            t1 = self.t1, 
            t2 = self.t2, 
            batch_size=self.batch_size)
        losses = np.full(self.epoch, np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        criterion = CustomBCEWithLogitsLoss()
        
        self.model.train()
        for iter in range(self.epoch):
            batch_loss = []
            with tqdm(dataset, desc="Training", leave=False) as dataset_wrapper:
                for pwds, edits in dataset_wrapper:
                    pwds = pwds.to(self.device)
                    
                    label = self.t2.RL_one_hot_encode(edits,weight=self.weight)
                    edits = edits.to(self.device)
                    label = label.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(pwds, edits)
                    batch_size = edits.size(0)
                    loss = criterion(outputs, label)
                    
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item() / batch_size)
                    avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
                    dataset_wrapper.set_postfix(loss=avg_loss)
            avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
            print(f">>> Epoch: {iter}, Loss: {avg_loss}")
            losses[iter] = avg_loss
        self.save(model_path=model_save)
        return losses
 
    def save(self, model_path):
        torch.save(self.model, model_path)
        print(f">>> Model saved in {model_path}")
        pass
       


def main():


    dataset = "pathtotrain"
    model_save = "model.pt"

    trainer = PassRLTrainer(
        epoch=1, 
        batch_size=128, 
        hidden_size=256, 
        embed_size=256, 
        num_layers=3,
        max_len=16, 
        dropout=0.4, 
        lr=0.01, 
        device=DEFAULT_DEVICE,
        weight=0.1
        )
    trainer.train(dataset, model_save)
    pass

if __name__ == '__main__':
    main()
