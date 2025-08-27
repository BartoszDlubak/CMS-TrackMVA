import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC

class TorchTrainer():
    def __init__(self, lr, weight_decay, device=None, epochs=10, save_freq =5,save_path="runs/checkpoint.pt"):
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epochs = epochs
        self.save_path=save_path
        self.save_freq = save_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loss_fn = WeightedCrossEntropyLoss()
        self.metric = BinaryAUROC().to(device)
        
    def train(self, X_train, X_valid, y_train, y_valid, callbacks=None):
        model = self.model
        train_loader = self.data_wrapper(X_train, y_train)
        valid_loader = self.data_wrapper(X_valid, y_valid)
        
        optimizer = get_optimizer(model, self.lr, self.weight_decay)
        scheduler =  get_scheduler(optimizer, self.epochs)
        train_losses = []; valid_losses = []
        for epoch in range(1, self.epochs+1):
            tqdm_train = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=True)
            train_loss = train_base(model, tqdm_train, self.loss_fn, optimizer, self.device, self.metric)
            
            tqdm_valid = tqdm(valid_loader, desc=f"Epoch {epoch:02d} [Valid]", leave=True)
            valid_loss = eval_base(model, tqdm_valid, self.loss_fn, self.device, self.metric)
            
            if scheduler:
                scheduler.step()
            if epoch % self.save_freq:
                torch.save(model.state_dict(), self.save_path)
            if callbacks:
                for cb in callbacks:
                    cb.on_epoch_end(epoch, train_loss, valid_loss)
                
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        torch.save(model.state_dict(), self.save_path) 
        return train_losses
    
    def evaluate(self, valid_loader):
        model = self.model
        tqdm_valid = tqdm(valid_loader, desc=f"Epoch {1:02d} [Train]", leave=False)
        loss = eval_base(model, tqdm_valid, self.loss_fn, self.device, self.metric)
        return loss
    
    def get_loss(self, X_valid, y_valid):
        model = self.model
        valid_loader = self.data_wrapper(X_valid, y_valid)
        tqdm_valid = tqdm(valid_loader, desc=f"[Getting valid loss]", leave=True)
        loss = eval_base(model, tqdm_valid, self.loss_fn, self.device, self.metric)
        return loss
    
    def set_model(self, model):
        print(model)
        self.model = model.to(self.device)
        
    def set_wrapper(self, data_wrapper):
        self.data_wrapper = data_wrapper
                

def train_base(model, tqdm_loader, criterion, optimizer, device, metric):
    model.train()
    total_loss = 0
    count = 0
    for i, batch in enumerate(tqdm_loader):
        X, y, w = batch
        batch_size = len(X)
        count += batch_size
        X = X.to(device)
        y = y.to(device)
        w = w.to(device) if w is not None else None
        optimizer.zero_grad()
        
        pred = model(X)
        probs = torch.softmax(pred, dim=1)[:, 1]
        metric.update(probs, y.int())
        
        loss = criterion(pred, y, w)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        total_loss += (batch_loss * batch_size)
        tqdm_loader.set_postfix(loss=total_loss / count, roc_auc=metric.compute().item())
    return total_loss / len(tqdm_loader)

    
def eval_base(model, tqdm_loader, criterion, device, metric):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm_loader):
            X, y, w = batch
            batch_size = len(X)
            count += batch_size
            X = X.to(device)
            y = y.to(device)
            w = w.to(device) if w is not None else None
            
            pred = model(X)
            probs = torch.softmax(pred, dim=1)[:, 1]
            metric.update(probs, y.int())
            
            loss = criterion(pred, y, w)
            batch_loss = loss.item()
            total_loss += (batch_loss * batch_size)
            tqdm_loader.set_postfix(loss=total_loss / count, roc_auc=metric.compute().item())        
    return total_loss / len(tqdm_loader)


# ------------ Optimizers and schedulers ------------:

def get_optimizer(model, lr, weight_decay):
    optimizer = torch.optim.Adam(
        model.parameters(),   # parameters to update
        lr=lr,              # learning rate
        weight_decay=weight_decay    # L2 regularization
        )
    return optimizer


def get_scheduler(optimizer, num_epochs):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs   # number of epochs for one cycle
    )
    return scheduler



#### ------Losses------

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, sample_weights=None):
        per_sample_loss = F.cross_entropy(outputs, targets)

        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights
        return per_sample_loss.mean()