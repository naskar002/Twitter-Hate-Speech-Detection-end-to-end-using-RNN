import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn as nn
import torch
from torchmetrics import Recall
from Hate_Speech.contant import *

class LSTMModel(pl.LightningModule):
    def __init__(self,max_words,embedding_dim,max_len,optim_name,lr):
        super().__init__()
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.optimizer_name = optim_name
        self.embedding = nn.Embedding(self.max_words,self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size = embedding_dim,batch_first = True,dropout=0)
        self.fc = nn.Linear(embedding_dim,1)
        self.dropout = nn.Dropout(0.3)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        x = x[:,-1,:]
        # x = self.dropout(x)
        x = self.fc(x)
        return x

    def save_hyperparameters(self):
        self.hparams["max_words"] = self.max_words
        self.hparams["embedding_dim"] = self.embedding_dim
        self.hparams["max_len"] = self.max_len
        self.hparams["optim_name"] = self.optimizer_name
        self.hparams["lr"] = self.lr


    def configure_optimizers(self):
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=1e-5)
        elif self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        else:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self,train_batch,batch_idx):
        x,y = train_batch
        y = torch.unsqueeze(y,dim = -1)
        y_pred = self(x)
        loss = self.criterion(y_pred,y)
        self.log("train_loss",loss,prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        y = torch.unsqueeze(y, dim=-1)

        # Compute loss
        loss = self.criterion(y_pred, y)

        # Compute accuracy
        y_pred_class = (torch.sigmoid(y_pred) > 0.5).float()  # Convert logits to class labels
        accuracy = (y_pred_class == y).float().mean()

        # recall_score = recall_metric(y_pred_class, y)

        # Log loss and accuracy
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("val_recall_score", recall_score, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self,batch,batch_idx):
        y_pred = self(batch)
        if torch.sigmoid(y_pred)>0.5:
            return 1
        else:
            return 0
        
class ModelArchitecture:
    def __init__(self):
       pass 
        
    def get_model(self):
        # model = LSTMModel()
        early_stop_callback = EarlyStopping(monitor="val_acc", patience=5, mode="max")
        model = LSTMModel(MAX_WORDS,100,MAX_LEN,OPTIMIZER_NAME,LEARNING_RATE)
        trainer = pl.Trainer(accelerator = "gpu",max_epochs = 20,callbacks = [early_stop_callback])
        return trainer, model


