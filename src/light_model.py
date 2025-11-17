import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
import config
from model import Bi_LSTM

class LightningBi_LSTM(L.LightningModule):
  """ LightningModule wrapper for Bi_LSTM """
  def __init__(self, vocab_size: int):
    super().__init__()
    # saves all the __init__ args to the checkpoint Lightning logs
    self.save_hyperparameters() 
    self.learning_rate = config.LEARNING_RATE
    

    # vocab_size come from the bert-based-case tokenizer in 
    # the main training script
    self.model = Bi_LSTM(input_size=vocab_size)

    self.criterion = nn.NLLLoss()

    # logging metrics
    self.train_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=config.NUM_CLASSES
    )
    self.val_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=config.NUM_CLASSES
    )
    self.test_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=config.NUM_CLASSES
    )
    self.test_conf_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=config.NUM_CLASSES
    )

  def forward(self, x):
    """Calls forward pass in the model"""
    return self.model(x)
  
  def _helper_step(self, batch, batch_idx):
    """ Training_step, validation_step, and test_step use 
    this function to get the pred, labels, and loss
    """
    # batch from the IMDBDataModule's collate_fn
    input_ids = batch['input_ids']
    labels = batch['ratings']
    
    # get the logits
    logits = self.forward(input_ids)
    
    # loss calc
    loss = self.criterion(logits, labels)
    
    # get preditions (class index with the highest logit)
    preds = torch.argmax(logits, dim=1)
    
    return loss, preds, labels
  

  def training_step(self, batch, batch_idx):
    """ Does one training step """
    loss, preds, labels = self._helper_step(batch, batch_idx)
    
    # Update and log metrics
    self.train_acc(preds, labels)
    self.log('train_loss', 
             loss, on_step=True, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    self.log('train_acc', 
             self.train_acc, 
             on_step=False, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    
    return loss # needed for backpropagation

  def validation_step(self, batch, batch_idx):
    """ Does one validation step """
    loss, preds, labels = self._helper_step(batch, batch_idx)
    
    # metrics logs and updates
    self.val_acc(preds, labels)
    self.log('val_loss', 
             loss, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    self.log('val_acc', 
             self.val_acc, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)

  def test_step(self, batch, batch_idx):
    """ Does one test step """
    loss, preds, labels = self._helper_step(batch, batch_idx)
    self.test_acc(preds, labels)
    self.test_conf_matrix(preds, labels)
    self.log('test_loss', 
             loss, on_epoch=True, 
             prog_bar=True, 
             logger=True)
    self.log('test_acc', 
             self.test_acc, 
             on_epoch=True, 
             prog_bar=True, 
             logger=True)
    
  def test_epoch_end(self, outputs):
      """ Runs after the test dataloader is done """
      
      # computes the last confusion matrix
      cm = self.test_conf_matrix.compute()
      print(f"Test Confusion Matrix:\n{cm}")
      # log the confusion matrix to wandb
      self.log("test_confusion_matrix", cm)
      # rester the matrix for the next test run
      self.test_conf_matrix.reset()

  def configure_optimizers(self):
    """ Prepares optimizer """
    optimizer = optim.AdamW(
          self.parameters(), 
          lr=self.learning_rate
    )
    return optimizer