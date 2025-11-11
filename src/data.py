from torchtext.datasets import IMDB
from torch.utils.data import random_split, DataLoader, ConcatDataset
from transformers import BertTokenizer
import torch
import lightning as L


class IMDBDataModule(L.LighteningDataModule):
  """Downloads and prepares the IMDB Dataset for model
  training and loads the training, validation, and tests sets"""
  def __init__(self, tokenizer: BertTokenizer, batch_size: int, max_length: int, root: str = ".data"):
          super().__init__()
          self.root = root # directory where the IMbD dataset will be saved
          self.tokenizer = tokenizer # BertTokenizer
          self.max_length = max_length # fixed uniform length for every review
          self.batch_size = batch_size # size of the batch for training

  def prepare_data(self):
     return
  
  def setup(self):
     return
  
  def train_dataloader(self):
    """loads the data for training dataset"""
    return DataLoader(
    )
  def val_dataloader(self):
    """loads validation dataset"""
    return DataLoader(
      )
  def test_dataloader(self):
    """loads test dataset"""
    return DataLoader(
      )