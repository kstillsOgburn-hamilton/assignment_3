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
     """Downloads the IMDB dataset the .data directory
     Important: torchtext.dataset provides the dataset in 2 parts - train & test;
     Pytorch won't assign IMBD calls to a vars within prepare_data() """
     # torchtext downloads the data as 2 parts: train and test
    
     IMDB(root=self.data_dir, split="train") # 25k
     IMDB(root=self.data_dir, split="test") # 25k
  
  def setup(self):
    """Seteup the training, validation, and test sets
    Reqs: 70 (train) /15 (validation)/ 15 (test)"""

    import config # holds the hyperparamters for each of src files
    train_set = IMDB(root=self.data_dir, split="train") # torchtext downloads the data as 2 parts: train and test
    test_set = IMDB(root=self.data_dir, split="test")

    # combine the sets
    IMBD_dataset = ConcatDataset([train_set, test_set])

    # full dataset size needed for 70/15/15 calc
    size = len(IMBD_dataset)
    train_set_size = size * config.TRAIN_SPLIT
    test_set_size = size * config.TEST_SPLIT
    val_set_size = size * config.VAL_SPLIT

    self.train_dataset, self.val_dataset, self.test_dataset = random_split(
      IMBD_dataset, 
      [train_set_size, test_set_size, val_set_size],
      generator = torch.Generator().manual_seed(config.RANDOM_SEED)
  )

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