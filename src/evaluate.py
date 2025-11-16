import lightning as L
from transformers import BertTokenizer
import torch

import config
from datamodule_IMBD import IMDBDataModule
from model_bilstm import LightningBi_LSTM

def main():
    CHECKPOINT_PATH = "checkpoints/OUR_SAVED_MODEL.ckpt" 
    print("loading bert-based-uncased tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_NAME)

    print(f"loading model at {CHECKPOINT_PATH}...")
    model_module = LightningBi_LSTM.load_from_checkpoint(CHECKPOINT_PATH)
    
    print("preparing the data module...")
    data_module = IMDBDataModule(
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_SEQ_LENGTH
    )
    print("initializing the trainer...")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
    )
    print("running the model with the test set...")
    trainer.test(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()