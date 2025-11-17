import lightning as L
from lightning.pytorch.callbacks import \
ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import \
    WandbLogger, TensorLogger
from transformers import BertTokenizer
import config
from datamodule_IMBD import IMDBDataModule
from model_bilstm import LightningBi_LSTM

import os
import wandb
from dotenv import load_dotenv

load_dotenv()  # loads api_key from .env
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found! Make sure it's set in your .env file.")

# turns the config file into a python dictionary
config_dict = {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("__")
    }

# wandb.init(
#         project="IMBD",
#         config=dict(
#             optimizer="adamw",
#             lr=1e-3,
#             scheduler="plateau",
#             loss="cross_entropy",
#             epochs=100,
#     ),
# )
def main():
    cfg = wandb.config # accesses hyperparameters

    # random seed to reproduce model experiment
    L.seed_everything(config.RANDOM_SEED, workers=True)

    # uses Google's pre-trained bert-base-uncased's tokenizer
    print("loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    # get vocab size to pass to the model
    vocab_size = tokenizer.vocab_size

    print("preparing and initializing the IMBD data module...")
    data_module = IMDBDataModule(
        tokenizer=tokenizer,
        # batch_size=config.BATCH_SIZE,
        batch_size=cfg.batch_size,
        max_length=config.MAX_SEQ_LENGTH
    )
    # passes the bert-base-uncased vocab_size to your LightningBi_LSTM
    model_module = LightningBi_LSTM(vocab_size=vocab_size)

    # saves best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # save the chkpts here
        filename="best_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,         # save the best model
        monitor="val_loss",   # monitor the val_loss
        mode="min"            # choose lowest val_loss
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10
    )
    wandb_logger = WandbLogger(
        project="bi_lstm",  # project name on wandb
        log_model=True,
        config=config_dict # from the config.py file
    )
    trainer = L.Trainer(
            precision="16-mixed",
            max_epochs=config.NUM_EPOCHS,
            callbacks=[checkpoint_callback, early_stop],
            logger=[wandb_logger, logger]
            devices=1,
            accelerator="gpu"
        )

    print("training in process...")
    trainer.fit(model_module, datamodule=data_module)

    print("training done!")
    trainer.test(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()