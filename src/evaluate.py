# Test accuracy
# Confusion matrix
# At least 3 misclassified examples
# (Optional) Precision, Recall, and F1-score


import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# import your modules
from model import SentimentModel
from data import dataModule

    all_preds = []
    all_labels = []
    misclassified_examples = []

    # evaluation loop
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # save misclassified examples
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    example = {
                        "predicted": preds[i].item(),
                        "true": labels[i].item(),
                        "text": batch["text"][i],
                    }
                    misclassified_examples.append(example)


if __name__ == "__main__":
    main()


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

    # compute metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("Test accuracy")
    print(acc)
    print("Confusion matrix")
    print(cm)


    # save misclassified examples
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            example = {
                "predicted": preds[i].item(),
                "true": labels[i].item(),
                "text": batch["text"][i],
            }
            misclassified_examples.append(example)

    print("Three misclassified examples")
    for item in misclassified_examples[:3]:
        print(item)

if __name__ == "__main__":
    main()