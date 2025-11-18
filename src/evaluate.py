import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from model import LightningBi_LSTM # used to import our trained model
from datamodule import IMDBDataModule


def main():
    """Peforms an inference test on the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained model
    checkpoint_path = "path_to_your_checkpoint.ckpt"
    model = SentimentModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # load IMDB dataset
    data_module = IMDBDataModule()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    all_preds = [] # our predictions
    all_labels = [] # correct answers/labels
    misclassified_examples = [] # misclassed examples

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

    # compute metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("Test accuracy")
    print(acc)
    print("Confusion matrix")
    print(cm)

    print("Three misclassified examples")
    for item in misclassified_examples[:3]:
        print(item)

if __name__ == "__main__":
    main()