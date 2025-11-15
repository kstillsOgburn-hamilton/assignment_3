import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import BertTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

import config
from model import Bi_LSTM
from data import IMDBDataModule


class LightningBiLSTM(L.LightningModule):
    """
    PyTorch Lightning wrapper for the Bi-LSTM model.
    Handles training, validation, and testing with automatic metric tracking.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = config.EMBEDDING_DIM,
        hidden_size: int = config.HIDDEN_DIM,
        output_size: int = config.NUM_CLASSES,
        num_layers: int = config.LSTM_LAYERS,
        dropout: float = config.DROPOUT_RATE,
        learning_rate: float = config.LEARNING_RATE,
    ):
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Initialize the Bi-LSTM model
        self.model = Bi_LSTM(
            input_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Loss function (CrossEntropyLoss for classification)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Learning rate
        self.learning_rate = learning_rate
        
        # Metrics for tracking performance
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step - called for each batch during training.
        
        Args:
            batch: Dictionary with 'input_ids' and 'ratings'
            batch_idx: Index of the current batch
            
        Returns:
            loss: Training loss for this batch
        """
        # Extract inputs and labels from batch
        input_ids = batch['input_ids']  # Shape: (batch_size, seq_length)
        labels = batch['ratings']        # Shape: (batch_size,)
        
        # Forward pass
        logits = self(input_ids)  # Shape: (batch_size, num_classes)
        
        # Calculate loss
        loss = self.loss_fn(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - called for each batch during validation.
        
        Args:
            batch: Dictionary with 'input_ids' and 'ratings'
            batch_idx: Index of the current batch
            
        Returns:
            loss: Validation loss for this batch
        """
        # Extract inputs and labels
        input_ids = batch['input_ids']
        labels = batch['ratings']
        
        # Forward pass
        logits = self(input_ids)
        
        # Calculate loss
        loss = self.loss_fn(logits, labels)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)
        precision = self.val_precision(preds, labels)
        recall = self.val_recall(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step - called for each batch during testing.
        
        Args:
            batch: Dictionary with 'input_ids' and 'ratings'
            batch_idx: Index of the current batch
        """
        # Extract inputs and labels
        input_ids = batch['input_ids']
        labels = batch['ratings']
        
        # Forward pass
        logits = self(input_ids)
        
        # Calculate loss
        loss = self.loss_fn(logits, labels)
        
        # Get predictions and calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and optional scheduler
        """
        # AdamW optimizer with weight decay for regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # ReduceLROnPlateau: reduce learning rate when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


def main():
    """
    Main training function that sets up the data, model, and trainer.
    """
    print("=" * 70)
    print("IMDB Sentiment Analysis - Bi-LSTM Training")
    print("=" * 70)
    
    # Set random seed for reproducibility
    L.seed_everything(config.RANDOM_SEED)
    print(f"✓ Random seed set to: {config.RANDOM_SEED}")
    
    # Initialize tokenizer
    print(f"\n✓ Loading tokenizer: {config.TOKENIZER_NAME}")
    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {vocab_size:,}")
    
    # Initialize data module
    print(f"\n✓ Preparing IMDB dataset...")
    dm = IMDBDataModule(
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_SEQ_LENGTH
    )
    dm.prepare_data()
    dm.setup()
    
    print(f"  Training samples: {len(dm.train_dataset):,}")
    print(f"  Validation samples: {len(dm.val_dataset):,}")
    print(f"  Test samples: {len(dm.test_dataset):,}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    
    # Initialize model
    print(f"\n✓ Initializing Bi-LSTM model...")
    model = LightningBiLSTM(
        vocab_size=vocab_size,
        embedding_size=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_DIM,
        output_size=config.NUM_CLASSES,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT_RATE,
        learning_rate=config.LEARNING_RATE,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    print(f"\n✓ Setting up training callbacks...")
    
    # ModelCheckpoint: Save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='bilstm-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        verbose=True
    )
    
    # EarlyStopping: Stop training if validation loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    
    # TensorBoard logger for tracking metrics
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name='bilstm_imdb'
    )
    
    # Initialize trainer
    print(f"\n✓ Initializing trainer...")
    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='auto',  # Automatically use GPU if available
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0,  # Prevent exploding gradients
        deterministic=True,  # For reproducibility
    )
    
    # Display training configuration
    print("\n" + "=" * 70)
    print("Training Configuration:")
    print("=" * 70)
    print(f"  Max epochs: {config.NUM_EPOCHS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Optimizer: {config.OPTIMIZER}")
    print(f"  LSTM layers: {config.LSTM_LAYERS}")
    print(f"  Hidden dimension: {config.HIDDEN_DIM}")
    print(f"  Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"  Dropout rate: {config.DROPOUT_RATE}")
    print(f"  Device: {trainer.device_ids}")
    print("=" * 70)
    
    # Start training
    print("\n🚀 Starting training...\n")
    trainer.fit(model, datamodule=dm)
    
    # Training completed
    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print("=" * 70)
    print(f"  Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"  Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    
    # Test the model
    print("\n🧪 Running test evaluation...\n")
    test_results = trainer.test(model, datamodule=dm, ckpt_path='best')
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")
    print("=" * 70)
    
    print("\n✅ All done! Check 'lightning_logs' for TensorBoard visualization.")
    print("   Run: tensorboard --logdir=lightning_logs")
    

if __name__ == "__main__":
    main()
