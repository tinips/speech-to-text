import torch
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
from models import ctc_decode, IDX_TO_CHAR

# Funció d'entrenament
def train_model(model, train_loader, optimizer, criterion, device, epoch, run=None):
    model.train()
    total_loss = 0
    processed_samples = 0
    
    for batch_idx, (features, targets, target_lengths) in enumerate(tqdm(train_loader)):
        features, targets = features.to(device), targets.to(device)
        
        batch_size, seq_len = features.size(0), features.size(2)
        processed_samples += batch_size
        
        # Forward pass
        outputs = model(features)
        
        # Preparar entrades per a CTC loss
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        
        # CTC loss
        loss = criterion(outputs.transpose(0, 1), targets, input_lengths, target_lengths)
        
        # Backward i optimització
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        
        # Logging cada 10 batches
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss {loss.item():.4f}")
            
            if run is not None:
                run.log({
                    "epoch": epoch, 
                    "train_batch_loss": loss.item(),
                    "step": epoch * len(train_loader) + batch_idx
                })
    
    avg_loss = total_loss / processed_samples
    return avg_loss

# Guardar checkpoints del model
def save_checkpoint(model, optimizer, epoch, train_loss, test_loss, test_cer, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'test_cer': test_cer
    }, checkpoint_path)
    print(f"Checkpoint guardat a {checkpoint_path}")
