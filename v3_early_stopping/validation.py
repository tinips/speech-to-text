import torch
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import editdistance
from torch.utils.data import DataLoader

# Importa les funcions i constants necessàries
from models import IDX_TO_CHAR, CHARS, ctc_decode, collate_fn

# Funció per validar el model periòdicament
def validate_model(model, validation_loader, criterion, device, epoch, run=None):
    model.eval()
    total_loss = 0
    total_cer = 0
    total_samples = 0
    example_predictions = []
    
    with torch.no_grad():  # Desactivem el càlcul de gradients
        for features, targets, target_lengths in tqdm(validation_loader, desc=f"Validant època {epoch+1}"):
            features, targets = features.to(device), targets.to(device)
            batch_size, seq_len = features.size(0), features.size(2)
            
            # Forward pass
            outputs = model(features)
            
            # CTC loss
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            loss = criterion(outputs.transpose(0, 1), targets, input_lengths, target_lengths)
            
            # Càlcul del CER
            pred_indices = torch.argmax(outputs, dim=2)
            
            for i in range(batch_size):
                # Decodificació CTC - utilitzant la funció importada de models.py
                pred_text = ctc_decode(pred_indices[i])
                true_text = "".join([IDX_TO_CHAR[idx.item()] for idx in targets[i][:target_lengths[i]]])
                
                # Character Error Rate
                distance = editdistance.eval(pred_text, true_text)
                cer = distance / max(len(true_text), 1)
                total_cer += cer
                total_samples += 1
                
                # Guardar alguns exemples per a visualització
                if len(example_predictions) < 5:
                    example_predictions.append({
                        "epoch": epoch,
                        "truth": true_text,
                        "prediction": pred_text,
                        "cer": cer
                    })
                
            total_loss += loss.item() * batch_size
    
    # Càlcul de mètriques mitjanes
    val_loss = total_loss / total_samples
    val_cer = total_cer / total_samples
    
    # Log exemples a wandb
    if run is not None:
        example_data = []
        for example in example_predictions:
            example_data.append([
                example["epoch"],
                example["truth"], 
                example["prediction"], 
                example["cer"]
            ])
        
        examples_table = wandb.Table(
            columns=["epoch", "truth", "prediction", "cer"],
            data=example_data
        )
        run.log({f"val_examples_epoch_{epoch+1}": examples_table})
    
    print(f"Validació època {epoch+1}: Loss = {val_loss:.4f}, CER = {val_cer:.4f}")
    
    return val_loss, val_cer

def early_stopping_check(val_cer, best_val_cer, patience_counter, patience, model, save_path="best_model.pt"):
    if val_cer < best_val_cer:
        best_val_cer = val_cer
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"Nou millor model guardat amb CER = {val_cer:.4f}")
    else:
        patience_counter += 1
        print(f"No hi ha millora. Paciència: {patience_counter}/{patience}")
        
    stop_training = patience_counter >= patience
    
    return best_val_cer, patience_counter, stop_training