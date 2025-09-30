import torch
import editdistance
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
from models import ctc_decode, IDX_TO_CHAR

# Rutes segons on es tingui instal·lat eSpeak:
ESPEAK_EXE = r"C:\Program Files (x86)\eSpeak\command_line\espeak.exe"
ESPEAK_DATA_DIR = r"C:\Program Files (x86)\eSpeak"

# Funció d'avaluació
def evaluate_model(model, test_loader, criterion, device, epoch, run=None):
    model.eval()
    total_loss = 0
    total_cer = 0
    total_samples = 0
    example_predictions = []
    
    with torch.no_grad():
        for features, targets, target_lengths in tqdm(test_loader):
            features, targets = features.to(device), targets.to(device)
            
            batch_size, seq_len = features.size(0), features.size(2)
            
            # Forward pass
            outputs = model(features)
            
            # Preparar entrades per a CTC loss
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            
            # CTC loss
            loss = criterion(outputs.transpose(0, 1), targets, input_lengths, target_lengths)
            total_loss += loss.item() * batch_size
            
            # Decodificar prediccions (utilitzant funció CTC)
            pred_indices = torch.argmax(outputs, dim=2)
            
            # Calcular CER
            for i in range(batch_size):
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
    
    avg_loss = total_loss / total_samples
    avg_cer = total_cer / total_samples
    
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
        run.log({f"TEST_EXAMPLES_EPOCH_{epoch+1}": examples_table})
    
    return avg_loss, avg_cer

# Funció per calcular el CER final amb tot el dataset de test
def calcular_cer_final(model, test_loader, device):
    model.eval()
    total_cer = 0
    total_samples = 0
    prediccions = []
    
    with torch.no_grad():
        for features, targets, target_lengths in tqdm(test_loader, desc="Calculant CER final"):
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Decodificar prediccions
            pred_indices = torch.argmax(outputs, dim=2)
            
            # Calcular CER
            batch_size = features.size(0)
            for i in range(batch_size):
                predicted_text = ctc_decode(pred_indices[i])
                target_text = "".join([IDX_TO_CHAR[idx.item()] for idx in targets[i][:target_lengths[i]]])
                
                # Calcular CER individual
                cer = editdistance.eval(predicted_text, target_text) / max(len(target_text), 1)
                total_cer += cer
                total_samples += 1
                
                # Guardar prediccions per mostrar
                if len(prediccions) < 20:
                    prediccions.append((target_text, predicted_text, cer))
    
    # Calcular CER mitjà
    cer_final = total_cer / total_samples
    
    # Mostrar exemples de prediccions
    print("\n=== Exemples de prediccions ===")
    for i, (original, prediccio, cer) in enumerate(prediccions[:10]):
        print(f"\nExemple {i+1}:")
        print(f"Original:   {original}")
        print(f"Predicció:  {prediccio}")
        print(f"CER:        {cer:.4f}")
    
    print(f"\n=== CER FINAL: {cer_final:.4f} (Precisió: {(1-cer_final)*100:.2f}%) ===")
    
    return cer_final, prediccions

# Funció per visualitzar corbes d'aprenentatge
def plot_learning_curves(train_losses, test_losses, test_cers, filename='learning_curves.png', run=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss per època')
    plt.xlabel('Època')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_cers, label='Character Error Rate')
    plt.title('CER per època')
    plt.xlabel('Època')
    plt.ylabel('CER (menor és millor)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    
    if run is not None:
        run.log({"learning_curves": wandb.Image(filename)})
    
    return filename