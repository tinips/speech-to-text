import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
import os
import wandb
from torch.multiprocessing import freeze_support
from datasets import load_from_disk

# Importar els mòduls locals
from models import (
    SimpleSpeechDataset, simple_collate_fn, SimpleCNNRNNModel, 
    BATCH_SIZE, EPOCHS, LEARNING_RATE, CHARS
)
from train import train_model, save_checkpoint
from test import evaluate_model

# Control d'ús de wandb
USE_WANDB = True  # Inicialment desactivat

def main():
    global USE_WANDB  # Afegeix aquesta línia
    
    # Carregar els datasets
    print("Carregant els datasets...")
    
    clean_dataset_path_train = "data/clean_catalan_commonvoice_train"
    clean_dataset_path_test = "data/clean_catalan_commonvoice_test"

    try:
        train_dataset_raw = load_from_disk(clean_dataset_path_train)
        test_dataset_raw = load_from_disk(clean_dataset_path_test)
        
        print(f"Dataset d'entrenament carregat: {len(train_dataset_raw)} mostres")
        print(f"Dataset de test carregat: {len(test_dataset_raw)} mostres")
    except Exception as e:
        print(f"Error carregant datasets: {e}")
        return

    # Fixar llavor per reproductibilitat
    random.seed(42)

    # Utilitzar el dataset complet sense limitacions
    train_dataset = SimpleSpeechDataset(train_dataset_raw)
    test_dataset = SimpleSpeechDataset(test_dataset_raw)

    print(f"Utilitzant {len(train_dataset)} mostres per entrenament i {len(test_dataset)} per avaluació")

    # Crear dataloaders amb 6 workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=simple_collate_fn, 
        num_workers=6,
        persistent_workers=True,  # Millora el rendiment amb múltiples workers
        pin_memory=(torch.cuda.is_available())  # Millora el rendiment amb GPU
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=simple_collate_fn, 
        num_workers=6,
        persistent_workers=True,
        pin_memory=(torch.cuda.is_available())
    )

    # Configurar dispositiu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilitzant dispositiu: {device}")

    # Crear model inicial simple
    input_dim = 20  # Menys característiques MFCC
    hidden_dim = 128  # Dimensió oculta més petita
    output_dim = len(CHARS)

    model = SimpleCNNRNNModel(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim
    ).to(device)
 
    # Inicialitzar wandb
    run = None
    if USE_WANDB:
        try:
            run = wandb.init(
                project="catalan-stt", 
                name="catalan_stt_simple_cnnrnn", 
                config={
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "model_type": "Simple RNN",
                    "features": "MFCC",
                    "hidden_dim": hidden_dim
                }
            )
        except Exception as e:
            print(f"Error inicialitzant wandb: {e}")
            USE_WANDB = False

    # Optimitzador i funció de pèrdua
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)  # SGD en lloc d'Adam
    criterion = nn.CTCLoss(blank=0)

    # Entrenament
    print("Iniciant entrenament...")
    train_losses = []
    test_losses = []
    test_cers = []

    try:
        for epoch in range(EPOCHS):
            print(f"Època {epoch+1}/{EPOCHS}")
            
            # Entrenar amb tot el dataset d'entrenament
            train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch, run)
            train_losses.append(train_loss)
            
            # Avaluar amb tot el dataset de test
            test_loss, test_cer = evaluate_model(model, test_loader, criterion, device, epoch, run)
            test_losses.append(test_loss)
            test_cers.append(test_cer)
            
            print(f"Època {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test CER = {test_cer:.4f}")
            
            # Guardar checkpoint
            save_checkpoint(
                model, optimizer, epoch, 
                train_loss, test_loss, test_cer, 
                f"simple_model_checkpoint_epoch_{epoch+1}.pt"
            )
            
        # Resultats finals
        print(f"Entrenament complet. CER final: {test_cers[-1]:.4f} (equivalent a una precisió del {(1-test_cers[-1])*100:.2f}%)")
        
        # Guardar model final
        torch.save(model.state_dict(), "simple_model_final.pt")
        print("Model final guardat correctament")
            
    except Exception as e:
        print(f"Error durant l'entrenament: {e}")
    finally:
        if USE_WANDB and run:
            try:
                wandb.finish()
            except:
                pass

if __name__ == '__main__':
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nEntrenament interromput per l'usuari")
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"Error general: {e}")
        if wandb.run is not None:
            wandb.finish()