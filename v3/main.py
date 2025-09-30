import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import wandb
from torch.multiprocessing import freeze_support
import sys
from datasets import load_from_disk

# Importa els mòduls locals
from models import (
    SpeechDataset, collate_fn, CNNRNNAutoencoderModel, 
    BATCH_SIZE, EPOCHS, LEARNING_RATE, CHARS, WANDB_PROJECT
)
from train import train_model, save_checkpoint
from test import evaluate_model, calcular_cer_final, plot_learning_curves

# Control d'ús de wandb
USE_WANDB = True

def main():
    global USE_WANDB

    # Carregar els datasets
    print("Carregant els datasets...")
    
    clean_dataset_path_train = "data/clean_catalan_commonvoice_train"
    clean_dataset_path_test = "data/clean_catalan_commonvoice_test"

    try:
        train_dataset_raw = load_from_disk(clean_dataset_path_train)
        test_dataset_raw = load_from_disk(clean_dataset_path_test)
        
        print(f"Dataset d'entrenament carregat: {len(train_dataset_raw)} mostres")
        print(f"Dataset de prova carregat: {len(test_dataset_raw)} mostres")
    except Exception as e:
        print(f"Error carregant datasets: {e}")
        return

    # Fixar llavor per reproductibilitat
    random.seed(42)
    
    # Crear datasets complets
    train_dataset_full = SpeechDataset(train_dataset_raw)
    test_dataset_full = SpeechDataset(test_dataset_raw)
    
    # Seleccionar mostres per entrenament i test
    train = len(train_dataset_full) 
    test = len(test_dataset_full) 
    
    train_indices = random.sample(range(len(train_dataset_full)), train)
    test_indices = random.sample(range(len(test_dataset_full)), test)
    
    train_dataset = Subset(train_dataset_full, train_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    print(f"Utilitzant {len(train_dataset)} mostres per entrenament i {len(test_dataset)} per avaluació")

    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=8,  
        persistent_workers=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        num_workers=8,  
        persistent_workers=True, 
        pin_memory=True
    )

    # Configurar dispositiu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilitzant dispositiu: {device}")

    # Crear model
    input_dim = 40  # MFCC features
    hidden_dim = 256
    latent_dim = 128
    output_dim = len(CHARS)

    model = CNNRNNAutoencoderModel(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        latent_dim=latent_dim, 
        output_dim=output_dim
    ).to(device)
 
    # Actualitzar nom a wandb
    run = None
    if USE_WANDB:
        try:
            run = wandb.init(
                project=WANDB_PROJECT, 
                name="catalan_stt_cnn_rnn_autoencoder", 
                config={
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "model_type": "CNN+RNN Autoencoder",
                    "features": "MFCC",
                    "hidden_dim": hidden_dim,
                    "latent_dim": latent_dim
                },
                reinit=True
            )
        except Exception as e:
            print(f"Error inicialitzant wandb: {e}")
            USE_WANDB = False

    # Optimitzador i funció de pèrdua
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0)

    if USE_WANDB and run:
        try:
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"Error amb wandb.watch: {e}")

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
            
            if USE_WANDB and run:
                try:
                    run.log({
                        "epoch": epoch+1,
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "test_cer": test_cer,
                        "accuracy": 1.0 - test_cer
                    })
                except Exception as e:
                    print(f"Error en el logging de wandb: {e}")
            
            # Guardar model checkpoint
            checkpoint_path = f"model_checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(
                model, optimizer, epoch, 
                train_loss, test_loss, test_cer, 
                checkpoint_path
            )
            
            if USE_WANDB and run:
                try:
                    wandb.save(checkpoint_path)
                except Exception as e:
                    print(f"Error guardant checkpoint a wandb: {e}")

        # Mostrar resultats
        print(f"Entrenament complet. CER final: {test_cers[-1]:.4f} (equivalent a una precisió del {(1-test_cers[-1])*100:.2f}%)")

        # Guardar el model final
        torch.save(model.state_dict(), "model_speech_to_text_final.pt")
        print("Model final guardat correctament")

        if USE_WANDB and run:
            wandb.save("model_speech_to_text_final.pt")

        # Visualitzar les corbes d'aprenentatge
        plot_learning_curves(train_losses, test_losses, test_cers, run=run)
        
        # Avaluar el model final
        print("\nAvaluant el model final...")
        cer_final, prediccions = calcular_cer_final(model, test_loader, device)
        
        if USE_WANDB and run:
            # Crear una taula amb les prediccions finals
            taula = wandb.Table(columns=["Original", "Predicció", "CER"])
            for original, prediccio, cer in prediccions:
                taula.add_data(original, prediccio, cer)
            
            run.log({
                "FINAL_TEST_RESULTS": taula, 
                "cer_final": cer_final,
                "precisio_final": (1-cer_final)*100
            })
            
    except Exception as e:
        print(f"Error durant l'entrenament: {e}")
    finally:
        # Tancar wandb correctament
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