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
from test import calcular_cer_final, plot_learning_curves
from validation import validate_model, early_stopping_check  

# Control d'ús de wandb
USE_WANDB = True

def entrena_i_avalua(model, train_loader, validation_loader, test_loader, optimizer, criterion, 
                     device, num_epochs=EPOCHS, nom_experiment="complet", run=None):
    train_losses = []
    val_losses = []
    val_cers = []
    
    # Variables per early stopping
    best_val_cer = float('inf')
    patience_counter = 0
    patience = 2
    
    # Iniciar entrenament
    print(f"\nIniciant entrenament {nom_experiment}...")
    
    for epoch in range(num_epochs):
        print(f"Època {epoch+1}/{num_epochs}")
        
        # Entrenar amb tot el dataset d'entrenament
        train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch, run)
        train_losses.append(train_loss)
        
        # Validació amb el validation_loader
        val_loss, val_cer = validate_model(model, validation_loader, criterion, device, epoch, run)
        val_losses.append(val_loss)
        val_cers.append(val_cer)
        
        # Comprovar early stopping
        best_val_cer, patience_counter, stop_training = early_stopping_check(
            val_cer, best_val_cer, patience_counter, patience, model, 
            save_path=f"best_model_{nom_experiment}.pt"
        )
        
        # Mostrar informació
        print(f"Època {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val CER = {val_cer:.4f}")
        
        # Log a wandb
        if USE_WANDB and run:
            try:
                run.log({
                    f"epoch_{nom_experiment}": epoch+1,
                    f"train_loss_{nom_experiment}": train_loss,
                    f"val_loss_{nom_experiment}": val_loss,
                    f"val_cer_{nom_experiment}": val_cer,
                    f"accuracy_{nom_experiment}": 1.0 - val_cer
                })
            except Exception as e:
                print(f"Error en el logging de wandb: {e}")
        
        # Guardar model checkpoint
        checkpoint_path = f"model_checkpoint_{nom_experiment}_epoch_{epoch+1}.pt"
        save_checkpoint(
            model, optimizer, epoch, 
            train_loss, val_loss, val_cer, 
            checkpoint_path
        )
        
        if USE_WANDB and run:
            try:
                wandb.save(checkpoint_path)
            except Exception as e:
                print(f"Error guardant checkpoint a wandb: {e}")
                
        # Early stopping
        if stop_training:
            print(f"Early stopping a l'època {epoch+1}")
            break

    # Actualitzem el missatge per mostrar CER de validació final
    print(f"Entrenament {nom_experiment} complet. CER validació final: {val_cers[-1]:.4f} (precisió: {(1-val_cers[-1])*100:.2f}%)")

    # Guardar el model final
    model_path = f"model_{nom_experiment}_final.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model {nom_experiment} guardat correctament")
 
    if USE_WANDB and run:
        wandb.save(model_path)

    # Carregar el millor model
    print(f"Carregant el millor model {nom_experiment} basat en validació...")
    model.load_state_dict(torch.load(f"best_model_{nom_experiment}.pt"))

    # Mostrar corbes d'aprenentatge
    plot_learning_curves(train_losses, val_losses, val_cers, title=f"Model {nom_experiment}", run=run)
    
    # Avaluar el model final amb test
    print(f"\nAvaluant el model {nom_experiment} final...")
    cer_final, prediccions = calcular_cer_final(model, test_loader, device)
    print(f"CER test final: {cer_final:.4f} (precisió: {(1-cer_final)*100:.2f}%)")
    
    if USE_WANDB and run:
        taula = wandb.Table(columns=["Original", "Predicció", "CER"])
        for original, prediccio, cer in prediccions:
            taula.add_data(original, prediccio, cer)
        
        run.log({
            f"prediccions_finals_{nom_experiment}": taula,
            f"cer_final_{nom_experiment}": cer_final,
            f"precisio_final_{nom_experiment}": (1-cer_final)*100
        })
    
    return model, cer_final

def main():
    global USE_WANDB

    # Carregar els datasets
    print("Carregant els datasets...")
    
    clean_dataset_path_train = "data/clean_catalan_commonvoice_train"
    clean_dataset_path_test = "data/clean_catalan_commonvoice_test"
    clean_dataset_path_validation = "data/clean_catalan_commonvoice_validation"

    try:
        train_dataset_raw = load_from_disk(clean_dataset_path_train)
        test_dataset_raw = load_from_disk(clean_dataset_path_test)
        validation_dataset_raw = load_from_disk(clean_dataset_path_validation)
        
        print(f"Dataset d'entrenament carregat: {len(train_dataset_raw)} mostres")
        print(f"Dataset de prova carregat: {len(test_dataset_raw)} mostres")
        print(f"Dataset de validació carregat: {len(validation_dataset_raw)} mostres")
    except Exception as e:
        print(f"Error carregant datasets: {e}")
        return

    # Fixar llavor per reproductibilitat
    random.seed(42)
    
    # Crear datasets complets
    train_dataset_full = SpeechDataset(train_dataset_raw)
    test_dataset_full = SpeechDataset(test_dataset_raw)
    validation_dataset_full = SpeechDataset(validation_dataset_raw)
    
    # Seleccionar mostres per entrenament, validació i test
    train_size = len(train_dataset_full)
    test_size = len(test_dataset_full)
    validation_size = len(validation_dataset_full)
    
    # Configurar dispositiu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilitzant dispositiu: {device}")

    # Crear model
    input_dim = 40  # MFCC features
    hidden_dim = 256
    latent_dim = 128
    output_dim = len(CHARS)

    # Preguntar a l'usuari quina opció vol
    print("\nSelecciona una opció d'entrenament:")
    print("1. Entrenar model complet amb totes les dades")
    print("2. Entrenar model amb la meitat del dataset")  # Text modificat
    
    opcio = input("\nSelecciona opció (1 o 2): ").strip()
    
    while opcio not in ['1', '2']:
        opcio = input("Opció invàlida. Selecciona 1 o 2: ").strip()
    
    # Inicialitzar wandb
    run = None
    if USE_WANDB:
        try:
            run = wandb.init(
                project=WANDB_PROJECT, 
                name=f"catalan_stt_opcio_{opcio}", 
                config={
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "model_type": "CNN+RNN Autoencoder",
                    "features": "MFCC",
                    "hidden_dim": hidden_dim,
                    "latent_dim": latent_dim,
                    "opcio_entrenament": opcio
                },
                reinit=True
            )
        except Exception as e:
            print(f"Error inicialitzant wandb: {e}")
            USE_WANDB = False

    try:
        if opcio == '1':
            # Opció 1: Entrenar model complet
            train_indices = random.sample(range(train_size), train_size)
            test_indices = random.sample(range(test_size), test_size)
            validation_indices = random.sample(range(validation_size), validation_size)
            
            train_dataset = Subset(train_dataset_full, train_indices)
            test_dataset = Subset(test_dataset_full, test_indices)
            validation_dataset = Subset(validation_dataset_full, validation_indices)
            
            print(f"Utilitzant {len(train_dataset)} mostres per entrenament, {len(validation_dataset)} per validació i {len(test_dataset)} per avaluació")
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                collate_fn=collate_fn, 
                num_workers=8,  
                persistent_workers=True,
                pin_memory=True
            )
            
            validation_loader = DataLoader(
                validation_dataset, 
                batch_size=BATCH_SIZE, 
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
            
            model = CNNRNNAutoencoderModel(input_dim, hidden_dim, latent_dim, output_dim).to(device)
            
            if USE_WANDB and run:
                try:
                    wandb.watch(model, log="all", log_freq=100)
                except Exception as e:
                    print(f"Error amb wandb.watch: {e}")
            
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CTCLoss(blank=0)
            
            entrena_i_avalua(model, train_loader, validation_loader, test_loader, optimizer, 
                             criterion, device, num_epochs=EPOCHS, nom_experiment="complet", run=run)
        
        elif opcio == '2':
            # Opció 2: Entrenar només amb la meitat del dataset
            train_x_dades = train_size // 2 #canviar si volem fer servir x dades
            train_x_indices = random.sample(range(train_size), train_x_dades)
            test_indices = random.sample(range(test_size), test_size)
            validation_indices = random.sample(range(validation_size), validation_size)
            
            train_dataset_half = Subset(train_dataset_full, train_x_indices)
            test_dataset = Subset(test_dataset_full, test_indices)
            validation_dataset = Subset(validation_dataset_full, validation_indices)
            
            print(f"\nUtilitzant {len(train_dataset_half)} mostres per entrenament (meitat del dataset)")
            
            train_loader_half = DataLoader(
                train_dataset_half, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                collate_fn=collate_fn, 
                num_workers=8,  
                persistent_workers=True,
                pin_memory=True
            )
            
            validation_loader = DataLoader(
                validation_dataset, 
                batch_size=BATCH_SIZE, 
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
            
            # Crear model
            model = CNNRNNAutoencoderModel(input_dim, hidden_dim, latent_dim, output_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CTCLoss(blank=0)
            
            if USE_WANDB and run:
                try:
                    wandb.watch(model, log="all", log_freq=100)
                except Exception as e:
                    print(f"Error amb wandb.watch: {e}")
            
            # Entrenar el model amb la meitat del dataset
            _, cer_half = entrena_i_avalua(model, train_loader_half, validation_loader, test_loader, 
                                        optimizer, criterion, device, num_epochs=5,  # Fixat a 5 èpoques
                                        nom_experiment="meitat_dataset", run=run)
            
            print("\n===== RESULTATS =====")
            print(f"Model amb meitat del dataset: CER = {cer_half:.4f} (precisió: {(1-cer_half)*100:.2f}%)")
            print("=====================")
                
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