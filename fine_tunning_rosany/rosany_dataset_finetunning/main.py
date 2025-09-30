import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import wandb
from torch.multiprocessing import freeze_support
from datasets import load_from_disk, concatenate_datasets

# Importa els mòduls locals
from models import (
    SpeechDataset, collate_fn, CNNRNNAutoencoderModel,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, CHARS, WANDB_PROJECT
)
from train import train_model, save_checkpoint
from test import evaluate_model, calcular_cer_final, plot_learning_curves

# Control d'ús de wandb
USE_WANDB = True

# Base dir on tens tots els subfolders de rosany_dataset
BASE_DIR = "/mnt/c/Users/HP/Downloads/rosany_datasets"

def main():
    global USE_WANDB

    # 1) Llista tots els subdirs
    all_subdirs = [
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

    # 2) Paths específicos de València i Balearic
    valencia_dirs = ["valencian_male", "valencian_female"]
    balearic_dirs = ["balearic_male", "balearic_fem"]
    valbal_subdirs = [os.path.join(BASE_DIR, d) for d in valencia_dirs + balearic_dirs]

    # 3) Carregar i concatenar TRAIN sobre **tot** el dataset, però
    #    amb una passada extra de València+Balearic per sobre-mostreig:
    try:
        full_raw     = concatenate_datasets([load_from_disk(p) for p in all_subdirs])
        valbal_raw   = concatenate_datasets([load_from_disk(p) for p in valbal_subdirs])
        # Per sobre-mostrejar, afegim valbal_raw un cop més:
        train_raw    = concatenate_datasets([full_raw, valbal_raw])
        print(f"[TRAIN] total rosany_dataset: {len(full_raw)} + sobre-mostreig {len(valbal_raw)} → {len(train_raw)} mostres")
    except Exception as e:
        print(f"Error carregant TRAIN des de {all_subdirs}: {e}")
        return

    # 4) Carregar i concatenar TEST sobre tot rosany_dataset
    try:
        test_raw = full_raw  # ja tenim full_raw
        print(f"[TEST] rosany_dataset complet: {len(test_raw)} mostres")
    except Exception as e:
        print(f"Error preparant TEST: {e}")
        return

    # 5) Semilla per reproductibilitat
    random.seed(42)
    torch.manual_seed(42)

    # 6) Crear els SpeechDataset i DataLoaders
    train_dataset = SpeechDataset(train_raw)
    test_dataset  = SpeechDataset(test_raw)
    print(f"Ús de {len(train_dataset)} per entrenament i {len(test_dataset)} per prova")

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
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )

    # 7) Configurar dispositiu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usant dispositiu: {device}")

    # 8) Instanciar el model
    input_dim, hidden_dim, latent_dim = 40, 256, 128
    output_dim = len(CHARS)
    model = CNNRNNAutoencoderModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim
    ).to(device)

    # 9) Fine-tuning: carregar pesos previs
    try:
        ckpt = torch.load("best_model.pt", map_location=device)
        model.load_state_dict(ckpt)
        print("Pesos pre-entrenats carregats de best_model.pt")
    except Exception as e:
        print(f"No s'ha pogut carregar best_model.pt: {e}")

    # 10) Inicialitzar wandb
    run = None
    if USE_WANDB:
        try:
            run = wandb.init(
                project=WANDB_PROJECT,
                name="rosany_finetune_oversample_valbal",
                config={
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "model_type": "CNN+RNN Autoencoder (fine-tuning)",
                    "features": "MFCC",
                    "hidden_dim": hidden_dim,
                    "latent_dim": latent_dim,
                    "oversample_splits": valencia_dirs + balearic_dirs
                },
                reinit=True
            )
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"Warning wandb: {e}")
            USE_WANDB = False

    # 11) Optimitzador i criteri
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0)

    # 12) Bucle de fine-tuning
    train_losses, test_losses, test_cers = [], [], []
    print("Iniciant fine-tuning…")
    for epoch in range(EPOCHS):
        print(f" Època {epoch+1}/{EPOCHS}")
        tl = train_model(model, train_loader, optimizer, criterion, device, epoch, run)
        train_losses.append(tl)
        vl, vc = evaluate_model(model, test_loader, criterion, device, epoch, run)
        test_losses.append(vl)
        test_cers.append(vc)
        print(f"   TrainLoss={tl:.4f} | TestLoss={vl:.4f} | TestCER={vc:.4f}")
        if run:
            run.log({
                "epoch": epoch+1,
                "train_loss": tl,
                "test_loss": vl,
                "test_cer": vc,
                "accuracy": 1 - vc
            })
        ckpt_name = f"finetune_ckpt_ep{epoch+1}.pt"
        save_checkpoint(model, optimizer, epoch, tl, vl, vc, ckpt_name)
        if run:
            wandb.save(ckpt_name)

    # 13) Desa el model final
    final_name = "model_speech_to_text_finetunedv1.pt"
    torch.save(model.state_dict(), final_name)
    print(f"Fine-tuning complet. CER final: {test_cers[-1]:.4f}")
    print(f"Model guardat a: {final_name}")
    if run:
        wandb.save(final_name)

    # 14) Corbes i avaluació detallada
    plot_learning_curves(train_losses, test_losses, test_cers, run=run)
    cf, preds = calcular_cer_final(model, test_loader, device)
    print(f"CER final a TEST: {cf:.4f} → {(1-cf)*100:.2f}% accuracy")
    if run:
        table = wandb.Table(columns=["Original","Predicció","CER"])
        for o, p, c in preds:
            table.add_data(o, p, c)
        run.log({"final_predictions": table, "cer_final": cf})
        wandb.finish()

if __name__ == "__main__":
    freeze_support()
    main()
