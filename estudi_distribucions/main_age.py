import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import load_from_disk, Audio, concatenate_datasets
import numpy as np
import wandb

from models import (
    SpeechDataset, collate_fn, CNNRNNAutoencoderModel,
    CHARS
)
from test import calcular_cer_final

# Model constants
INPUT_DIM = 40
HIDDEN_DIM = 256
LATENT_DIM = 128
OUTPUT_DIM = len(CHARS)

# Splits disponibles
SPLITS = [
    'balearic_fem', 'balearic_male',
    'central_female', 'central_male',
    'northern_female', 'northern_male',
    'northwestern_female', 'northwestern_male',
    'valencian_female', 'valencian_male'
]

# Grups d'edat vàlids
AGE_GROUPS = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies']

# Carpeta arrel on esta el dataset
BASE_DATA_DIR = r"C:\Users\polte\UNI\3R_CURS\2n_semestre\Deep_learning\Projecte\Rosany"
# Directori de l'script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Funció per netejar un dataset abans de l'avaluació
def clean_dataset(ds, max_words=200):
    ds = ds.filter(lambda x: bool(x['sentence'].strip()))
    ds = ds.filter(lambda x: len(x['sentence'].split()) <= max_words)
    return ds

def build_model(device, checkpoint_filename):
    checkpoint_path = os.path.join(SCRIPT_DIR, checkpoint_filename)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No s'ha trobat el fitxer de pesos: {checkpoint_path}")

    model = CNNRNNAutoencoderModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        output_dim=OUTPUT_DIM
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def make_loader_for_split(split_name, bs):
    split_path = os.path.join(BASE_DATA_DIR, split_name)
    print(f"- Load from disk: {split_path}")
    raw = load_from_disk(split_path)
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))
    raw = clean_dataset(raw, max_words=200)
    ds  = SpeechDataset(raw)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    return loader

def main():
    # Inicialització de wandb
    wandb.init(
        project="catalan_estudis",
        name="cer_by_age",
        config={
            "batch_size": 32,
            "model_checkpoint": "best_model.pt",
            "splits": SPLITS,
            "max_words_test": 200,
            "resample_rate": 16000
        }
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    wandb.config.device = str(device)

    # 1) Carrega el model
    model = build_model(device, wandb.config.model_checkpoint)
    bs = wandb.config.batch_size
    
    # 2) Càlcul del CER per grup d'edat amb un únic passe
    cer_by_age = {}
    for age in AGE_GROUPS:
        print(f"\n--- Avaluant edat: {age} ---")
        ds_list = []
        for split in SPLITS:
            raw = load_from_disk(os.path.join(BASE_DATA_DIR, split))
            raw = raw.cast_column("audio", Audio(sampling_rate=wandb.config.resample_rate))
            raw = clean_dataset(raw)
            # Filtra només les mostres d'aquesta edat
            ds_f = raw.filter(lambda x: x.get('age','') == age)
            if len(ds_f) > 0:
                ds_list.append(ds_f)

        if not ds_list:
            print(f"No hi ha mostres per {age}")
            continue

        # Concatena tots els subdatasets amb aquesta edat
        ds_age = concatenate_datasets(ds_list)


        loader = DataLoader(
            SpeechDataset(ds_age),
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn
        )
        cer, _ = calcular_cer_final(model, loader, device)
        cer_by_age[age] = cer
        wandb.log({f"CER/{age}": cer})
        print(f"CER mitjà per {age}: {cer:.4f}")

    # 3) Generar gràfic
    if cer_by_age:
        plt.figure(figsize=(10, 6))
        
        valid_ages = list(cer_by_age.keys())
        valid_cers = [cer_by_age[a] for a in valid_ages]
        cmap = plt.get_cmap('tab10')  
        colors = [cmap(i) for i in range(len(valid_ages))]
        bars = bar(valid_ages, valid_cers, color=colors)
       
        plt.xlabel('CER mitjà')
        plt.title('CER per grups d\'edat')

        # Afegir valors sobre cada barra mitjançant l'índex
        for bar, valor in zip(bars, valid_cers):
            plt.text(
                bar.get_x() + bar.get_width()/2.,  
                bar.get_height() + 0.001,           
                f'{valor:.3f}', 
                ha='center', va='bottom', fontsize=10
            )
            
        plt.tight_layout()
        
        # Guardem i log el plot a wandb dins 'outputfiles'
        out_dir = os.path.join(SCRIPT_DIR, "output_files")
        os.makedirs(out_dir, exist_ok=True)
        out_fig = os.path.join(out_dir, "cer_by_age.png")
        
        plt.savefig(out_fig, dpi=300, bbox_inches='tight')
        wandb.log({'cer_by_age_plot': wandb.Image(out_fig)})
        print(f"\nGràfic de age desat a {out_fig}")
        plt.show()
    else:
        print("No hi ha dades vàlides per crear el gràfic")

    wandb.finish()

if __name__ == '__main__':
    main()