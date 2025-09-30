import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import load_from_disk, Audio
import numpy as np
import wandb
import json

from models import (
    SpeechDataset, collate_fn, CNNRNNAutoencoderModel,
    CHARS
)
from test import calcular_per_final

# Model constants
INPUT_DIM = 40
HIDDEN_DIM = 256
LATENT_DIM = 128
OUTPUT_DIM = len(CHARS)

# Llista de splits a testejar
SPLITS = [
    'balearic_fem', 'balearic_male',
    'central_female', 'central_male',
    'northern_female', 'northern_male',
    'northwestern_female', 'northwestern_male',
    'valencian_female', 'valencian_male'
]

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
        name="per_by_dialecte",
        config={
            "batch_size": 32,
            "model_checkpoint": "best_model.pt",
            "splits": SPLITS,
            "max_words_test": 200,
            "resample_rate": 16000
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usant dispositiu: {device}")
    wandb.config.device = str(device)

    # 1) Carrega el model
    model = build_model(device, wandb.config.model_checkpoint)

    per_per_split = {}

    # 2) Avaluem cada split i log a wandb
    cache_dir  = os.path.join(SCRIPT_DIR, "output_files")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "per_per_split.json")
    
    if os.path.isfile(cache_file):
        print(f"Carregant PER per split des de cache: {cache_file}")
        with open(cache_file, 'r') as f:
            per_per_split = json.load(f)
        
    else:
        for split in SPLITS:
            print(f"\n=== Avaluant split: {split} ===")
            loader = make_loader_for_split(split, wandb.config.batch_size)
            per = calcular_per_final(model, loader, device)
            per_per_split[split] = per
            print(f"PER mitjà per {split}: {per:.4f}")
            wandb.log({f"PER/{split}": per})
            
        with open(cache_file, 'w') as f:
            json.dump(per_per_split, f, indent=2)
        print(f"Cache desat a: {cache_file}")

    # Agrupar per dialecte i calcular mitjana
    dialectes = ['balearic', 'central', 'northern', 'northwestern', 'valencian']
    per_by_dialecte = {}
    for dialecte in dialectes:
        vals = [p for spl, p in per_per_split.items() if spl.startswith(dialecte)]
        if vals:
            per_by_dialecte[dialecte] = float(np.mean(vals))
            print(f"PER mitjà per {dialecte}: {per_by_dialecte[dialecte]:.4f} (n={len(vals)} splits)")
            wandb.log({f"PER/{dialecte}": per_by_dialecte[dialecte]})
        else:
            print(f"No hi ha dades per al dialecte: {dialecte}")

    # 3) Generar gràfic de PER per dialecte
    if per_by_dialecte:
        label_map = {
            'balearic': 'Balear',
            'central': 'Central',
            'northern': 'Nord',
            'northwestern': 'Nord-occidental',
            'valencian': 'Valencià'
        }

        data = [(d, per, label_map.get(d, d)) for d, per in per_by_dialecte.items()]
        data.sort(key=lambda x: x[1])
        codes_ord, pers_ord, labels_ord = zip(*data)

        cmap   = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(pers_ord))]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(labels_ord, pers_ord, color=colors)
        plt.xlabel("PER mitjà")
        plt.title("Comparativa de PER per dialecte català")

        for bar, valor in zip(bars, pers_ord):
            x = bar.get_width() + 0.002
            y = bar.get_y() + bar.get_height() / 2
            plt.text(x, y, f"{valor:.3f}", va='center', fontsize=10)

        plt.tight_layout()

        out_dir = os.path.join(SCRIPT_DIR, "output_files")
        os.makedirs(out_dir, exist_ok=True)
        out_fig = os.path.join(out_dir, "per_by_dialecte.png")
        
        plt.savefig(out_fig, dpi=300, bbox_inches='tight')
        wandb.log({"per_by_dialecte_plot": wandb.Image(out_fig)})
        print(f"\nGràfic de dialecte desat a {out_fig}")
        plt.show()
    else:
        print("No hi ha dades vàlides per crear el gràfic")

    wandb.finish()

if __name__ == "__main__":
    main()
