import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import load_from_disk, Audio
import wandb
import json

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

# Llista de splits a testejar
SPLITS = [
    'balearic_fem', 'balearic_male',
    'central_female', 'central_male',
    'northern_female', 'northern_male',
    'northwestern_female', 'northwestern_male',
    'valencian_female', 'valencian_male'
]

labels_cat = [
    'Balear – dona', 'Balear – home',
    'Central – dona',        'Central – home',
    'Nord – dona',           'Nord – home',
    'Nord-occidental – dona','Nord-occidental – home',
    'Valencià – dona',       'Valencià – home'
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
        name="cer_by_dialecte/genre",
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

    cer_per_split = {}

    # 2) Avaluem cada split i log a wandb
    cache_dir  = os.path.join(SCRIPT_DIR, "output_files")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cer_per_split.json")
    
    if os.path.isfile(cache_file):
        print(f"Carregant CER per split des de cache: {cache_file}")
        with open(cache_file, 'r') as f:
            cer_per_split = json.load(f)
        
    else:
        for split in SPLITS:
            print(f"\n=== Avaluant split: {split} ===")
            loader = make_loader_for_split(split, wandb.config.batch_size)
            cer, _ = calcular_cer_final(model, loader, device)
            cer_per_split[split] = cer
            print(f"CER mitjà per {split}: {cer:.4f}")
            wandb.log({f"CER/{split}": cer})
            
        with open(cache_file, 'w') as f:
            json.dump(cer_per_split, f, indent=2)
        print(f"Cache desat a: {cache_file}")

    # 3) Generem gràfic de barres
    plt.figure(figsize=(10, 6))
    splits = list(cer_per_split.keys())
    cers   = [cer_per_split[s] for s in splits]
    data = list(zip(cers, splits, labels_cat))

    # Ordenem per CER (element 0)
    data.sort(key=lambda x: x[0]) 
    # Desempaquetem de nou
    cers_ord, splits_ord, labels_ord = zip(*data)
    # Colors reposicionats segons l’ordre
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(cers_ord))]

    bars = plt.barh(
        range(len(cers_ord)),
        cers_ord,
        color=colors,
        tick_label=labels_ord
    )
    plt.xlabel("CER mitjà")
    plt.title("Comparativa de CER per dialecte i gènere")
    
    for bar, valor in zip(bars, cers_ord):
        x = bar.get_width()       
        y = bar.get_y() + bar.get_height() / 2
        plt.text(x + 0.001, y,     
                f"{valor:.3f}",   
                va="center",     
                ha="left",        
                fontsize=9)
    
    plt.tight_layout()
    
    # Guardem i log el plot a wandb dins 'outputfiles'
    out_dir = os.path.join(SCRIPT_DIR, "output_files")
    os.makedirs(out_dir, exist_ok=True)
    out_fig = os.path.join(out_dir, "cer_by_dialecte_gender.png")
    
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    wandb.log({"cer_by_dialecte/gender_plot": wandb.Image(out_fig)})
    print(f"\nGràfic de dialecte/gender desat a {out_fig}")
    plt.show()

    wandb.finish()

if __name__ == "__main__":
    main()
