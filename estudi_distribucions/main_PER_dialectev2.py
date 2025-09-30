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

from test_PER_epitran import calculate_per_epitran, test_epitran

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

def clean_dataset(ds, max_words=200):
    """Neteja el dataset eliminant mostres buides o massa llargues."""
    ds = ds.filter(lambda x: bool(x['sentence'].strip()))
    ds = ds.filter(lambda x: len(x['sentence'].split()) <= max_words)
    return ds

def build_model(device, checkpoint_filename):
    """Construeix i carrega el model des del checkpoint."""
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
    """Crea un DataLoader per un split específic."""
    split_path = os.path.join(BASE_DATA_DIR, split_name)
    print(f"- Carregant des de disc: {split_path}")
    
    raw = load_from_disk(split_path)
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))
    raw = clean_dataset(raw, max_words=200)
    
    ds = SpeechDataset(raw)
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
        name="per_by_dialecte_epitran",
        config={
            "batch_size": 32,
            "model_checkpoint": "best_model.pt",
            "splits": SPLITS,
            "max_words_test": 200,
            "resample_rate": 16000,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usant dispositiu: {device}")
    wandb.config.device = str(device)

    # Test inicial de epitran
    print("=== Testejant epitran ===")
    test_epitran()
    print("=== Fi del test ===\n")

    # 1) Carrega el model
    print("Carregant el model...")
    model = build_model(device, wandb.config.model_checkpoint)
    print("Model carregat correctament!")

    per_per_split = {}

    # 2) Avaluem cada split i log a wandb
    cache_dir = os.path.join(SCRIPT_DIR, "output_files")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "per_per_split_epitran.json")
    
    if os.path.isfile(cache_file):
        print(f"Carregant PER per split des de cache: {cache_file}")
        with open(cache_file, 'r') as f:
            per_per_split = json.load(f)
        print("Cache carregat correctament!")
        
    else:
        print("No s'ha trobat cache, calculant PER per cada split...")
        
        for i, split in enumerate(SPLITS):
            print(f"\n=== [{i+1}/{len(SPLITS)}] Avaluant split: {split} ===")
            
            try:
                loader = make_loader_for_split(split, wandb.config.batch_size)
                print(f"DataLoader creat amb {len(loader)} batches")
                
                # Usar la nueva función con epitran
                per = calculate_per_epitran(
                    model, 
                    loader, 
                    device
                )
                
                per_per_split[split] = per
                print(f"PER mitjà per {split}: {per:.4f}")
                
                # Log individual a wandb
                wandb.log({f"PER/{split}": per})
                
            except Exception as e:
                print(f"Error avaluant {split}: {e}")
                per_per_split[split] = None
                continue
        
        # Guardar cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(per_per_split, f, indent=2)
            print(f"Cache desat a: {cache_file}")
        except Exception as e:
            print(f"Error desant cache: {e}")

    # Filtrar splits amb errors
    valid_per_splits = {k: v for k, v in per_per_split.items() if v is not None}
    
    if not valid_per_splits:
        print("No hi ha dades vàlides de PER. Sortint...")
        wandb.finish()
        return

    print(f"\n=== Resultats individuals ===")
    for split, per in valid_per_splits.items():
        print(f"{split}: {per:.4f}")

    # 3) Agrupar per dialecte i calcular mitjana
    dialectes = ['balearic', 'central', 'northern', 'northwestern', 'valencian']
    per_by_dialecte = {}
    
    print(f"\n=== Agrupant per dialectes ===")
    for dialecte in dialectes:
        # Buscar tots els splits que comencin amb aquest dialecte
        vals = [per for split, per in valid_per_splits.items() 
                if split.startswith(dialecte) and per is not None]
        
        if vals:
            per_mitjà = float(np.mean(vals))
            per_by_dialecte[dialecte] = per_mitjà
            print(f"{dialecte}: {per_mitjà:.4f} (basaat en {len(vals)} splits)")
            
            # Log a wandb
            wandb.log({f"PER_dialecte/{dialecte}": per_mitjà})
        else:
            print(f"{dialecte}: No hi ha dades vàlides")

    # 4) Estadístiques generals
    if per_by_dialecte:
        millor_dialecte = min(per_by_dialecte.items(), key=lambda x: x[1])
        pitjor_dialecte = max(per_by_dialecte.items(), key=lambda x: x[1])
        per_general = np.mean(list(per_by_dialecte.values()))
        
        print(f"\n=== Estadístiques generals ===")
        print(f"PER general: {per_general:.4f}")
        print(f"Millor dialecte: {millor_dialecte[0]} ({millor_dialecte[1]:.4f})")
        print(f"Pitjor dialecte: {pitjor_dialecte[0]} ({pitjor_dialecte[1]:.4f})")
        
        # Log estadístiques generals
        wandb.log({
            "PER_general": per_general,
            "millor_dialecte_per": millor_dialecte[1],
            "pitjor_dialecte_per": pitjor_dialecte[1],
            "diferencia_dialectes": pitjor_dialecte[1] - millor_dialecte[1]
        })

    # 5) Generar gràfic de PER per dialecte
    if per_by_dialecte:
        # Mapeig de noms més llegibles
        label_map = {
            'balearic': 'Balear',
            'central': 'Central', 
            'northern': 'Nord',
            'northwestern': 'Nord-occidental',
            'valencian': 'Valencià'
        }

        # Ordenar per PER (millor a pitjor)
        data = [(dialecte, per, label_map.get(dialecte, dialecte)) 
                for dialecte, per in per_by_dialecte.items()]
        data.sort(key=lambda x: x[1])  # Ordenar per PER ascendent
        
        codes_ord, pers_ord, labels_ord = zip(*data)

        # Colors diferents per cada dialecte
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(pers_ord))]

        # Crear gràfic horitzontal
        plt.figure(figsize=(12, 8))
        bars = plt.barh(labels_ord, pers_ord, color=colors)
        
        plt.xlabel("PER mitjà (Phoneme Error Rate)", fontsize=12)
        plt.title("Comparativa de PER per dialecte català", fontsize=14, pad=20)
        
        # Afegir valors a les barres
        for bar, valor in zip(bars, pers_ord):
            x = bar.get_width() + 0.001
            y = bar.get_y() + bar.get_height() / 2
            plt.text(x, y, f"{valor:.3f}", va='center', fontsize=11)
        
        # Millorar l'aspecte
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Guardar gràfic
        out_dir = os.path.join(SCRIPT_DIR, "output_files")
        os.makedirs(out_dir, exist_ok=True)
        out_fig = os.path.join(out_dir, "per_by_dialecte_epitran.png")
        
        plt.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
        wandb.log({"per_by_dialecte_plot": wandb.Image(out_fig)})
        
        print(f"\nGràfic desat a: {out_fig}")
        plt.show()
    else:
        print("No hi ha dades vàlides per crear el gràfic")

    # 6) Resum final
    print(f"\n{'='*50}")
    print("RESUM FINAL")
    print(f"{'='*50}")
    print(f"Splits processats: {len(valid_per_splits)}/{len(SPLITS)}")
    print(f"Dialectes amb dades: {len(per_by_dialecte)}/{len(dialectes)}")
    if per_by_dialecte:
        print(f"PER mitjà general: {np.mean(list(per_by_dialecte.values())):.4f}")
    print(f"{'='*50}")

    wandb.finish()
    print("Procés completat!")

if __name__ == "__main__":
    main()