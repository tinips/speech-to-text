from datasets import Audio
from norm_dataset.load_dataset import carregar_dataset
import os


def neteja_dataset(train_dataset, columnes_a_eliminar):
    # 1.5. Eliminar les columnes 'segment' i 'locale'
    print("Eliminant columnes 'segment' i 'locale'...")
    train_dataset = train_dataset.remove_columns(columnes_a_eliminar)
    # 2. Convertir l'àudio al sampling rate objectiu
    print("Convertint l'àudio al sampling rate objectiu...")
    # Directament utilitzem el valor de 16000 Hz com a freqüència estàndard
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    return train_dataset

def main():
    columnes_a_eliminar = ['segment', 'locale']

    # 1. Carregar el dataset de train
    dataset_name = "shields/catalan_commonvoice"
    train_dataset = carregar_dataset(dataset_name, split="validation")
    clean_dataset = neteja_dataset(train_dataset, columnes_a_eliminar)
  

    print(f"Dataset original: {len(train_dataset)} clips")
    print(f"Dataset netejat:  {len(clean_dataset)} clips")

    print(f"Exemple de mostra netejada:\n{clean_dataset[0]}")
    
    return clean_dataset

if __name__ == "__main__":
    clean_dataset = main()
    os.makedirs("data/clean_catalan_commonvoice_validation", exist_ok=True)
    clean_dataset.save_to_disk("data/clean_catalan_commonvoice_validation")


