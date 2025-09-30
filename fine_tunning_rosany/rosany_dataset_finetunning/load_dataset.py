from datasets import load_dataset
import os

def carregar_i_guardar(dataset_name, split, output_dir):
    # Carrega el split
    ds = load_dataset(dataset_name, split=split)
    print(f"- Split '{split}' carregat amb èxit: {len(ds)} mostres")
    # Carpeta de sortida dedicada
    path = os.path.join(output_dir, split)
    os.makedirs(path, exist_ok=True)
    ds.save_to_disk(path)
    return ds

if __name__ == "__main__":
    dataset_name = "Rosany/catalan-dataset"
    output_base = "/mnt/c/Users/HP/Downloads/rosany_datasets"

    # Obtenim la llista de splits disponibles
    all_splits = load_dataset(dataset_name, split=None).keys()
    print("Splits disponibles:", all_splits)

    # Diccionari per guardar tots els datasets en memòria si vols
    datasets = {}

    # Iterem per cada split
    for split in all_splits:
        print(f"Carregant split: {split}...")
        ds = carregar_i_guardar(dataset_name, split, output_base)
        datasets[split] = ds

    # Opcional: mostrar un exemple de cada split
    print("\nExemples per split:")
    for split, ds in datasets.items():
        print(f"- {split}: {ds[0]}")
