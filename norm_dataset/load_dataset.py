from datasets import load_dataset

def carregar_dataset(dataset_name, split):
    
    #Carreguem el dataset. Si ja està descarregat, utilitzem la versió en memòria cache.
    
    dataset = load_dataset(
        dataset_name,
        split=split
    )
    print(f"Dataset carregat amb èxit: {len(dataset)} mostres")
    return dataset
    
"""
# Carregar el dataset
dataset_name = "shields/catalan_commonvoice"
dataset = carregar_dataset(dataset_name, split="train")
dataset_test = carregar_dataset(dataset_name, split="test")
dataset_validation = carregar_dataset(dataset_name, split="validation")


print(f"{len(dataset)} mostres")
print(f"Columnas disponibles: {dataset.column_names}")
print(f"Exemple de mostra:\n{dataset[0]}")

print(f"Exemple de mostra:\n{dataset_test[0]}")
print(f"Exemple de mostra:\n{dataset_validation[0]}")
"""