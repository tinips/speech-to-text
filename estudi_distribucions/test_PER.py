import torch
import editdistance
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
from models import ctc_decode, IDX_TO_CHAR
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import unicodedata
import logging
from models import CHARS
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

# dile a phonemizer dónde está la DLL
EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

# un separador que ponga un espacio entre fonemas
SEPARATOR = Separator(phone=' ', word=None)

# Configurar logging para phonemizer
logging.getLogger('phonemizer').setLevel(logging.ERROR)

# Configuración de eSpeak para phonemizer
def setup_phonemizer():
    try:
        backend = EspeakBackend(
            language='ca',
            punctuation_marks=';:,.!?¡¿—…"«»""',
            preserve_punctuation=False,
            with_stress=False,
            language_switch='remove-flags'
        )
        return backend
    except Exception as e:
        print(f"Error configurando eSpeak backend: {e}")
        return None

# Instancia global del backend
PHONEMIZER_BACKEND = setup_phonemizer()

def normalize_text_for_phonemes(text: str) -> str:
    """
    Normaliza texto para procesamiento fonético.
    """
    if not text:
        return ""
    
    # Normalización Unicode
    text = unicodedata.normalize('NFD', text)
    
    # Convertir a minúsculas
    text = text.lower().strip()
    
    # Filtrar solo caracteres válidos (excluyendo <blank>)
    # Mantenemos espacios para que phonemizer funcione correctamente
    valid_chars = set(' ') | set(char for char in CHARS[1:] if char != '<blank>')
    text = ''.join(c for c in text if c in valid_chars)
    
    return text

def text_to_phones_phonemizer(text: str) -> list[str]:
    if not text or not text.strip() or PHONEMIZER_BACKEND is None:
        return []
    normalized = normalize_text_for_phonemes(text)
    if not normalized:
        return []
    try:
        out = PHONEMIZER_BACKEND.phonemize(
            normalized,
            separator=SEPARATOR,
            strip=True,
            preserve_punctuation=False,
            njobs=1
        )
        return out.split()
    except Exception as e:
        print(f"Error phonemizando '{text}': {e}")
        return []


def text_to_phones_simple(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    normalized = normalize_text_for_phonemes(text)
    if not normalized or PHONEMIZER_BACKEND is None:
        return []
    try:
        # llamamos al método phonemize de la instancia, pasándole ya el separador
        out = PHONEMIZER_BACKEND.phonemize(
            normalized,
            separator=SEPARATOR,
            strip=True,
            preserve_punctuation=False
        )
        # devolvemos la cadena resultante partida por espacios
        return out.split()
    except Exception as e:
        print(f"Error en phonemize simple: {e}")
        return []


def calcular_per_final_phonemizer(model, test_loader, device, use_simple=True):
    """
    Calcula PER final usando phonemizer.
    
    Args:
        model: Modelo entrenado
        test_loader: DataLoader de test
        device: Dispositivo (cuda/cpu)  
        use_simple: Si usar la versión simple de phonemize
    """
    model.eval()
    total_per = 0
    total_samples = 0
    phonemize_errors = 0
    
    # Elegir función de phonemización
    phonemize_func = text_to_phones_simple if use_simple else text_to_phones_phonemizer
    
    print(f"Usando función: {'simple' if use_simple else 'backend'}")
    
    with torch.no_grad():
        for features, targets, target_lengths in tqdm(test_loader, desc="Calculando PER"):
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            pred_indices = torch.argmax(outputs, dim=2)
            
            batch_size = features.size(0)
            for i in range(batch_size):
                # Obtener textos
                true_text = "".join([
                    IDX_TO_CHAR[idx.item()] 
                    for idx in targets[i][:target_lengths[i]]
                ])
                pred_text = ctc_decode(pred_indices[i])
                
                # Convertir a fonemas
                ref_phones = phonemize_func(true_text)
                hyp_phones = phonemize_func(pred_text)
                
                # Verificar que ambos tengan fonemas
                if not ref_phones and not hyp_phones:
                    # Ambos vacíos - PER = 0
                    per = 0.0
                elif not ref_phones or not hyp_phones:
                    # Uno vacío - PER = 1
                    per = 1.0
                    phonemize_errors += 1
                else:
                    # Calcular distancia de edición entre fonemas
                    distance = editdistance.eval(hyp_phones, ref_phones)
                    per = distance / len(ref_phones)
                
                total_per += per
                total_samples += 1
    
    avg_per = total_per / total_samples if total_samples > 0 else 0.0
    
    print(f"PER promedio: {avg_per:.4f}")
    print(f"Muestras procesadas: {total_samples}")
    print(f"Errores de phonemización: {phonemize_errors}")
    
    return avg_per

# Función de test para verificar que phonemizer funciona
def test_phonemizer():
    """
    Función para probar que phonemizer funciona correctamente.
    """
    test_texts = [
        "hola món",
        "bon dia",
        "què tal estàs?",
        "gràcies per tot",
        "això és una prova"
    ]
    
    print("=== Test de phonemizer ===")
    print(f"Backend disponible: {PHONEMIZER_BACKEND is not None}")
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        
        # Probar versión simple
        phones_simple = text_to_phones_simple(text)
        print(f"Simple: {phones_simple}")
        
        # Probar versión con backend (si está disponible)
        if PHONEMIZER_BACKEND:
            phones_backend = text_to_phones_phonemizer(text)
            print(f"Backend: {phones_backend}")
    
    print("\n=== Fi del test ===")

# Función mejorada para el script principal
def evaluate_model_with_per(model, test_loader, criterion, device, epoch, run=None):
    """
    Evaluación que incluye tanto CER como PER.
    """
    model.eval()
    total_loss = 0
    total_cer = 0
    total_per = 0
    total_samples = 0
    example_predictions = []
    
    with torch.no_grad():
        for features, targets, target_lengths in tqdm(test_loader):
            features, targets = features.to(device), targets.to(device)
            
            batch_size, seq_len = features.size(0), features.size(2)
            
            # Forward pass
            outputs = model(features)
            
            # CTC loss
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            loss = criterion(outputs.transpose(0, 1), targets, input_lengths, target_lengths)
            total_loss += loss.item() * batch_size
            
            # Decodificar predicciones
            pred_indices = torch.argmax(outputs, dim=2)
            
            # Calcular CER y PER
            for i in range(batch_size):
                pred_text = ctc_decode(pred_indices[i])
                true_text = "".join([IDX_TO_CHAR[idx.item()] for idx in targets[i][:target_lengths[i]]])
                
                # CER
                cer_distance = editdistance.eval(pred_text, true_text)
                cer = cer_distance / max(len(true_text), 1)
                total_cer += cer
                
                # PER
                ref_phones = text_to_phones_simple(true_text)
                hyp_phones = text_to_phones_simple(pred_text)
                
                if not ref_phones and not hyp_phones:
                    per = 0.0
                elif not ref_phones or not hyp_phones:
                    per = 1.0
                else:
                    per_distance = editdistance.eval(hyp_phones, ref_phones) 
                    per = per_distance / len(ref_phones)
                
                total_per += per
                total_samples += 1
                
                # Guardar ejemplos
                if len(example_predictions) < 5:
                    example_predictions.append({
                        "epoch": epoch,
                        "truth": true_text,
                        "prediction": pred_text,
                        "cer": cer,
                        "per": per,
                        "ref_phones": " ".join(ref_phones),
                        "hyp_phones": " ".join(hyp_phones)
                    })
    
    avg_loss = total_loss / total_samples
    avg_cer = total_cer / total_samples
    avg_per = total_per / total_samples
    
    # Log a wandb
    if run is not None:
        run.log({
            f"test_loss_epoch_{epoch}": avg_loss,
            f"test_cer_epoch_{epoch}": avg_cer,
            f"test_per_epoch_{epoch}": avg_per
        })
        
        # Tabla de ejemplos mejorada
        example_data = []
        for example in example_predictions:
            example_data.append([
                example["epoch"],
                example["truth"],
                example["prediction"], 
                example["cer"],
                example["per"],
                example["ref_phones"],
                example["hyp_phones"]
            ])
        
        examples_table = wandb.Table(
            columns=["epoch", "truth", "prediction", "cer", "per", "ref_phones", "hyp_phones"],
            data=example_data
        )
        run.log({f"TEST_EXAMPLES_EPOCH_{epoch+1}": examples_table})
    
    return avg_loss, avg_cer, avg_per

if __name__ == "__main__":
    PHONEMIZER_BACKEND = setup_phonemizer()
    print("Backend disponible:", PHONEMIZER_BACKEND is not None)

    # Lanza aquí el test para ver los fonemas en pantalla
    test_phonemizer()

