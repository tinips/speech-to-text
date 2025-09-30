import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ——— Monkey-patch para que panphon abra sus CSV en UTF-8 y rutas absolutas ———
import csv, os
import panphon.featuretable as _ft
import panphon

def _read_bases_utf8(self, bases_fn, weights):
    # Convierte rutas relativas a absolutas
    if not os.path.isabs(bases_fn):
        pkg_dir = os.path.dirname(panphon.__file__)
        bases_fn = os.path.join(pkg_dir, bases_fn)
    segments, seg_dict = [], {}
    with open(bases_fn, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        for row in reader:
            symbol = row[0]
            feats = []
            for val in row[1:]:
                if val in ('', '-'):  # no feature
                    feats.append(0)
                elif val == '+':      # feature present
                    feats.append(1)
                else:
                    feats.append(int(val))
            segments.append((symbol, feats))
            seg_dict[symbol] = feats
    return segments, seg_dict, names

_ft.FeatureTable._read_bases = _read_bases_utf8
# —————————————————————————————————————————————————————————————

import unicodedata
import epitran
import editdistance
import torch
from tqdm.auto import tqdm
from models import ctc_decode, IDX_TO_CHAR


def normalize_text(text: str) -> str:
    """
    Normaliza Unicode y pasa a minúsculas.
    """
    text = unicodedata.normalize('NFD', text)
    return text.lower().strip()


def setup_epitran() -> epitran.Epitran:
    """
    Crea la instancia de Epitran para catalán (Latin).
    """
    return epitran.Epitran('cat-Latn')


def text_to_phones_epitran(epi: epitran.Epitran, text: str) -> list[str]:
    """
    Translitera texto catalán a IPA y devuelve lista de tokens.
    """
    norm = normalize_text(text)
    if not norm:
        return []
    ipa = epi.transliterate(norm)
    # Quita marcas de acento
    ipa = ipa.replace('ˈ', '').replace('ˌ', '')
    return ipa.split()


def _calculate_per_epitran(
    epi: epitran.Epitran,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """
    Core de cálculo de PER usando Epitran.
    """
    model.eval()
    total_per, total_samples = 0.0, 0
    with torch.no_grad():
        for features, targets, target_lengths in tqdm(test_loader, desc="PER Epitran"):
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            pred_indices = torch.argmax(outputs, dim=2)
            bs = features.size(0)
            for i in range(bs):
                true_text = "".join(IDX_TO_CHAR[idx.item()] for idx in targets[i][:target_lengths[i]])
                pred_text = ctc_decode(pred_indices[i])
                ref_phones = text_to_phones_epitran(epi, true_text)
                hyp_phones = text_to_phones_epitran(epi, pred_text)
                if not ref_phones and not hyp_phones:
                    per = 0.0
                elif not ref_phones or not hyp_phones:
                    per = 1.0
                else:
                    dist = editdistance.eval(hyp_phones, ref_phones)
                    per = dist / len(ref_phones) if ref_phones else 0.0
                total_per += per
                total_samples += 1
    return (total_per / total_samples) if total_samples > 0 else 0.0


def calculate_per_epitran(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Wrapper para compatibilidad con main:
    ignora 'use_simple' y calcula PER con Epitran.
    """
    epi = setup_epitran()
    return _calculate_per_epitran(epi, model, test_loader, device)


def test_epitran():
    """
    Prueba rápida de transliteración con Epitran.
    """
    epi = setup_epitran()
    samples = [
        "hola món",
        "bon dia",
        "què tal estàs?",
        "gràcies per tot",
        "això és una prova"
    ]
    print("=== Test Epitran ===")
    for text in samples:
        phones = text_to_phones_epitran(epi, text)
        print(f"{text!r} -> {phones}")
    print("===================")

if __name__ == "__main__":
    test_epitran()
