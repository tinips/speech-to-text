import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from itertools import groupby

# Constants
BATCH_SIZE = 32  # Més petit que la versió final
EPOCHS = 4       # Menys èpoques d'entrenament
LEARNING_RATE = 0.005  # Learning rate més alt
MAX_AUDIO_LEN = 160000
MAX_TEXT_LEN = 100
WANDB_PROJECT = "catalan-stt"

# Mateix vocabulari que la versió final
CHARS = ['<blank>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
         'à', 'á', 'è', 'é', 'í', 'ï', 'ò', 'ó', 'ú', 'ü', 'ç', "'", '-']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}

# Model inicial simple basat només en RNN
class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNNModel, self).__init__()
        
        # RNN unidireccional simple (GRU)
        self.rnn = nn.GRU(
            input_dim, hidden_dim,
            num_layers=1,         # Només una capa
            bidirectional=False,  # No bidireccional
            batch_first=True
        )
        
        # Capa de sortida simple
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x té forma (batch, features, time)
        # Convertim a (batch, time, features) per RNN
        x = x.transpose(1, 2)
        
        # Passem per RNN
        rnn_output, _ = self.rnn(x)
        
        # Capa de sortida i log softmax
        output = self.output_layer(rnn_output)
        return nn.functional.log_softmax(output, dim=2)

# Classe pel dataset (més simple, sense normalització robusta)
class SimpleSpeechDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=20,  # Menys coeficients MFCC
            melkwargs={"n_mels": 40, "n_fft": 512, "hop_length": 256}  # Paràmetres diferents
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        audio = self.dataset[idx]["audio"]["array"]
        
        # Normalització bàsica
        audio = torch.tensor(audio, dtype=torch.float32)
        if torch.max(torch.abs(audio)) > 0:
            audio = audio / torch.max(torch.abs(audio))
        
        # Truncar si és massa llarg
        if len(audio) > MAX_AUDIO_LEN:
            audio = audio[:MAX_AUDIO_LEN]
        
        # Padding simple
        if len(audio) < MAX_AUDIO_LEN:
            padding = torch.zeros(MAX_AUDIO_LEN - len(audio))
            audio = torch.cat([audio, padding])
        
        # Extracció de característiques
        features = self.feature_extractor(audio)
        
        # Preprocessament de text bàsic
        text = self.dataset[idx]["sentence"].lower()
        text_encoded = []
        for c in text:
            if c in CHAR_TO_IDX:
                text_encoded.append(CHAR_TO_IDX[c])
        
        text_length = len(text_encoded)
        
        return features, torch.tensor(text_encoded), text_length

# Funció collate simple
def simple_collate_fn(batch):
    features, targets, target_lengths = zip(*batch)
    
    # Apilar característiques
    features = torch.stack(features, 0)
    
    # Padding bàsic
    max_target_length = max(len(t) for t in targets)
    padded_targets = []
    for t in targets:
        padded = torch.zeros(max_target_length, dtype=torch.long)
        padded[:len(t)] = t
        padded_targets.append(padded)
    
    # Convertir a tensors
    targets = torch.stack(padded_targets, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return features, targets, target_lengths

# Funció per decodificar les sortides CTC
def ctc_decode(pred_indices):
    # Elimina caràcters repetits consecutius
    collapsed = [k for k, _ in groupby(pred_indices)]
    # Elimina els blanks (índex 0)
    decoded = [IDX_TO_CHAR[i.item()] for i in collapsed if i.item() != 0]
    return ''.join(decoded)