import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from itertools import groupby

# Constants
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.002
MAX_AUDIO_LEN = 160000  # 10s
MAX_TEXT_LEN = 100
WANDB_PROJECT = "catalan-stt"  # Nom del projecte wandb

# Millor definició dels caràcters pels models CTC
CHARS = ['<blank>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
         'à', 'á', 'è', 'é', 'í', 'ï', 'ò', 'ó', 'ú', 'ü', 'ç', "'", '-']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}

# Classe per preprocessar les dades
class SpeechDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Crear l'extractor de característiques dins del mètode
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_mels": 80, "n_fft": 400, "hop_length": 160}
        )
        
        audio = self.dataset[idx]["audio"]["array"]
        
        # Normalize audio
        audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Convert audio to features
        if len(audio) > MAX_AUDIO_LEN:
            audio = audio[:MAX_AUDIO_LEN]
        
        # Padding si és necessari
        if len(audio) < MAX_AUDIO_LEN:
            padding = torch.zeros(MAX_AUDIO_LEN - len(audio))
            audio = torch.cat([audio, padding])
        
        # Convert to MFCC features - ara creat localment
        features = feature_extractor(audio)
        
        # Text preprocessing
        text = self.dataset[idx]["sentence"].lower()
        text = ''.join([c for c in text if c in CHARS[1:]])
        text_encoded = [CHAR_TO_IDX[c] for c in text][:MAX_TEXT_LEN]
        text_length = len(text_encoded)
        
        return features, torch.tensor(text_encoded), text_length

# Funció per agrupar lots
def collate_fn(batch):
    # Separar característiques, etiquetes i longituds
    features, targets, target_lengths = zip(*batch)
    
    # Apilar característiques
    features = torch.stack(features, 0)
    
    # Trobar la longitud màxima en aquest lot
    max_target_length = max(len(t) for t in targets)
    
    # Emplenar amb zeros les etiquetes fins a la longitud màxima
    padded_targets = []
    for t in targets:
        # Crear tensor de zeros de mida màxima
        padded = torch.zeros(max_target_length, dtype=torch.long)
        # Copiar valors reals
        padded[:len(t)] = t
        padded_targets.append(padded)
    
    # Apilar
    targets = torch.stack(padded_targets, 0)
    
    # Convertir longituds a tensor perque el ctc sapiga la longitud de cada target
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return features, targets, target_lengths

# Model CNN+RNN amb Autoencoder/Decoder
class CNNRNNAutoencoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(CNNRNNAutoencoderModel, self).__init__()
        
        # Part CNN (extractor de característiques) dues capes convolucionals
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
        
        # RNN Encoder (compressió)
        self.rnn_encoder = nn.LSTM(
            hidden_dim, latent_dim,
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.3
        )
        
        # Capa de bottleneck (representació latent)
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(), 
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.Dropout(0.3)
        )
        
        # RNN Decoder (descompressió)
        self.rnn_decoder = nn.LSTM(
            latent_dim * 2, hidden_dim,
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.3 
        )
        
        # Capa de sortida
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):                
        # Part CNN
        x = self.cnn_encoder(x)
        
        # Tornem a la forma (batch, time, features) pels RNNs
        x = x.transpose(1, 2)
        
        # Encoder RNN
        enc_output, _ = self.rnn_encoder(x)
        
        # Bottleneck (processament per cada pas temporal)
        bottleneck_output = self.bottleneck(enc_output)
        
        # Decoder RNN
        dec_output, _ = self.rnn_decoder(bottleneck_output)
        
        # Capa de sortida
        output = self.output_layer(dec_output)
        
        # Log softmax per CTC loss
        return nn.functional.log_softmax(output, dim=2)

# Funció per decodificar les sortides CTC
def ctc_decode(pred_indices):
    # Elimina caràcters repetits consecutius
    collapsed = [k for k, _ in groupby(pred_indices)]
    # Elimina els blanks (índex 0)
    decoded = [IDX_TO_CHAR[i.item()] for i in collapsed if i.item() != 0]
    return ''.join(decoded)