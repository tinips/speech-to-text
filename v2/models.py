import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from itertools import groupby

# Constants
BATCH_SIZE    = 32
EPOCHS        = 4
LEARNING_RATE = 0.001
MAX_AUDIO_LEN = 160000
MAX_TEXT_LEN  = 100
WANDB_PROJECT = "catalan-stt"

# Vocabulari
CHARS = ['<blank>', ' ', 'a','b','c','d','e','f','g','h','i','j','k','l','m',
         'n','o','p','q','r','s','t','u','v','w','x','y','z',
         'à','á','è','é','í','ï','ò','ó','ú','ü','ç',"'",'-']
CHAR_TO_IDX = {c:i for i,c in enumerate(CHARS)}
IDX_TO_CHAR = {i:c for i,c in enumerate(CHARS)}

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

class SimpleCNNRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleCNNRNNModel, self).__init__()
        # 1 capa Conv1d: transformar MFCCs → 128 canals
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # GRU unidireccional
        self.rnn = nn.GRU(
            128, hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, n_mfcc, time) → conv manté (batch, 128, time)
        x = self.conv(x)
        # conv output → (batch, time, 128) per RNN
        x = x.transpose(1,2)
        rnn_out, _ = self.rnn(x)
        out = self.output_layer(rnn_out)
        return nn.functional.log_softmax(out, dim=2)

class SimpleSpeechDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.fe = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=20,
            melkwargs={"n_mels":40,"n_fft":512,"hop_length":256}
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio = torch.tensor(self.dataset[idx]["audio"]["array"], dtype=torch.float32)
        if audio.abs().max()>0:
            audio /= audio.abs().max()
        if audio.numel()>MAX_AUDIO_LEN:
            audio = audio[:MAX_AUDIO_LEN]
        else:
            pad = torch.zeros(MAX_AUDIO_LEN - audio.numel())
            audio = torch.cat([audio, pad])
        feats = self.fe(audio)               # (n_mfcc=20, time)
        text = self.dataset[idx]["sentence"].lower()
        enc  = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX][:MAX_TEXT_LEN]
        return feats, torch.tensor(enc), len(enc)

def simple_collate_fn(batch):
    feats, targets, lengths = zip(*batch)
    feats = torch.stack(feats,0)
    max_len = max(t.numel() for t in targets)
    padded = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i,t in enumerate(targets):
        padded[i,:t.numel()] = t
    return feats, padded, torch.tensor(lengths, dtype=torch.long)

def ctc_decode(pred_indices):
    collapsed = [k for k,_ in groupby(pred_indices)]
    return ''.join(IDX_TO_CHAR[i.item()] for i in collapsed if i.item()!=0)