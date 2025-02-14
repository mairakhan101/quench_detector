import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

class RNN_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, extra_feature_dim):
        super(RNN_MLP, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 + extra_feature_dim, 128)
        self.output_layer = nn.Linear(128, num_classes)
    
    def forward(self, x, lengths, extra_features):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(x_packed)
        h_combined = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # Bidirectional concat
        combined_features = torch.cat((h_combined, extra_features), dim=-1)
        out = F.relu(self.fc(combined_features))
        out = self.output_layer(out)
        return out

class CNN_MLP(nn.Module):
    def __init__(self, input_size, num_classes, extra_feature_dim):
        super(CNN_MLP, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * (input_size // 2) + extra_feature_dim, 128)
        self.output_layer = nn.Linear(128, num_classes)
    
    def forward(self, x, extra_features):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        combined_features = torch.cat((x, extra_features), dim=-1)
        out = F.relu(self.fc(combined_features))
        out = self.output_layer(out)
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, extra_features, labels):
        self.sequences = sequences
        self.extra_features = extra_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.extra_features[idx], self.labels[idx]

def collate_fn(batch):
    sequences, extra_features, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True)
    extra_features = torch.stack(extra_features)
    labels = torch.tensor(labels)
    return padded_sequences, lengths, extra_features, labels

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for sequences, lengths, extra_features, labels in dataloader:
        sequences, lengths, extra_features, labels = sequences.to(device), lengths.to(device), extra_features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences, lengths, extra_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def plot_tsne(model, dataloader, device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for sequences, lengths, extra_features, label in dataloader:
            sequences, lengths, extra_features = sequences.to(device), lengths.to(device), extra_features.to(device)
            latent_space = model.rnn(sequences, lengths, extra_features)
            embeddings.append(latent_space.cpu().numpy())
            labels.append(label.cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title("t-SNE Visualization of Latent Space")
    plt.show()

def shap_analysis(model, dataloader, device):
    model.eval()
    batch = next(iter(dataloader))
    sequences, lengths, extra_features, _ = batch
    sequences, lengths, extra_features = sequences.to(device), lengths.to(device), extra_features.to(device)
    explainer = shap.Explainer(model, sequences)
    shap_values = explainer(sequences)
    shap.summary_plot(shap_values, sequences.cpu().numpy())
