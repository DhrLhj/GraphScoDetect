import os
import re
import math
import numpy as np
import pandas as pd
import sklearn.preprocessing
from scipy.signal import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================
PATH = 'data2'
TRAIN_DIR = 'train01zhee'
TEST_DIR = 'test01zhee'
RESAMPLE_LEN = 500
SEGMENT_LEN = 25   # 500 / 25 = 20 segments
BATCH_SIZE = 8
PRETRAIN_EPOCHS = 20
TRAIN_EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
LAMBDA_INTER = 0.5
GAMMA_CLASS = 1.0
TEMPERATURE = 0.1
INTRA_MARGIN = 1.0
NORMAL_LABEL = 0   # assume class 0 is healthy / normal
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_label_from_filename(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r'(\d+)$', stem)
    if m:
        return int(m.group(1))
    # fall back to legacy rule used in the original notebook: filename[-5]
    m2 = re.search(r'(\d)(?=\.[^.]+$)', filename)
    if m2:
        return int(m2.group(1))
    raise ValueError(f'Cannot parse label from filename: {filename}')


def parse_subject_id_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    # remove trailing label token if available
    stem = re.sub(r'([_\-]?\d+)$', '', stem)
    if stem:
        return stem
    return os.path.splitext(os.path.basename(filename))[0]


def standardize_per_channel(x: np.ndarray) -> np.ndarray:
    # x: [T, C]
    return sklearn.preprocessing.scale(x, axis=0)


def build_label_mapping(root_path: str):
    labels = []
    for split_dir in [TRAIN_DIR, TEST_DIR]:
        split_path = os.path.join(root_path, split_dir)
        for filepath, _, filenames in os.walk(split_path):
            for file in filenames:
                if file.startswith('~$'):
                    continue
                if not file.lower().endswith(('.xls', '.xlsx')):
                    continue
                try:
                    labels.append(parse_label_from_filename(file))
                except Exception:
                    continue
    labels = sorted(set(labels))
    if not labels:
        raise RuntimeError('No valid labels found. Please check file names.')
    return {lab: idx for idx, lab in enumerate(labels)}


class SensorSegmentDataset(Dataset):
    def __init__(self, train: bool, path: str = PATH, resample_len: int = RESAMPLE_LEN,
                 segment_len: int = SEGMENT_LEN, label_map=None):
        self.train = train
        self.root = os.path.join(path, TRAIN_DIR if train else TEST_DIR)
        self.resample_len = resample_len
        self.segment_len = segment_len
        self.label_map = label_map or build_label_mapping(path)

        signals = []
        labels = []
        subject_ids = []
        filenames = []

        for filepath, _, files in os.walk(self.root):
            for file in files:
                if file.startswith('~$'):
                    continue
                if not file.lower().endswith(('.xls', '.xlsx')):
                    continue
                full_path = os.path.join(filepath, file)
                try:
                    df = pd.read_excel(
                        full_path,
                        names=['time', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6'],
                        header=0,
                    )
                except Exception as exc:
                    print(f'[skip] failed to read {full_path}: {exc}')
                    continue

                if np.sum(np.sum(df == '--')) != 0:
                    print(f'[skip] invalid marker in {full_path}')
                    continue

                df = df.drop(columns=['time'], errors='ignore')
                x = df.fillna(500).to_numpy(dtype=np.float32)
                x = standardize_per_channel(x)
                x = resample(x, resample_len, axis=0).astype(np.float32)  # [500, 6]

                if resample_len % segment_len != 0:
                    raise ValueError('resample_len must be divisible by segment_len.')
                num_segments = resample_len // segment_len
                x = x.reshape(num_segments, segment_len, x.shape[1]).transpose(0, 2, 1)  # [T, C, S]

                raw_label = parse_label_from_filename(file)
                label = self.label_map[raw_label]
                subject_id = parse_subject_id_from_filename(file)

                signals.append(torch.tensor(x, dtype=torch.float32))
                labels.append(label)
                subject_ids.append(subject_id)
                filenames.append(file)

        if not signals:
            raise RuntimeError(f'No valid samples found in {self.root}')

        self.signals = torch.stack(signals, dim=0)    # [N, T, C, S]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = subject_ids
        self.filenames = filenames
        self.num_segments = self.signals.shape[1]
        self.num_channels = self.signals.shape[2]

        print(f'Loaded {len(self.labels)} samples from {self.root}')
        print(f'Signal shape: {tuple(self.signals.shape)}')
        print(f'Classes: {self.label_map}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx], self.subject_ids[idx], self.filenames[idx]


class SegmentEmbedding(nn.Module):
    def __init__(self, segment_len: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(segment_len, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        # x: [B, T, C, S]
        return self.net(x)


class GraphMessagePassing(nn.Module):
    def __init__(self, num_channels: int, hidden_dim: int):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.adj_logits = nn.Parameter(torch.randn(num_channels, num_channels))
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, T, C, D]
        adj = 0.5 * (self.adj_logits + self.adj_logits.t())
        adj = F.softmax(adj, dim=-1)
        identity = torch.eye(self.num_channels, device=x.device)
        adj = adj + identity
        adj = adj / adj.sum(dim=-1, keepdim=True)

        neigh = torch.einsum('ij,btjd->btid', adj, x)
        out = self.self_proj(x) + self.neigh_proj(neigh)
        out = self.act(self.norm(out))
        return out


class GraphEncoder(nn.Module):
    def __init__(self, num_channels: int, segment_len: int, hidden_dim: int = 64, num_graph_layers: int = 2):
        super().__init__()
        self.embed = SegmentEmbedding(segment_len, hidden_dim)
        self.graph_layers = nn.ModuleList([
            GraphMessagePassing(num_channels=num_channels, hidden_dim=hidden_dim)
            for _ in range(num_graph_layers)
        ])
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, T, C, S]
        node_repr = self.embed(x)  # [B, T, C, D]
        for layer in self.graph_layers:
            node_repr = layer(node_repr)
        node_repr = self.node_norm(node_repr)
        segment_repr = node_repr.mean(dim=2)  # [B, T, D]
        sample_repr = segment_repr.mean(dim=1)  # [B, D]
        return node_repr, segment_repr, sample_repr


class GraphScoDetect(nn.Module):
    def __init__(self, num_channels: int, segment_len: int, num_classes: int,
                 hidden_dim: int = 64, lstm_hidden: int = 128):
        super().__init__()
        self.encoder = GraphEncoder(num_channels=num_channels, segment_len=segment_len, hidden_dim=hidden_dim)
        self.temporal_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x):
        node_repr, segment_repr, sample_repr = self.encoder(x)
        temporal_out, _ = self.temporal_model(segment_repr)
        pooled = temporal_out.mean(dim=1)
        logits = self.classifier(pooled)
        return {
            'logits': logits,
            'node_repr': node_repr,
            'segment_repr': segment_repr,
            'sample_repr': sample_repr,
            'temporal_repr': pooled,
        }


def intra_subject_loss(segment_repr: torch.Tensor, labels: torch.Tensor,
                       normal_label: int = NORMAL_LABEL, margin: float = INTRA_MARGIN) -> torch.Tensor:
    # segment_repr: [B, T, D]
    diffs = segment_repr[:, 1:, :] - segment_repr[:, :-1, :]
    sq_dist = (diffs ** 2).sum(dim=-1)         # [B, T-1]
    l_sym = sq_dist.mean(dim=1)                # [B]
    dist = torch.sqrt(sq_dist + 1e-8)
    l_asym = F.relu(margin - dist).pow(2).mean(dim=1)  # [B]

    is_patient = (labels != normal_label).float()
    loss = (1.0 - is_patient) * l_sym + is_patient * l_asym
    return loss.mean()


def supervised_contrastive_subject_loss(segment_repr: torch.Tensor, subject_ids, temperature: float = TEMPERATURE):
    # Flatten segments so that temporal segments act as motion segments.
    # segment_repr: [B, T, D]
    B, T, D = segment_repr.shape
    z = segment_repr.reshape(B * T, D)
    z = F.normalize(z, dim=-1)

    expanded_subject_ids = []
    for sid in subject_ids:
        expanded_subject_ids.extend([sid] * T)

    device = z.device
    sim = torch.matmul(z, z.t()) / temperature
    logits_mask = ~torch.eye(B * T, dtype=torch.bool, device=device)
    sim = sim.masked_fill(~logits_mask, float('-inf'))

    subject_arr = np.array(expanded_subject_ids, dtype=object)
    pos_mask_np = subject_arr[:, None] == subject_arr[None, :]
    pos_mask = torch.tensor(pos_mask_np, device=device, dtype=torch.bool)
    pos_mask = pos_mask & logits_mask

    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1))
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def compute_total_loss(outputs, labels, subject_ids,
                       lambda_inter: float = LAMBDA_INTER,
                       gamma_class: float = GAMMA_CLASS):
    l_intra = intra_subject_loss(outputs['segment_repr'], labels)
    l_inter = supervised_contrastive_subject_loss(outputs['segment_repr'], subject_ids)
    l_class = F.cross_entropy(outputs['logits'], labels)
    total = l_intra + lambda_inter * l_inter + gamma_class * l_class
    return total, {
        'total': total.detach().item(),
        'intra': l_intra.detach().item(),
        'inter': l_inter.detach().item(),
        'class': l_class.detach().item(),
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    for signals, labels, subject_ids, filenames in loader:
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(signals)
        preds = outputs['logits'].argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    acc = correct / max(total, 1)
    return acc, y_true, y_pred


def run_encoder_pretrain(model, train_loader, optimizer, epochs=PRETRAIN_EPOCHS):
    print('=== Stage 1: encoder pretraining ===')
    for epoch in range(epochs):
        model.train()
        loss_meter = []
        intra_meter = []
        inter_meter = []
        for signals, labels, subject_ids, filenames in train_loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(signals)
            l_intra = intra_subject_loss(outputs['segment_repr'], labels)
            l_inter = supervised_contrastive_subject_loss(outputs['segment_repr'], subject_ids)
            loss = l_intra + LAMBDA_INTER * l_inter
            loss.backward()
            optimizer.step()

            loss_meter.append(loss.item())
            intra_meter.append(l_intra.item())
            inter_meter.append(l_inter.item())

        print(
            f'[pretrain] epoch={epoch + 1:03d} '
            f'loss={np.mean(loss_meter):.4f} '
            f'intra={np.mean(intra_meter):.4f} '
            f'inter={np.mean(inter_meter):.4f}'
        )


def run_joint_training(model, train_loader, test_loader, optimizer, epochs=TRAIN_EPOCHS):
    print('=== Stage 2: joint training ===')
    history = {'train_total': [], 'train_intra': [], 'train_inter': [], 'train_class': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_meter = []
        intra_meter = []
        inter_meter = []
        class_meter = []

        for signals, labels, subject_ids, filenames in train_loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(signals)
            loss, loss_dict = compute_total_loss(outputs, labels, subject_ids)
            loss.backward()
            optimizer.step()

            total_meter.append(loss_dict['total'])
            intra_meter.append(loss_dict['intra'])
            inter_meter.append(loss_dict['inter'])
            class_meter.append(loss_dict['class'])

        test_acc, _, _ = evaluate(model, test_loader)
        history['train_total'].append(np.mean(total_meter))
        history['train_intra'].append(np.mean(intra_meter))
        history['train_inter'].append(np.mean(inter_meter))
        history['train_class'].append(np.mean(class_meter))
        history['test_acc'].append(test_acc)

        print(
            f'[train] epoch={epoch + 1:03d} '
            f'total={np.mean(total_meter):.4f} '
            f'intra={np.mean(intra_meter):.4f} '
            f'inter={np.mean(inter_meter):.4f} '
            f'class={np.mean(class_meter):.4f} '
            f'test_acc={test_acc * 100:.2f}%'
        )
    return history


def main():
    label_map = build_label_mapping(PATH)
    train_dataset = SensorSegmentDataset(train=True, path=PATH, label_map=label_map)
    test_dataset = SensorSegmentDataset(train=False, path=PATH, label_map=label_map)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GraphScoDetect(
        num_channels=train_dataset.num_channels,
        segment_len=SEGMENT_LEN,
        num_classes=len(label_map),
        hidden_dim=64,
        lstm_hidden=128,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    run_encoder_pretrain(model, train_loader, optimizer, epochs=PRETRAIN_EPOCHS)
    history = run_joint_training(model, train_loader, test_loader, optimizer, epochs=TRAIN_EPOCHS)

    test_acc, y_true, y_pred = evaluate(model, test_loader)
    print(f'Final test accuracy: {test_acc * 100:.2f}%')
    return model, history, y_true, y_pred


if __name__ == '__main__':
    model, history, y_true, y_pred = main()
