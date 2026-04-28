import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
import scipy.optimize
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, hidden_size, seq_len]
        x_permuted = x.permute(0, 2, 1) # [batch_size, seq_len, hidden_size]
        attn_weights = F.softmax(self.attention(x_permuted), dim=1) # [batch_size, seq_len, 1]
        weighted = x_permuted * attn_weights # [batch_size, seq_len, hidden_size]
        return weighted.sum(dim=1) # [batch_size, hidden_size]

class CNN1DEncoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int = 128):
        super(CNN1DEncoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.attention = TemporalAttention(hidden_size=128)
        self.projection = nn.Linear(128, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, seq_len]
        conv_out = self.conv_stack(x) # [batch_size, 128, seq_len]
        attn_out = self.attention(conv_out) # [batch_size, 128]
        return self.projection(attn_out) # [batch_size, embedding_size]

# Ensure backward compatibility with existing usages of ProtoNet
ProtoNet = CNN1DEncoder

class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit temperature parameter to minimize NLL on calibration set."""
        def eval_nll(t):
            scaled_logits = logits / t[0]
            log_probs = F.log_softmax(scaled_logits, dim=1)
            loss = F.nll_loss(log_probs, labels)
            return loss.item()

        # Optimize T
        res = scipy.optimize.minimize(eval_nll, x0=np.array([1.0]), bounds=[(0.01, 100.0)], method='L-BFGS-B')
        with torch.no_grad():
            self.temperature.copy_(torch.tensor([res.x[0]]))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))

class WEibullOpenMax:
    def __init__(self, tail_size: int = 20, alpha: int = 3):
        self.tail_size = tail_size
        self.alpha = alpha
        self.weibull_models: Dict[str, Any] = {}

    def fit(self, embeddings_per_class: Dict[str, np.ndarray]) -> None:
        """
        Fits a Weibull distribution per class based on the distance of support embeddings
        to their respective class prototype.
        """
        for class_name, embeddings in embeddings_per_class.items():
            prototype = np.mean(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - prototype, axis=1)
            # Take the tail_size largest distances
            tail_distances = np.sort(distances)[-self.tail_size:]
            
            # If we don't have enough samples, use what we have or add tiny noise to avoid fitting errors
            if len(tail_distances) < 2:
                 tail_distances = np.append(tail_distances, tail_distances[0] + 1e-5)
                 
            # Fit Weibull (shape, loc, scale)
            shape, loc, scale = scipy.stats.weibull_min.fit(tail_distances, floc=0)
            self.weibull_models[class_name] = {'shape': shape, 'loc': loc, 'scale': scale, 'prototype': prototype}

    def compute_open_set_prob(self, embedding: np.ndarray, class_names: List[str], distances: List[float]) -> float:
        """
        Compute the probability of an embedding being 'unknown' (open set).
        """
        if not self.weibull_models:
            return 0.0
            
        # Get distances to prototypes from the models, since they align with our Weibull parameters
        model_distances = []
        available_classes = []
        for class_name in class_names:
            if class_name in self.weibull_models:
                proto = self.weibull_models[class_name]['prototype']
                dist = np.linalg.norm(embedding - proto)
                model_distances.append((dist, class_name))
                available_classes.append(class_name)
                
        if not model_distances:
            return 1.0

        # Sort by distance
        model_distances.sort(key=lambda x: x[0])
        top_alpha_classes = model_distances[:self.alpha]
        
        unknown_probs = []
        for dist, class_name in top_alpha_classes:
            params = self.weibull_models[class_name]
            # Weibull CDF: probability that a distance from the prototype is <= dist
            # High CDF means it is an outlier (far away)
            cdf = scipy.stats.weibull_min.cdf(dist, params['shape'], loc=params['loc'], scale=params['scale'])
            unknown_probs.append(cdf)
            
        # If the sample is far from the top-alpha nearest classes, it's likely unknown
        # Probability of being unknown is the maximum over the nearest classes
        return float(np.max(unknown_probs)) if unknown_probs else 0.0

class SupportSetManager:
    def __init__(self):
        self.raw_windows: Dict[str, List[np.ndarray]] = {}

    def add_support(self, class_name: str, window: np.ndarray) -> None:
        """Adds a raw window to the support set for a given class."""
        if class_name not in self.raw_windows:
            self.raw_windows[class_name] = []
        self.raw_windows[class_name].append(window)

    def compute_prototypes(self, encoder: nn.Module, device: torch.device = torch.device('cpu')) -> Dict[str, np.ndarray]:
        """Computes prototype embeddings for each class using the encoder."""
        prototypes = {}
        encoder.eval()
        with torch.no_grad():
            for class_name, windows in self.raw_windows.items():
                tensor_windows = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(1).to(device)
                embeddings = encoder(tensor_windows)
                prototypes[class_name] = embeddings.mean(dim=0).cpu().numpy()
        return prototypes

    def fit_openmax(self, encoder: nn.Module, weibull_model: WEibullOpenMax, device: torch.device = torch.device('cpu')) -> None:
        """Fits the Weibull OpenMax model."""
        encoder.eval()
        embeddings_per_class = {}
        with torch.no_grad():
            for class_name, windows in self.raw_windows.items():
                tensor_windows = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(1).to(device)
                embeddings = encoder(tensor_windows).cpu().numpy()
                embeddings_per_class[class_name] = embeddings
        weibull_model.fit(embeddings_per_class)

    def classify(self, window: np.ndarray, encoder: nn.Module, weibull: WEibullOpenMax, scaler: TemperatureScaler, confidence_threshold: float, device: torch.device = torch.device('cpu')) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies a window, returns (class_name, confidence, distances).
        Returns 'unknown' if open_set_prob > (1 - confidence_threshold).
        """
        encoder.eval()
        scaler.eval()
        
        # Prepare input
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = encoder(x).squeeze(0) # [embedding_size]
            
            prototypes = self.compute_prototypes(encoder, device)
            if not prototypes:
                 return "unknown", 0.0, {}
                 
            class_names = list(prototypes.keys())
            proto_tensors = torch.tensor(np.array([prototypes[c] for c in class_names]), dtype=torch.float32).to(device)
            
            # Compute squared Euclidean distances
            dists = torch.cdist(embedding.unsqueeze(0), proto_tensors, p=2).pow(2).squeeze(0) # [num_classes]
            
            # Logits are negative distances
            logits = -dists
            
            # Scale logits and compute softmax probabilities
            scaled_logits = scaler(logits.unsqueeze(0)).squeeze(0)
            probs = F.softmax(scaled_logits, dim=0)
            
            max_prob, max_idx = torch.max(probs, dim=0)
            predicted_class = class_names[max_idx.item()]
            confidence = max_prob.item()
            
            dist_dict = {name: dists[i].item() for i, name in enumerate(class_names)}
            
            # OpenMax check
            emb_np = embedding.cpu().numpy()
            open_set_prob = weibull.compute_open_set_prob(emb_np, class_names, list(dist_dict.values()))
            
            if open_set_prob > (1.0 - confidence_threshold):
                return "unknown", open_set_prob, dist_dict
                
            return predicted_class, confidence, dist_dict

    def save_registry(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.raw_windows, f)

    def load_registry(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.raw_windows = pickle.load(f)

    def incremental_update(self, class_name: str, new_window: np.ndarray, encoder: nn.Module) -> None:
        """Adds one sample (incremental update without full retraining)."""
        self.add_support(class_name, new_window)
        # Re-computation of prototypes happens dynamically in classify() or fit_openmax()


class EpisodicDataset:
    def __init__(self, raw_windows_per_class: Dict[str, List[np.ndarray]]):
        self.windows = {k: np.array(v) for k, v in raw_windows_per_class.items()}
        self.class_names = list(self.windows.keys())

    def sample_episode(self, n_way: int, k_shot: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (support_set, query_set, labels).
        support_set: [n_way * k_shot, 1, seq_len]
        query_set: [n_way * n_query, 1, seq_len]
        labels: [n_way * n_query]
        """
        # Randomly pick n_way classes
        selected_classes = np.random.choice(self.class_names, size=min(n_way, len(self.class_names)), replace=False)
        n_way_actual = len(selected_classes)
        
        support_list = []
        query_list = []
        labels_list = []
        
        for i, cls in enumerate(selected_classes):
            cls_windows = self.windows[cls]
            num_samples = len(cls_windows)
            
            # Need at least k_shot + n_query samples
            if num_samples < k_shot + n_query:
                # Fallback: duplicate samples if not enough data
                indices = np.random.choice(num_samples, size=k_shot + n_query, replace=True)
            else:
                indices = np.random.choice(num_samples, size=k_shot + n_query, replace=False)
                
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:]
            
            support_list.append(torch.tensor(cls_windows[support_idx], dtype=torch.float32).unsqueeze(1))
            query_list.append(torch.tensor(cls_windows[query_idx], dtype=torch.float32).unsqueeze(1))
            labels_list.extend([i] * len(query_idx))
            
        support_set = torch.cat(support_list, dim=0) # [n_way * k_shot, 1, seq_len]
        query_set = torch.cat(query_list, dim=0) # [n_way * n_query, 1, seq_len]
        labels = torch.tensor(labels_list, dtype=torch.long) # [n_way * n_query]
        
        return support_set, query_set, labels
