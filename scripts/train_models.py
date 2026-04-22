import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging

# Adjust python path to allow importing from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.protonet import ProtoNet, SupportSetManager
from src.rl.agent import TabularQLearningAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WEIGHTS_DIR = "backend/models/weights"

def create_weights_dir():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

def mock_ukdale_data(num_samples=1000, window_size=60):
    """Mocks UK-DALE 1Hz data for training."""
    logger.info("Loading UK-DALE dataset...")
    X = torch.randn(num_samples, 1, window_size) * 10 + 100
    y = np.random.choice(["fridge", "microwave"], num_samples)
    return X, y

def mock_redd_data(num_samples=200, window_size=60):
    """Mocks REDD data for validation/generalization."""
    logger.info("Loading REDD dataset...")
    X = torch.randn(num_samples, 1, window_size) * 12 + 105
    y = np.random.choice(["fridge", "microwave"], num_samples)
    return X, y

def train_protonet():
    logger.info("Starting CNN & ProtoNet Training Phase...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    window_size = 60
    embedding_size = 64
    batch_size = 32
    epochs = 200
    patience = 10
    
    model = ProtoNet(input_size=window_size, embedding_size=embedding_size).to(device)
    support_manager = SupportSetManager(max_memory_per_class=20)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train, y_train = mock_ukdale_data(num_samples=1000, window_size=window_size)
    X_val, y_val = mock_redd_data(num_samples=200, window_size=window_size)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Shuffle training data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size].to(device)
            batch_y = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            embeddings = model(batch_X) # [batch_size, embedding_size]
            
            for j in range(len(batch_y)):
                support_manager.add_embedding(batch_y[j], embeddings[j].detach().cpu())
            
            # Simplified loss calculation for mocked script
            loss = torch.mean(torch.sum(embeddings ** 2, dim=1)) * 0.001
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X_val = X_val[i:i+batch_size].to(device)
                embeddings_val = model(batch_X_val)
                loss_val = torch.mean(torch.sum(embeddings_val ** 2, dim=1)) * 0.001
                val_loss += loss_val.item()
                
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best weights
            create_weights_dir()
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "cnn_weights.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logger.info("ProtoNet Training Complete.")
    
    # Save Anchors
    anchors_path = os.path.join(WEIGHTS_DIR, "protonet_anchors.pt")
    torch.save(support_manager.support_sets, anchors_path)
    logger.info(f"Saved ProtoNet anchors to {anchors_path}")

def train_rl_agent():
    logger.info("Starting RL Agent Training Phase...")
    agent = TabularQLearningAgent()
    
    # Simulated Epoch
    epochs = 1000
    for _ in range(epochs):
        # Random state: time=12, power=5, 4 devices off (0)
        state = agent.get_state_tuple(time_bucket=12, power_bin=5, device_states=(0, 0, 0, 0))
        action = agent.get_action(state)
        
        # Fake reward and next state
        reward = 1.0 if action == 0 else -1.0
        next_state = agent.get_state_tuple(time_bucket=12, power_bin=5, device_states=(0, 0, 0, 0))
        
        agent.update(state, action, reward, next_state)
        
    logger.info("RL Agent Training Complete.")
    
    create_weights_dir()
    q_table_path = os.path.join(WEIGHTS_DIR, "q_table.pkl")
    with open(q_table_path, "wb") as f:
        pickle.dump(agent.q_table, f)
    logger.info(f"Saved Q-Table to {q_table_path}")

if __name__ == "__main__":
    logger.info("=== STARTING OFFLINE ML TRAINING PIPELINE ===")
    train_protonet()
    train_rl_agent()
    logger.info("=== ALL PHASES COMPLETE ===")
