"""
integration_step3.py - Deep Q-Network Training Pipeline for adaptive Learning Tutor
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time
from datetime import datetime
import os
from pathlib import Path
import pickle
from tqdm import tqdm

# Imports from improved Modules
from integration_LS_AT import TutorIntegrationManager, Config
from integration_setup_RL import (
    AdaptiveLearningEnvironment, 
    create_enhanced_content_library
)

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seeds(seed: int = 42):
    """Set all Random Seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# Experience Tuple
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])

class AttentionLayer(nn.Module):
    """Self-Attention Layer for better Feature-Interactions"""
    
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended = torch.matmul(attention_weights, V)
        return attended + x  # Residual connection

class ImprovedDQNNetwork(nn.Module):
    """Improved DQN with Attention and Skip Connections"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [512, 512, 256],
                 dropout_rate: float = 0.1,
                 use_attention: bool = True):
        super(ImprovedDQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_attention = use_attention
        
        # Input projection
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dims[i+1]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_dims[-1])
            self.attention_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Output layers
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Improved weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with skip connections"""
        # Input projection
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Hidden layers with residual connections
        for i, (layer, norm, dropout) in enumerate(
            zip(self.hidden_layers, self.layer_norms, self.dropouts)):
            
            residual = x if i == 0 or x.shape[-1] == layer.out_features else 0
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            
            if isinstance(residual, torch.Tensor):
                x = x + residual * 0.1  # Scaled residual
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
            x = self.attention_norm(x)
        
        # Output
        q_values = self.output_layer(x)
        
        return q_values

class ImprovedDuelingDQN(nn.Module):
    """Improved Dueling DQN with Noisy Networks"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [512, 512],
                 dropout_rate: float = 0.1,
                 use_noisy: bool = True):
        super(ImprovedDuelingDQN, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.shared_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling architecture
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values

class ImprovedReplayBuffer:
    """Improved Replay Buffer with better Management"""
    
    def __init__(self, capacity: int, prioritized: bool = True, alpha: float = 0.6):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, experience: Experience, td_error: float = None):
        """Adding Experience"""
        if self.prioritized and td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = max(self.priorities, default=1.0)
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample Batch with Importance Sampling"""
        if len(self.buffer) < batch_size:
            return None, None, None
        
        if self.prioritized:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            
            indices = np.random.choice(
                len(self.buffer), batch_size, p=probabilities
            )
            
            # Importance sampling weights
            n = len(self.buffer)
            weights = (n * probabilities[indices]) ** (-beta)
            weights /= weights.max()
            
            experiences = [self.buffer[idx] for idx in indices]
            
            return experiences, indices, weights
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size)
            experiences = [self.buffer[idx] for idx in indices]
            return experiences, indices, np.ones(batch_size)
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities after Training"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class ImprovedDQNAgent:
    """Improved DQN Agent with modern Techniques"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any] = None):
        
        # Default configuration
        default_config = {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.005,  # Soft update parameter
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.9995,
            'batch_size': 64,
            'memory_size': 50000,
            'target_update_freq': 1000,
            'use_double_dqn': True,
            'use_dueling': True,
            'use_prioritized_replay': True,
            'use_attention': True,
            'clip_grad_norm': 1.0,
            'device': None
        }
        
        # Merge with provided config
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Device
        if self.config['device'] is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config['device'])
        
        logger.info(f"DQN Agent using device: {self.device}")
        
        # Networks
        if self.config['use_dueling']:
            self.q_network = ImprovedDuelingDQN(
                state_dim, action_dim
            ).to(self.device)
            self.target_network = ImprovedDuelingDQN(
                state_dim, action_dim
            ).to(self.device)
        else:
            self.q_network = ImprovedDQNNetwork(
                state_dim, action_dim,
                use_attention=self.config['use_attention']
            ).to(self.device)
            self.target_network = ImprovedDQNNetwork(
                state_dim, action_dim,
                use_attention=self.config['use_attention']
            ).to(self.device)
        
        # Copy weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=0.0001
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        # Replay buffer
        self.memory = ImprovedReplayBuffer(
            self.config['memory_size'],
            prioritized=self.config['use_prioritized_replay']
        )
        
        # Exploration
        self.epsilon = self.config['epsilon_start']
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
        # Metrics
        self.metrics = {
            'losses': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'td_errors': deque(maxlen=1000),
            'rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'learning_rates': deque(maxlen=1000)
        }
        
    def get_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Improved epsilon-greedy action selection"""
        
        if training and random.random() < self.epsilon:
            # Structured exploration
            action = np.random.uniform(0, 1, self.action_dim)
            
            # Bias exploration towards reasonable values
            action[0] = np.clip(action[0], 0.3, 0.9)  # Content selection
            action[1] = np.clip(action[1], 0.4, 0.8)  # Explanation depth
            action[2] = np.clip(action[2], 0.5, 0.9)  # Interaction level
            
            return action
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Store Q-values for analysis
            self.metrics['q_values'].append(np.mean(q_values))
            
            # Map Q-values to continuous actions
            action = self._q_values_to_action(q_values)
            
            return action
    
    def _q_values_to_action(self, q_values: np.ndarray) -> np.ndarray:
        """Map Q-values to continuous action space"""
        # Normalize Q-values
        if q_values.max() - q_values.min() > 0:
            q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min())
        else:
            q_norm = np.ones_like(q_values) * 0.5
        
        # Map to action dimensions
        if len(q_norm) >= self.action_dim:
            action = q_norm[:self.action_dim]
        else:
            action = np.pad(q_norm, (0, self.action_dim - len(q_norm)), 
                           constant_values=0.5)
        
        # Apply action-specific constraints
        action[0] = np.clip(action[0], 0.1, 0.9)  # Content selection
        action[4] = 1.0 if action[4] > 0.7 else 0.0  # Break suggestion (binary)
        action[6] = 1.0 if action[6] > 0.8 else 0.0  # Recap (binary)
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done, info=None):
        """Store experience with TD error calculation"""
        
        experience = Experience(state, action, reward, next_state, done, info)
        
        # Calculate TD error for prioritization
        if self.config['use_prioritized_replay']:
            with torch.no_grad():
                state_tensor = torch.FloatTensor([state]).to(self.device)
                next_state_tensor = torch.FloatTensor([next_state]).to(self.device)
                
                current_q = self.q_network(state_tensor).max(1)[0].item()
                
                if self.config['use_double_dqn']:
                    next_action = self.q_network(next_state_tensor).argmax(1)
                    next_q = self.target_network(next_state_tensor).gather(
                        1, next_action.unsqueeze(1)
                    ).item()
                else:
                    next_q = self.target_network(next_state_tensor).max(1)[0].item()
                
                td_error = abs(reward + self.config['gamma'] * next_q * (1 - done) - current_q)
                
                self.metrics['td_errors'].append(td_error)
        else:
            td_error = None
        
        self.memory.add(experience, td_error)
    
    def train_step(self) -> Optional[float]:
        """Improved training step"""
        
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample batch
        if self.config['use_prioritized_replay']:
            beta = min(1.0, 0.4 + 0.6 * (self.training_step / 50000))
            experiences, indices, weights = self.memory.sample(
                self.config['batch_size'], beta
            )
        else:
            experiences, indices, weights = self.memory.sample(
                self.config['batch_size']
            )
        
        if experiences is None:
            return None
        
        # Prepare batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states)
        
        # For continuous actions, we use the mean Q-value as target
        current_q = current_q_values.mean(dim=1)
        
        # Target Q values
        with torch.no_grad():
            if self.config['use_double_dqn']:
                # Double DQN
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + self.config['gamma'] * next_q * (1 - dones)
        
        # Loss with importance sampling
        td_errors = current_q - target_q
        loss = (weights * td_errors.pow(2)).mean()
        
        # Add regularization
        l2_reg = sum(p.pow(2).sum() for p in self.q_network.parameters())
        loss = loss + 0.0001 * l2_reg
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), 
                self.config['clip_grad_norm']
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        if self.config['use_prioritized_replay'] and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors_np)
        
        # Soft update target network
        if self.training_step % 100 == 0:  # More frequent soft updates
            self._soft_update_target_network()
        
        # Hard update at intervals
        if self.training_step % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"Target network updated at step {self.training_step}")
        
        self.training_step += 1
        
        # Store metrics
        self.metrics['losses'].append(loss.item())
        self.metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])
        
        return loss.item()
    
    def _soft_update_target_network(self):
        """Soft update of target network parameters"""
        tau = self.config['tau']
        
        for target_param, param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    def update_epsilon(self):
        """Update exploration rate"""
        if self.epsilon > self.config['epsilon_end']:
            self.epsilon *= self.config['epsilon_decay']
            self.epsilon = max(self.config['epsilon_end'], self.epsilon)
    
    def save_checkpoint(self, filepath: Path, additional_info: Dict = None):
        """Save complete checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config,
            'metrics': {k: list(v) for k, v in self.metrics.items()},
            'additional_info': additional_info or {}
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Path):
        """Load complete checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.config['epsilon_start'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        # Restore metrics
        if 'metrics' in checkpoint:
            for key, values in checkpoint['metrics'].items():
                if key in self.metrics:
                    self.metrics[key] = deque(values, maxlen=self.metrics[key].maxlen)
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint.get('additional_info', {})

class ImprovedTrainingPipeline:
    """Improved Training Pipeline with advanced Features"""
    
    def __init__(self,
                 env: AdaptiveLearningEnvironment,
                 agent: ImprovedDQNAgent,
                 config: Config = None,
                 save_dir: str = "improved_models"):
        
        self.env = env
        self.agent = agent
        self.config = config or Config()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Training configuration
        self.training_config = {
            'log_interval': 10,
            'eval_interval': 50,
            'save_interval': 100,
            'eval_episodes': 10,
            'early_stopping_patience': 20,
            'min_episodes': 100
        }
        
        # Metrics tracking
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_performances': [],
            'eval_rewards': [],
            'eval_performances': [],
            'best_eval_reward': float('-inf'),
            'episodes_without_improvement': 0
        }
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup detailed logging"""
        log_dir = self.save_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(
            log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
    
    def train_episode(self) -> Dict[str, float]:
        """Train one episode with detailed tracking"""
        
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        performances = []
        engagement_levels = []
        knowledge_gains = []
        
        done = False
        
        while not done:
            # Get action
            action = self.agent.get_action(state, training=True)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(
                state, action, reward, next_state, done, info
            )
            
            # Train
            loss = self.agent.train_step()
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Track metrics
            if 'response' in info:
                performances.append(info['response'].get('performance', 0))
                knowledge_gains.append(info['response'].get('knowledge_gain', 0))
            
            if 'state' in info:
                engagement_levels.append(info['state'].get('engagement', 0))
        
        # Update exploration
        self.agent.update_epsilon()
        self.agent.episode_count += 1
        
        # Calculate episode metrics
        metrics = {
            'reward': episode_reward,
            'length': episode_length,
            'avg_performance': np.mean(performances) if performances else 0,
            'avg_engagement': np.mean(engagement_levels) if engagement_levels else 0,
            'total_knowledge_gain': sum(knowledge_gains) if knowledge_gains else 0,
            'final_satisfaction': self.env.current_student.satisfaction_score
        }
        
        return metrics
    
    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """Evaluate agent without exploration"""
        
        num_episodes = num_episodes or self.training_config['eval_episodes']
        
        eval_metrics = {
            'rewards': [],
            'lengths': [],
            'performances': [],
            'engagements': [],
            'knowledge_gains': [],
            'satisfactions': []
        }
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            performances = []
            engagements = []
            knowledge_gains = []
            
            done = False
            steps = 0
            
            while not done and steps < 200:  # Max steps per eval episode
                action = self.agent.get_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                # Track metrics
                if 'response' in info:
                    performances.append(info['response'].get('performance', 0))
                    knowledge_gains.append(info['response'].get('knowledge_gain', 0))
                
                if 'state' in info:
                    engagements.append(info['state'].get('engagement', 0))
            
            eval_metrics['rewards'].append(episode_reward)
            eval_metrics['lengths'].append(steps)
            eval_metrics['performances'].append(np.mean(performances) if performances else 0)
            eval_metrics['engagements'].append(np.mean(engagements) if engagements else 0)
            eval_metrics['knowledge_gains'].append(sum(knowledge_gains) if knowledge_gains else 0)
            eval_metrics['satisfactions'].append(self.env.current_student.satisfaction_score)
        
        # Aggregate metrics
        aggregated = {
            'avg_reward': np.mean(eval_metrics['rewards']),
            'std_reward': np.std(eval_metrics['rewards']),
            'avg_length': np.mean(eval_metrics['lengths']),
            'avg_performance': np.mean(eval_metrics['performances']),
            'avg_engagement': np.mean(eval_metrics['engagements']),
            'avg_knowledge_gain': np.mean(eval_metrics['knowledge_gains']),
            'avg_satisfaction': np.mean(eval_metrics['satisfactions'])
        }
        
        return aggregated
    
    def train(self, num_episodes: int = 1000, resume_from: str = None):
        """Main training loop with checkpointing and early stopping"""
        
        logger.info(f"Starting training for {num_episodes} episodes")
        logger.info(f"Agent: {type(self.agent.q_network).__name__}")
        logger.info(f"Environment: {len(self.env.content_library)} contents")
        
        # Resume from checkpoint if provided
        start_episode = 0
        if resume_from:
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                info = self.agent.load_checkpoint(checkpoint_path)
                start_episode = info.get('episode', 0)
                self.training_metrics = info.get('training_metrics', self.training_metrics)
                logger.info(f"Resumed from episode {start_episode}")
        
        # Training loop
        start_time = time.time()
        
        try:
            for episode in tqdm(range(start_episode, num_episodes), 
                               desc="Training Episodes"):
                
                # Train episode
                episode_metrics = self.train_episode()
                
                # Store metrics
                self.training_metrics['episode_rewards'].append(episode_metrics['reward'])
                self.training_metrics['episode_lengths'].append(episode_metrics['length'])
                self.training_metrics['episode_performances'].append(
                    episode_metrics['avg_performance']
                )
                
                # Logging
                if episode % self.training_config['log_interval'] == 0:
                    self._log_training_progress(episode)
                
                # Evaluation
                if episode % self.training_config['eval_interval'] == 0 and episode > 0:
                    eval_metrics = self.evaluate()
                    self.training_metrics['eval_rewards'].append(eval_metrics['avg_reward'])
                    self.training_metrics['eval_performances'].append(
                        eval_metrics['avg_performance']
                    )
                    
                    # Check for improvement
                    if eval_metrics['avg_reward'] > self.training_metrics['best_eval_reward']:
                        self.training_metrics['best_eval_reward'] = eval_metrics['avg_reward']
                        self.training_metrics['episodes_without_improvement'] = 0
                        
                        # Save best model
                        best_path = self.save_dir / "best_model.pth"
                        self.agent.save_checkpoint(best_path, {
                            'episode': episode,
                            'eval_metrics': eval_metrics,
                            'training_metrics': self.training_metrics
                        })
                        logger.info(f"New best model saved: {eval_metrics['avg_reward']:.2f}")
                    else:
                        self.training_metrics['episodes_without_improvement'] += 1
                    
                    self._log_evaluation_results(episode, eval_metrics)
                    
                    # Early stopping
                    if (self.training_metrics['episodes_without_improvement'] > 
                        self.training_config['early_stopping_patience'] and
                        episode > self.training_config['min_episodes']):
                        
                        logger.info("Early stopping triggered")
                        break
                
                # Save checkpoint
                if episode % self.training_config['save_interval'] == 0 and episode > 0:
                    checkpoint_path = self.save_dir / f"checkpoint_ep{episode}.pth"
                    self.agent.save_checkpoint(checkpoint_path, {
                        'episode': episode,
                        'training_metrics': self.training_metrics
                    })
                    
                    # Save metrics
                    self.save_metrics()
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final save
            training_time = time.time() - start_time
            final_path = self.save_dir / "final_model.pth"
            self.agent.save_checkpoint(final_path, {
                'episode': episode,
                'training_time': training_time,
                'training_metrics': self.training_metrics
            })
            
            self.save_metrics()
            self.create_training_plots()
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final epsilon: {self.agent.epsilon:.4f}")
            logger.info(f"Total training steps: {self.agent.training_step}")
        
        return self.training_metrics
    
    def _log_training_progress(self, episode: int):
        """Log training progress"""
        recent_rewards = self.training_metrics['episode_rewards'][-10:]
        recent_performances = self.training_metrics['episode_performances'][-10:]
        
        logger.info(
            f"Episode {episode:4d} | "
            f"Reward: {np.mean(recent_rewards):7.2f} | "
            f"Performance: {np.mean(recent_performances):5.3f} | "
            f"Epsilon: {self.agent.epsilon:.3f} | "
            f"Loss: {np.mean(list(self.agent.metrics['losses'])):.4f} | "
            f"LR: {self.agent.scheduler.get_last_lr()[0]:.6f}"
        )
    
    def _log_evaluation_results(self, episode: int, eval_metrics: Dict):
        """Log evaluation results"""
        logger.info(
            f"Evaluation {episode:4d} | "
            f"Reward: {eval_metrics['avg_reward']:7.2f} ± {eval_metrics['std_reward']:.2f} | "
            f"Performance: {eval_metrics['avg_performance']:5.3f} | "
            f"Engagement: {eval_metrics['avg_engagement']:5.3f} | "
            f"Knowledge: {eval_metrics['avg_knowledge_gain']:5.3f}"
        )
    
    def save_metrics(self):
        """Save all training metrics"""
        metrics_path = self.save_dir / "training_metrics.json"
        
        # Prepare metrics for JSON
        save_metrics = {
            'config': self.agent.config,
            'training_config': self.training_config,
            'episode_rewards': self.training_metrics['episode_rewards'],
            'episode_lengths': self.training_metrics['episode_lengths'],
            'episode_performances': self.training_metrics['episode_performances'],
            'eval_rewards': self.training_metrics['eval_rewards'],
            'eval_performances': self.training_metrics['eval_performances'],
            'best_eval_reward': self.training_metrics['best_eval_reward'],
            'final_epsilon': self.agent.epsilon,
            'total_steps': self.agent.training_step,
            'total_episodes': self.agent.episode_count
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_path}")
    
    def create_training_plots(self):
        """Create comprehensive training plots"""
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Episode Rewards
        ax1 = plt.subplot(3, 3, 1)
        rewards = self.training_metrics['episode_rewards']
        if len(rewards) > 0:
            ax1.plot(rewards, alpha=0.3, label='Raw')
            if len(rewards) > 50:
                smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
                ax1.plot(smoothed, label='Smoothed (50)')
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
        
        # 2. Performance
        ax2 = plt.subplot(3, 3, 2)
        performances = self.training_metrics['episode_performances']
        if len(performances) > 0:
            ax2.plot(performances, alpha=0.3, label='Raw')
            if len(performances) > 50:
                smoothed = np.convolve(performances, np.ones(50)/50, mode='valid')
                ax2.plot(smoothed, label='Smoothed (50)')
            ax2.set_title('Student Performance')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Performance')
            ax2.set_ylim([0, 1])
            ax2.legend()
        
        # 3. Evaluation Results
        ax3 = plt.subplot(3, 3, 3)
        eval_rewards = self.training_metrics['eval_rewards']
        if len(eval_rewards) > 0:
            episodes = np.arange(len(eval_rewards)) * self.training_config['eval_interval']
            ax3.plot(episodes, eval_rewards, 'o-', label='Eval Reward')
            ax3.axhline(y=self.training_metrics['best_eval_reward'], 
                       color='r', linestyle='--', label='Best')
            ax3.set_title('Evaluation Rewards')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Average Reward')
            ax3.legend()
        
        # 4. Loss
        ax4 = plt.subplot(3, 3, 4)
        losses = list(self.agent.metrics['losses'])
        if len(losses) > 0:
            ax4.plot(losses)
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
            ax4.set_yscale('log')
        
        # 5. Q-Values
        ax5 = plt.subplot(3, 3, 5)
        q_values = list(self.agent.metrics['q_values'])
        if len(q_values) > 0:
            ax5.plot(q_values)
            ax5.set_title('Average Q-Values')
            ax5.set_xlabel('Training Step')
            ax5.set_ylabel('Q-Value')
        
        # 6. TD Errors
        ax6 = plt.subplot(3, 3, 6)
        td_errors = list(self.agent.metrics['td_errors'])
        if len(td_errors) > 0:
            ax6.plot(td_errors, alpha=0.5)
            ax6.set_title('TD Errors')
            ax6.set_xlabel('Training Step')
            ax6.set_ylabel('TD Error')
            ax6.set_yscale('log')
        
        # 7. Epsilon Decay
        ax7 = plt.subplot(3, 3, 7)
        epsilons = np.array([self.agent.config['epsilon_start'] * 
                            (self.agent.config['epsilon_decay'] ** i) 
                            for i in range(len(rewards))])
        ax7.plot(epsilons)
        ax7.axhline(y=self.agent.config['epsilon_end'], 
                   color='r', linestyle='--', label='Min Epsilon')
        ax7.set_title('Epsilon Decay')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Epsilon')
        ax7.legend()
        
        # 8. Learning Rate
        ax8 = plt.subplot(3, 3, 8)
        lrs = list(self.agent.metrics['learning_rates'])
        if len(lrs) > 0:
            ax8.plot(lrs)
            ax8.set_title('Learning Rate Schedule')
            ax8.set_xlabel('Training Step')
            ax8.set_ylabel('Learning Rate')
            ax8.set_yscale('log')
        
        # 9. Episode Lengths
        ax9 = plt.subplot(3, 3, 9)
        lengths = self.training_metrics['episode_lengths']
        if len(lengths) > 0:
            ax9.plot(lengths, alpha=0.5)
            if len(lengths) > 50:
                smoothed = np.convolve(lengths, np.ones(50)/50, mode='valid')
                ax9.plot(smoothed, label='Smoothed (50)')
            ax9.set_title('Episode Lengths')
            ax9.set_xlabel('Episode')
            ax9.set_ylabel('Steps')
            ax9.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {plot_path}")
        
        # Create additional detailed plots
        self._create_detailed_analysis_plots()
    
    def _create_detailed_analysis_plots(self):
        """Create additional detailed analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Performance vs Reward Correlation
        ax1 = axes[0, 0]
        if (len(self.training_metrics['episode_rewards']) > 0 and 
            len(self.training_metrics['episode_performances']) > 0):
            
            rewards = self.training_metrics['episode_rewards']
            performances = self.training_metrics['episode_performances']
            
            # Scatter plot with trend
            ax1.scatter(performances, rewards, alpha=0.5, s=10)
            
            # Trend line
            z = np.polyfit(performances, rewards, 1)
            p = np.poly1d(z)
            ax1.plot(sorted(performances), p(sorted(performances)), 
                    "r--", alpha=0.8, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            
            ax1.set_xlabel('Average Performance')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Performance vs Reward Correlation')
            ax1.legend()
        
        # 2. Learning Progress Heatmap
        ax2 = axes[0, 1]
        if len(self.training_metrics['episode_performances']) > 100:
            # Reshape performances into blocks
            perfs = np.array(self.training_metrics['episode_performances'])
            block_size = 20
            n_blocks = len(perfs) // block_size
            
            if n_blocks > 0:
                perf_blocks = perfs[:n_blocks*block_size].reshape(n_blocks, block_size)
                
                im = ax2.imshow(perf_blocks.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
                ax2.set_xlabel('Training Blocks (20 episodes each)')
                ax2.set_ylabel('Episode within Block')
                ax2.set_title('Learning Progress Heatmap')
                plt.colorbar(im, ax=ax2, label='Performance')
        
        # 3. Reward Distribution Evolution
        ax3 = axes[1, 0]
        rewards = self.training_metrics['episode_rewards']
        if len(rewards) > 100:
            # Split into quarters
            quarter_size = len(rewards) // 4
            
            for i in range(4):
                start_idx = i * quarter_size
                end_idx = (i + 1) * quarter_size
                quarter_rewards = rewards[start_idx:end_idx]
                
                ax3.hist(quarter_rewards, bins=20, alpha=0.5, 
                        label=f'Quarter {i+1}', density=True)
            
            ax3.set_xlabel('Episode Reward')
            ax3.set_ylabel('Density')
            ax3.set_title('Reward Distribution Evolution')
            ax3.legend()
        
        # 4. Efficiency Metrics
        ax4 = axes[1, 1]
        if (len(self.training_metrics['episode_rewards']) > 0 and 
            len(self.training_metrics['episode_lengths']) > 0):
            
            # Calculate efficiency (reward per step)
            efficiency = [r/l if l > 0 else 0 for r, l in 
                         zip(self.training_metrics['episode_rewards'],
                             self.training_metrics['episode_lengths'])]
            
            ax4.plot(efficiency, alpha=0.3, label='Raw')
            if len(efficiency) > 50:
                smoothed = np.convolve(efficiency, np.ones(50)/50, mode='valid')
                ax4.plot(smoothed, label='Smoothed (50)')
            
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward per Step')
            ax4.set_title('Learning Efficiency')
            ax4.legend()
        
        plt.tight_layout()
        
        # Save
        detail_path = self.save_dir / f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detailed analysis plots saved to {detail_path}")


def run_training_experiment(config_name: str = 'standard'):
    """Run a complete training experiment"""
    
    print("🔬 IMPROVED TRAINING EXPERIMENT")
    print("=" * 60)
    
    # Training configurations
    training_configs = {
        'debug': {
            'num_episodes': 50,
            'agent_config': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'memory_size': 10000,
                'target_update_freq': 100,
                'epsilon_decay': 0.99
            },
            'description': 'Quick debug run'
        },
        'standard': {
            'num_episodes': 500,
            'agent_config': {
                'learning_rate': 0.0003,
                'batch_size': 64,
                'memory_size': 50000,
                'target_update_freq': 500,
                'epsilon_decay': 0.9995
            },
            'description': 'Standard training configuration'
        },
        'intensive': {
            'num_episodes': 2000,
            'agent_config': {
                'learning_rate': 0.0001,
                'batch_size': 128,
                'memory_size': 100000,
                'target_update_freq': 1000,
                'epsilon_decay': 0.9998
            },
            'description': 'Intensive training for best results'
        }
    }
    
    if config_name not in training_configs:
        print(f"Unknown configuration: {config_name}")
        print(f"Available: {list(training_configs.keys())}")
        return
    
    selected_config = training_configs[config_name]
    print(f"\nUsing configuration: {config_name}")
    print(f"Description: {selected_config['description']}")
    print(f"Episodes: {selected_config['num_episodes']}")
    
    # Setup
    config = Config()
    
    # Initialize components
    print("\n Initializing components...")
    
    integration_manager = TutorIntegrationManager(config)
    if not integration_manager.initialize():
        print("❌ Integration Manager initialization failed")
        return
    
    # Create environment
    content_library = create_enhanced_content_library()
    env = AdaptiveLearningEnvironment(
        integration_manager=integration_manager,
        content_library=content_library,
        config=config,
        max_session_time=60, 
    )
    
    print(f" Environment ready: {len(content_library)} contents")
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = ImprovedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=selected_config['agent_config']
    )
    
    print(f" Agent ready: {state_dim}D state -> {action_dim}D action")
    
    # Create training pipeline
    pipeline = ImprovedTrainingPipeline(
        env=env,
        agent=agent,
        config=config,
        save_dir=f"improved_models_{config_name}"
    )
    
    # Run training
    print(f"\n Starting training...")
    print(f"This may take a while. Check logs in: {pipeline.save_dir}/logs/")
    
    try:
        results = pipeline.train(num_episodes=selected_config['num_episodes'])
        
        print(f"\n Training completed!")
        print(f"Best eval reward: {results['best_eval_reward']:.2f}")
        print(f"Final performance: {np.mean(results['episode_performances'][-50:]):.3f}")
        
        return pipeline, agent, results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_improved_training():
    """Test the improved training pipeline"""
    
    print("🧪 TESTING IMPROVED TRAINING PIPELINE")
    print("=" * 60)
    
    # Quick test with debug configuration
    pipeline, agent, results = run_training_experiment('debug')
    
    if pipeline is not None:
        print("\n TEST RESULTS:")
        print(f"Episodes completed: {len(results['episode_rewards'])}")
        print(f"Average final reward: {np.mean(results['episode_rewards'][-10:]):.2f}")
        print(f"Average final performance: {np.mean(results['episode_performances'][-10:]):.3f}")       
        print("\n IMPROVED TRAINING TEST SUCCESSFUL!")
        print("\n Next steps:")
        print("1. Run standard training: run_training_experiment('standard')")
        print("2. Run intensive training: run_training_experiment('intensive')")
        print("3. Analyze results in the saved plots and metrics")
    else:
        print("\n❌ Training test failed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved DQN Training')
    parser.add_argument('--config', type=str, default='standard',
                        choices=['debug', 'standard', 'intensive'],
                        help='Training configuration to use')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.test:
        test_improved_training()
    else:
        if args.resume:
            print(f"Resuming from: {args.resume}")
        
        pipeline, agent, results = run_training_experiment(args.config)