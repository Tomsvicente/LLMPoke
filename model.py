"""
Configuración del modelo de Reinforcement Learning.
Usa PPO (Proximal Policy Optimization) de Stable-Baselines3.
"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
import os


class PokemonCNN(BaseFeaturesExtractor):
    """
    Red neuronal convolucional personalizada para Pokémon.
    Procesa las imágenes del juego para extraer features relevantes.
    
    Arquitectura:
    - 3 capas convolucionales con ReLU
    - Flatten
    - 2 capas fully connected
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Obtener dimensiones de entrada
        n_input_channels = observation_space.shape[2]  # 4 frames stacked
        
        # Red convolucional
        self.cnn = nn.Sequential(
            # Conv layer 1: 84x84x4 -> 20x20x32
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Conv layer 2: 20x20x32 -> 9x9x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # Conv layer 3: 9x9x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Flatten
            nn.Flatten(),
        )
        
        # Calcular tamaño después de convoluciones
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape).permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_input).shape[1]
        
        # Capas fully connected
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red.
        observations: [batch_size, height, width, channels]
        """
        # Cambiar de (B, H, W, C) a (B, C, H, W) para PyTorch
        observations = observations.permute(0, 3, 1, 2)
        
        # Pasar por CNN
        features = self.cnn(observations)
        
        # Pasar por fully connected
        features = self.linear(features)
        
        return features


class RewardLogger(BaseCallback):
    """
    Callback personalizado para logging de métricas durante el entrenamiento.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """
        Llamado en cada step del entrenamiento.
        """
        # Acumular reward
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Si el episodio terminó
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log en TensorBoard
            self.logger.record('rollout/ep_reward', self.current_episode_reward)
            self.logger.record('rollout/ep_length', self.current_episode_length)
            
            # Reset
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


def create_model(env, learning_rate=3e-4, n_steps=2048, batch_size=64, 
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 device='auto', verbose=1):
    """
    Crear modelo PPO con hiperparámetros optimizados para Pokémon.
    
    Args:
        env: Entorno de Gymnasium
        learning_rate: Tasa de aprendizaje (default: 3e-4)
        n_steps: Pasos por actualización (default: 2048)
        batch_size: Tamaño del batch (default: 64)
        n_epochs: Épocas por actualización (default: 10)
        gamma: Factor de descuento (default: 0.99)
        gae_lambda: Lambda para GAE (default: 0.95)
        clip_range: Rango de clipping PPO (default: 0.2)
        ent_coef: Coeficiente de entropía (exploración) (default: 0.01)
        vf_coef: Coeficiente de value function (default: 0.5)
        max_grad_norm: Max gradient norm (default: 0.5)
        device: 'cuda', 'cpu' o 'auto'
        verbose: Nivel de verbosidad
    
    Returns:
        Modelo PPO configurado
    """
    
    # Política con CNN personalizada
    policy_kwargs = dict(
        features_extractor_class=PokemonCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Policy y Value networks
    )
    
    # Crear modelo PPO
    model = PPO(
        policy='CnnPolicy',
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tensorboard/",
        device=device,
        verbose=verbose
    )
    
    return model


def load_model(path, env=None):
    """
    Cargar modelo entrenado desde archivo.
    
    Args:
        path: Ruta al archivo .zip del modelo
        env: Entorno (opcional, para continuar entrenamiento)
    
    Returns:
        Modelo cargado
    """
    model = PPO.load(path, env=env)
    print(f"✓ Modelo cargado desde: {path}")
    return model


def save_model(model, path, name="pokemon_agent"):
    """
    Guardar modelo entrenado.
    
    Args:
        model: Modelo a guardar
        path: Directorio donde guardar
        name: Nombre base del archivo
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, name)
    model.save(full_path)
    print(f"✓ Modelo guardado en: {full_path}.zip")


def get_model_info(model):
    """
    Obtener información del modelo.
    
    Returns:
        Dict con información del modelo
    """
    info = {
        'learning_rate': model.learning_rate,
        'n_steps': model.n_steps,
        'batch_size': model.batch_size,
        'n_epochs': model.n_epochs,
        'gamma': model.gamma,
        'gae_lambda': model.gae_lambda,
        'clip_range': model.clip_range,
        'ent_coef': model.ent_coef,
        'policy': model.policy.__class__.__name__,
        'device': model.device,
    }
    return info


def print_model_info(model):
    """
    Imprimir información del modelo de forma legible.
    """
    info = get_model_info(model)
    
    print("=" * 70)
    print("CONFIGURACIÓN DEL MODELO PPO")
    print("=" * 70)
    print(f"Policy:            {info['policy']}")
    print(f"Device:            {info['device']}")
    print(f"Learning Rate:     {info['learning_rate']}")
    print(f"Steps per update:  {info['n_steps']}")
    print(f"Batch size:        {info['batch_size']}")
    print(f"Epochs:            {info['n_epochs']}")
    print(f"Gamma (discount):  {info['gamma']}")
    print(f"GAE Lambda:        {info['gae_lambda']}")
    print(f"Clip range:        {info['clip_range']}")
    print(f"Entropy coef:      {info['ent_coef']}")
    print("=" * 70)
