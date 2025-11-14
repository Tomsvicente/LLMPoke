"""
Wrappers para preprocesar observaciones del entorno de Pokémon.
Optimizan las imágenes para aprendizaje más eficiente.
"""
import gymnasium as gym
import numpy as np
import cv2
from collections import deque


class GrayscaleObservation(gym.ObservationWrapper):
    """
    Convierte observaciones RGB a escala de grises.
    Reduce de 3 canales a 1, disminuyendo la complejidad.
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        # Nuevo shape: (height, width, 1) en vez de (height, width, 3)
        new_shape = (old_shape[0], old_shape[1], 1)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )
    
    def observation(self, obs):
        """Convertir RGB a escala de grises."""
        # Usar pesos estándar para conversión RGB -> Grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Añadir dimensión de canal
        return np.expand_dims(gray, axis=-1)


class ResizeObservation(gym.ObservationWrapper):
    """
    Redimensiona observaciones a un tamaño específico.
    Útil para reducir aún más la dimensionalidad.
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], old_shape[2]),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        """Redimensionar observación."""
        # cv2.resize espera (width, height) pero shape es (height, width)
        resized = cv2.resize(obs, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        # Si es grayscale, mantener dimensión de canal
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        return resized


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normaliza valores de píxeles de [0, 255] a [0, 1].
    Ayuda a la convergencia de la red neuronal.
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=old_shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Normalizar valores."""
        return obs.astype(np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """
    Stack de múltiples frames consecutivos.
    Permite a la red neuronal capturar movimiento y dirección.
    Por ejemplo, 4 frames apilados permiten ver la trayectoria.
    """
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        old_shape = env.observation_space.shape
        # Nuevo shape: apilar frames en el canal
        new_shape = (old_shape[0], old_shape[1], old_shape[2] * num_stack)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        """Reset y llenar frames con el frame inicial."""
        obs, info = self.env.reset(**kwargs)
        # Llenar con el mismo frame inicial
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        """Step y añadir nuevo frame al stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Concatenar frames en el eje de canales."""
        return np.concatenate(list(self.frames), axis=-1)


class SkipFrame(gym.Wrapper):
    """
    Ejecuta la misma acción durante N frames consecutivos.
    Reduce la frecuencia de decisiones y acelera el entrenamiento.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        """Repetir acción por skip frames."""
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info


def make_preprocessed_env(rom_path="POKEMON_RED.gb", render_mode=None, max_steps=10000,
                          grayscale=True, resize=True, normalize=True, 
                          frame_stack=4, skip_frames=4):
    """
    Crea un entorno de Pokémon con todos los preprocesos aplicados.
    
    Args:
        rom_path: Ruta al ROM
        render_mode: "human", "rgb_array" o None
        max_steps: Máximo de steps por episodio
        grayscale: Convertir a escala de grises
        resize: Redimensionar a 84x84
        normalize: Normalizar valores [0,1]
        frame_stack: Número de frames a apilar (0 = desactivado)
        skip_frames: Repetir acción N frames (0 = desactivado)
    
    Returns:
        Environment preprocesado listo para RL
    """
    from pokemon_env import PokemonRedEnv
    
    # Crear entorno base
    env = PokemonRedEnv(rom_path=rom_path, render_mode=render_mode, max_steps=max_steps)
    
    # Aplicar wrappers en orden
    if skip_frames > 0:
        env = SkipFrame(env, skip=skip_frames)
    
    if resize:
        env = ResizeObservation(env, shape=(84, 84))
    
    if grayscale:
        env = GrayscaleObservation(env)
    
    if normalize:
        env = NormalizeObservation(env)
    
    if frame_stack > 0:
        env = FrameStack(env, num_stack=frame_stack)
    
    return env


def make_training_env(rom_path="POKEMON_RED.gb", **kwargs):
    """
    Crea entorno optimizado para entrenamiento (sin render).
    Configuración por defecto: grayscale, 84x84, normalizado, 4 frames stacked.
    """
    return make_preprocessed_env(
        rom_path=rom_path,
        render_mode=None,
        grayscale=True,
        resize=True,
        normalize=True,
        frame_stack=4,
        skip_frames=4,
        **kwargs
    )


def make_eval_env(rom_path="POKEMON_RED.gb", **kwargs):
    """
    Crea entorno para evaluación (con render para visualizar).
    Misma configuración que training pero con ventana.
    """
    return make_preprocessed_env(
        rom_path=rom_path,
        render_mode="human",
        grayscale=True,
        resize=True,
        normalize=True,
        frame_stack=4,
        skip_frames=4,
        **kwargs
    )
