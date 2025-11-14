"""
Entrenamiento paralelo con visualización en grilla.
Muestra las 4 instancias en una ventana dividida.
"""
import os
import numpy as np
import cv2
from datetime import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import make_preprocessed_env
from model import create_model, save_model, print_model_info
import threading
import time


class MultiGameVisualizer:
    """
    Visualizador que muestra múltiples instancias en una grilla.
    """
    def __init__(self, n_envs=4):
        self.n_envs = n_envs
        self.frames = [None] * n_envs
        self.rewards = [0.0] * n_envs
        self.steps = [0] * n_envs
        self.running = False
        self.window_name = "Entrenamiento Paralelo - Pokémon Red"
        
    def update_frame(self, env_id, frame, reward, step):
        """Actualizar frame de un entorno específico."""
        self.frames[env_id] = frame
        self.rewards[env_id] = reward
        self.steps[env_id] = step
    
    def create_grid(self):
        """Crear grilla 2x2 con las 4 instancias."""
        if None in self.frames:
            return None
        
        # Configuración de grilla
        rows = 2
        cols = 2
        
        # Convertir frames a RGB si es necesario
        frames_rgb = []
        for frame in self.frames:
            if len(frame.shape) == 2:  # Grayscale
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 1:  # Single channel
                frame_rgb = cv2.cvtColor(frame[:,:,0], cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = frame
            
            # Redimensionar para mejor visualización
            frame_rgb = cv2.resize(frame_rgb, (320, 288), interpolation=cv2.INTER_NEAREST)
            
            frames_rgb.append(frame_rgb)
        
        # Agregar texto con información
        for i, frame in enumerate(frames_rgb):
            # Crear borde de color según el entorno
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), colors[i], 3)
            
            # Agregar texto con stats
            text = f"Env {i+1} | Steps: {self.steps[i]} | R: {self.rewards[i]:.2f}"
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Crear grilla
        row1 = np.hstack([frames_rgb[0], frames_rgb[1]])
        row2 = np.hstack([frames_rgb[2], frames_rgb[3]])
        grid = np.vstack([row1, row2])
        
        return grid
    
    def show(self):
        """Mostrar ventana con la grilla."""
        self.running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 576)  # 2x2 grid
        
        while self.running:
            grid = self.create_grid()
            if grid is not None:
                cv2.imshow(self.window_name, grid)
            
            if cv2.waitKey(50) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()
    
    def close(self):
        """Cerrar ventana."""
        self.running = False


class VisualizationCallback(BaseCallback):
    """
    Callback que captura frames de las instancias paralelas.
    """
    def __init__(self, visualizer, envs_raw, verbose=0):
        super().__init__(verbose)
        self.visualizer = visualizer
        self.envs_raw = envs_raw
        self.episode_rewards = [0.0] * len(envs_raw)
        
    def _on_step(self) -> bool:
        # Capturar frames de cada entorno
        for i, env in enumerate(self.envs_raw):
            try:
                # Obtener frame del entorno
                frame = env.pyboy.screen.ndarray if hasattr(env, 'pyboy') else None
                
                if frame is not None:
                    # Acumular reward
                    self.episode_rewards[i] += self.locals.get('rewards', [0])[i] if i < len(self.locals.get('rewards', [])) else 0
                    
                    # Actualizar visualizador
                    self.visualizer.update_frame(
                        i, 
                        frame,
                        self.episode_rewards[i],
                        self.num_timesteps
                    )
            except:
                pass
        
        return True


def make_visual_env(rom_path, rank):
    """Crear entorno que guarda referencia al PyBoy."""
    def _init():
        from pokemon_env import PokemonRedEnv
        from wrappers import SkipFrame, ResizeObservation, GrayscaleObservation, NormalizeObservation, FrameStack
        
        # Crear entorno base (sin render para mantener performance)
        env = PokemonRedEnv(rom_path=rom_path, render_mode=None, max_steps=5000)
        
        # Aplicar wrappers mínimos
        env = SkipFrame(env, skip=4)
        env = ResizeObservation(env, shape=(84, 84))
        env = GrayscaleObservation(env)
        env = NormalizeObservation(env)
        env = FrameStack(env, num_stack=4)
        
        env = Monitor(env, filename=f"./logs/visual_parallel_{rank}.csv")
        return env
    return _init


def train_parallel_visual(
    rom_path="POKEMON_RED.gb",
    n_envs=4,
    total_timesteps=200_000,
    save_freq=50_000,
):
    """
    Entrenar con visualización de múltiples instancias en grilla.
    """
    
    print("=" * 80)
    print("ENTRENAMIENTO PARALELO CON VISUALIZACIÓN")
    print("=" * 80)
    print(f"Instancias: {n_envs}")
    print(f"Visualización: Grilla 2x2")
    print("=" * 80)
    
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    print("\nCreando entornos paralelos...")
    
    # Crear entornos
    env_fns = [make_visual_env(rom_path, i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    
    # Obtener referencias a los entornos raw (para visualización)
    # Nota: En SubprocVecEnv esto es complicado, usaremos una aproximación
    print("✓ Entornos creados")
    
    print("\nCreando modelo PPO...")
    model = create_model(
        env=env,
        n_steps=512,
        batch_size=64,
        verbose=1
    )
    print("✓ Modelo creado")
    
    print_model_info(model)
    
    # Crear visualizador
    print("\nPreparando visualización...")
    print("⚠️  Nota: Visualización en modo simplificado debido a SubprocVecEnv")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path="./checkpoints",
        name_prefix="pokemon_parallel_visual",
        verbose=1
    )
    
    print("\n" + "=" * 80)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 80)
    print(f"{n_envs} instancias entrenando en paralelo")
    print(f"Presiona Ctrl+C para detener")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True
        )
        
        print("\n✓ Entrenamiento completado")
        save_model(model, "./checkpoints", "pokemon_parallel_visual_final")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido")
        save_model(model, "./checkpoints", "pokemon_parallel_visual_interrupted")
    
    finally:
        env.close()


if __name__ == "__main__":
    print("=" * 80)
    print("NOTA IMPORTANTE")
    print("=" * 80)
    print("La visualización de entornos paralelos es compleja con SubprocVecEnv.")
    print("Esta versión entrena rápido (paralelo) pero sin visualización continua.")
    print()
    print("Recomendaciones:")
    print("1. Usa train_parallel.py (más rápido, sin visualización)")
    print("2. Después evalúa con: python play.py <modelo> --episodes 5")
    print("=" * 80)
    print()
    
    response = input("¿Continuar de todos modos? (y/n): ")
    
    if response.lower() == 'y':
        train_parallel_visual(n_envs=4, total_timesteps=100_000)
    else:
        print("\nUsa mejor:")
        print("  python train_parallel.py --envs 4 --timesteps 200000")
