"""
Script de entrenamiento con 4 instancias visibles en paralelo.
Muestra una ventana con las 4 instancias del juego en una cuadrícula 2x2.
"""

import argparse
import os
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from pokemon_env import PokemonRedEnv
from wrappers import make_training_env
from model import create_model, RewardLogger


class QuadVisualizer:
    """Visualizador que muestra 4 instancias del juego en una cuadrícula 2x2."""
    
    def __init__(self, window_name="Pokemon RL - 4 Instancias"):
        self.window_name = window_name
        self.frame_count = 0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 960)  # 640x480 por cuadrante
        
    def create_grid(self, frames, stats_list):
        """
        Crea una cuadrícula 2x2 con los frames de las 4 instancias.
        
        Args:
            frames: Lista de 4 frames (cada uno 144x160x3)
            stats_list: Lista de 4 diccionarios con estadísticas
        """
        grid_frames = []
        
        for i, (frame, stats) in enumerate(zip(frames, stats_list)):
            # Escalar frame a 640x480 (mantiene proporción aprox)
            display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            
            # Agregar texto con estadísticas
            y_offset = 30
            cv2.putText(display_frame, f"Instancia {i+1}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
            
            cv2.putText(display_frame, f"Episodio: {stats['episode']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25
            
            cv2.putText(display_frame, f"Pasos: {stats['steps']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25
            
            cv2.putText(display_frame, f"Reward: {stats['reward']:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            grid_frames.append(display_frame)
        
        # Crear cuadrícula 2x2
        top_row = np.hstack([grid_frames[0], grid_frames[1]])
        bottom_row = np.hstack([grid_frames[2], grid_frames[3]])
        grid = np.vstack([top_row, bottom_row])
        
        return grid
    
    def show(self, frames, stats_list):
        """Muestra la cuadrícula con las 4 instancias."""
        self.frame_count += 1
        
        # Solo actualizar cada 2 frames para reducir carga
        if self.frame_count % 2 != 0:
            return
        
        grid = self.create_grid(frames, stats_list)
        cv2.imshow(self.window_name, grid)
        cv2.waitKey(1)
    
    def close(self):
        """Cierra la ventana."""
        cv2.destroyAllWindows()


def make_monitored_env(rank, rom_path, window_type="null", seed=0):
    """
    Crea un environment con Monitor para tracking.
    
    Args:
        rank: ID del environment
        rom_path: Path a la ROM
        window_type: Tipo de ventana PyBoy ("null" para sin ventana)
        seed: Semilla para reproducibilidad
    """
    def _init():
        # Crear env base sin wrappers de visualización
        base_env = PokemonRedEnv(rom_path=rom_path, window_type=window_type)
        
        # Aplicar wrappers de entrenamiento
        env = make_training_env(base_env)
        
        # Wrappear con Monitor para estadísticas
        log_dir = f"./logs/monitor_quad_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        env.reset(seed=seed + rank)
        return env
    
    return _init


class VisualCallback:
    """Callback para capturar y visualizar frames de las 4 instancias."""
    
    def __init__(self, vec_env, rom_path, visualizer):
        self.vec_env = vec_env
        self.rom_path = rom_path
        self.visualizer = visualizer
        self.episode_counts = [0] * 4
        self.step_counts = [0] * 4
        self.episode_rewards = [0.0] * 4
        
        # Crear 4 environments adicionales solo para obtener frames
        self.render_envs = []
        for i in range(4):
            env = PokemonRedEnv(rom_path=rom_path, window_type="null")
            self.render_envs.append(env)
    
    def on_step(self):
        """Llamado después de cada step del entrenamiento."""
        frames = []
        stats_list = []
        
        # Obtener frame de cada instancia
        for i, render_env in enumerate(self.render_envs):
            try:
                # Obtener frame actual
                frame = render_env._get_screen()
                frames.append(frame)
                
                # Preparar estadísticas
                stats = {
                    'episode': self.episode_counts[i],
                    'steps': self.step_counts[i],
                    'reward': self.episode_rewards[i]
                }
                stats_list.append(stats)
                
            except Exception as e:
                # En caso de error, usar frame negro
                frames.append(np.zeros((144, 160, 3), dtype=np.uint8))
                stats_list.append({
                    'episode': 0,
                    'steps': 0,
                    'reward': 0.0
                })
        
        # Mostrar cuadrícula
        if len(frames) == 4:
            self.visualizer.show(frames, stats_list)
        
        return True
    
    def sync_with_training_envs(self, dones, rewards):
        """Sincroniza estadísticas con los environments de entrenamiento."""
        for i in range(4):
            self.step_counts[i] += 1
            self.episode_rewards[i] += rewards[i]
            
            if dones[i]:
                self.episode_counts[i] += 1
                self.episode_rewards[i] = 0.0
                self.step_counts[i] = 0
    
    def close(self):
        """Cierra los environments de renderizado."""
        for env in self.render_envs:
            env.close()


def train_with_quad_visual(args):
    """Entrena con 4 instancias visibles en paralelo."""
    
    print("🎮 Iniciando entrenamiento con 4 instancias visibles...")
    print(f"📦 ROM: {args.rom}")
    print(f"⏱️  Timesteps totales: {args.timesteps:,}")
    print(f"💾 Guardando cada: {args.save_freq:,} steps")
    print(f"📊 Evaluando cada: {args.eval_freq:,} steps")
    print()
    
    # Crear directorios
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Crear 4 environments con DummyVecEnv (mismo proceso)
    print("🔧 Creando 4 environments paralelos...")
    n_envs = 4
    env_fns = [make_monitored_env(i, args.rom, window_type="null", seed=args.seed) 
               for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    # Crear environment de evaluación
    eval_env = DummyVecEnv([make_monitored_env(99, args.rom, window_type="null", seed=args.seed + 1000)])
    
    # Crear visualizador
    visualizer = QuadVisualizer()
    
    # Crear o cargar modelo
    if args.checkpoint:
        print(f"📂 Cargando modelo desde: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=vec_env)
    else:
        print("🆕 Creando nuevo modelo PPO...")
        model = create_model(
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            device=args.device,
            tensorboard_log="./logs/tensorboard_quad"
        )
    
    # Configurar callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // n_envs,  # Ajustar por número de envs
        save_path="./checkpoints",
        name_prefix=f"pokemon_quad_{args.name}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/best_model_{args.name}",
        log_path=f"./logs/eval_{args.name}",
        eval_freq=args.eval_freq // n_envs,  # Ajustar por número de envs
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    
    reward_logger = RewardLogger()
    
    callbacks = [checkpoint_callback, eval_callback, reward_logger]
    
    print()
    print("🚀 Iniciando entrenamiento...")
    print("💡 Verás las 4 instancias en una ventana separada")
    print("⚠️  NOTA: DummyVecEnv es más lento que SubprocVecEnv pero permite visualización")
    print()
    
    try:
        # Usar el método learn estándar con callbacks
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Loop de visualización en paralelo
        obs = vec_env.reset()
        step_count = 0
        
        while step_count < 1000:  # Mostrar primeros 1000 steps
            # Predecir acciones
            actions, _ = model.predict(obs, deterministic=False)
            
            # Ejecutar step
            obs, rewards, dones, infos = vec_env.step(actions)
            
            # Obtener frames actuales de cada env cada 5 steps
            if step_count % 5 == 0:
                frames = []
                stats_list = []
                
                for i in range(n_envs):
                    try:
                        # Acceder al environment original
                        base_env = vec_env.envs[i]
                        while hasattr(base_env, 'env'):
                            base_env = base_env.env
                        
                        frame = base_env._get_screen()
                        frames.append(frame)
                        
                        # Estadísticas
                        episode_count = infos[i].get('episode', {})
                        stats = {
                            'episode': len(reward_logger.episode_rewards) if hasattr(reward_logger, 'episode_rewards') else 0,
                            'steps': step_count,
                            'reward': episode_count.get('r', 0.0) if episode_count else 0.0
                        }
                        stats_list.append(stats)
                    except Exception as e:
                        frames.append(np.zeros((144, 160, 3), dtype=np.uint8))
                        stats_list.append({'episode': 0, 'steps': 0, 'reward': 0.0})
                
                if len(frames) == n_envs:
                    visualizer.show(frames, stats_list)
            
            step_count += 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por el usuario")
    finally:
        # Guardar modelo final
        final_path = f"./checkpoints/pokemon_quad_{args.name}_final.zip"
        model.save(final_path)
        print(f"\n💾 Modelo final guardado en: {final_path}")
        
        # Cerrar todo
        visualizer.close()
        vec_env.close()
        eval_env.close()
    
    print("\n✅ Entrenamiento completado!")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento con 4 instancias visibles")
    parser.add_argument("--rom", type=str, default="POKEMON_RED.GB", help="Path a la ROM")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total de timesteps")
    parser.add_argument("--save-freq", type=int, default=25000, help="Guardar cada N steps")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluar cada N steps")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps por actualización PPO")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint para continuar")
    parser.add_argument("--name", type=str, default="visual", help="Nombre del experimento")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    
    args = parser.parse_args()
    
    train_with_quad_visual(args)


if __name__ == "__main__":
    main()
