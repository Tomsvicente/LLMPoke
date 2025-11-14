"""
Entrenamiento con 4 instancias, cada una con su propia ventana PyBoy.
La forma más simple y directa de ver 4 emuladores ejecutándose en paralelo.
"""

# IMPORTAR TODO PRIMERO - ANTES de crear ventanas SDL
import argparse
import os
import time
import sys

# Forzar import completo de todas las dependencias
print("📦 Cargando dependencias...")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from pokemon_env import PokemonRedEnv
from wrappers import make_training_env
from model import create_model, RewardLogger

print("✅ Dependencias cargadas")


def make_visible_env(rank, rom_path, seed=0):
    """
    Crea un environment CON ventana visible.
    
    Args:
        rank: ID del environment
        rom_path: Path a la ROM
        seed: Semilla
    """
    def _init():
        # Crear env base CON ventana visible
        base_env = PokemonRedEnv(rom_path=rom_path, render_mode="human")
        
        # Aplicar wrappers de entrenamiento
        env = make_training_env(base_env)
        
        # Monitor para estadísticas
        log_dir = f"./logs/monitor_4w_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        env.reset(seed=seed + rank)
        return env
    
    return _init


def train_with_4_windows(args):
    """Entrena con 4 ventanas PyBoy separadas."""
    
    print("🎮 Iniciando entrenamiento con 4 ventanas PyBoy...")
    print(f"📦 ROM: {args.rom}")
    print(f"⏱️  Timesteps: {args.timesteps:,}")
    print()
    print("💡 Se abrirán 4 ventanas de PyBoy - organízalas en tu pantalla")
    print()
    
    # Crear directorios
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Crear 4 environments con ventanas visibles
    print("🔧 Creando 4 environments con ventanas...")
    n_envs = 4
    
    env_fns = [make_visible_env(i, args.rom, seed=args.seed) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    print("✅ 4 ventanas creadas!")
    time.sleep(2)  # Dar tiempo para organizar ventanas
    
    # Environment de evaluación (sin ventana para velocidad)
    eval_env_fn = lambda: make_training_env(PokemonRedEnv(rom_path=args.rom, render_mode=None))
    eval_env = DummyVecEnv([eval_env_fn])
    
    # Crear modelo
    if args.checkpoint:
        print(f"📂 Cargando modelo: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=vec_env)
    else:
        print("🆕 Creando nuevo modelo PPO...")
        model = create_model(
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            device=args.device,
            tensorboard_log="./logs/tensorboard_4w"
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // n_envs,
        save_path="./checkpoints",
        name_prefix=f"pokemon_4w_{args.name}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/best_model_{args.name}",
        log_path=f"./logs/eval_{args.name}",
        eval_freq=args.eval_freq // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=3,
    )
    
    reward_logger = RewardLogger()
    
    print()
    print("🚀 Iniciando entrenamiento...")
    print("👀 Observa las 4 ventanas para ver el progreso!")
    print()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback, reward_logger],
            progress_bar=True
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido")
    finally:
        # Guardar modelo final
        final_path = f"./checkpoints/pokemon_4w_{args.name}_final.zip"
        model.save(final_path)
        print(f"\n💾 Modelo guardado: {final_path}")
        
        # Cerrar
        vec_env.close()
        eval_env.close()
    
    print("\n✅ Entrenamiento completado!")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento con 4 ventanas PyBoy visibles")
    parser.add_argument("--rom", type=str, default="POKEMON_RED.GB")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--save-freq", type=int, default=25000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--name", type=str, default="4win")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train_with_4_windows(args)


if __name__ == "__main__":
    main()
