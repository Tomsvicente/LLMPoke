"""
Entrenamiento paralelo con múltiples instancias del juego.
Aprende MUCHO más rápido usando varios entornos a la vez.
"""
import os
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv  # Cambiado de SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import make_training_env
from model import create_model, save_model, print_model_info, RewardLogger


def make_env(rom_path, rank, seed=0):
    """
    Función helper para crear entorno (necesaria para SubprocVecEnv).
    Cada entorno corre en su propio proceso.
    """
    def _init():
        from pokemon_env import PokemonRedEnv
        # Crear environment base SIN ventana
        base_env = PokemonRedEnv(rom_path=rom_path, render_mode=None, max_steps=10000)
        # Aplicar wrappers de preprocesamiento
        from wrappers import (GrayscaleObservation, ResizeObservation, 
                             NormalizeObservation, FrameStack, SkipFrame)
        env = SkipFrame(base_env, skip=4)
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, shape=(84, 84))
        env = NormalizeObservation(env)
        env = FrameStack(env, num_stack=4)
        # Monitor
        env = Monitor(env, filename=f"./logs/monitor_{rank}.csv")
        env.reset(seed=seed + rank)
        return env
    return _init


def train_parallel(
    rom_path="POKEMON_RED.gb",
    n_envs=4,  # Número de instancias paralelas
    total_timesteps=1_000_000,
    save_freq=100_000,
    eval_freq=20_000,
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    model_name="pokemon_parallel",
    learning_rate=3e-4,
    device='auto'
):
    """
    Entrenar con múltiples instancias del juego en paralelo.
    
    Args:
        n_envs: Número de instancias paralelas (recomendado: 4-8)
                Más instancias = más rápido, pero más RAM/CPU
        Resto de parámetros iguales a train.py
    """
    
    print("=" * 80)
    print("ENTRENAMIENTO PARALELO - POKÉMON RED")
    print("=" * 80)
    print(f"🚀 Instancias paralelas: {n_envs}")
    print(f"⚡ Aceleración estimada: ~{n_envs}x más rápido")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 80)
    
    # Crear directorios
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
    
    # Crear entornos paralelos
    print(f"\n1. Creando {n_envs} entornos paralelos...")
    print("   (Esto puede tardar un poco...)")
    
    env = DummyVecEnv([  # Cambiado de SubprocVecEnv
        make_env(rom_path, i) for i in range(n_envs)
    ])
    
    print(f"   ✓ {n_envs} entornos creados (cada uno en su propio proceso)")
    
    # Crear entorno de evaluación (solo 1)
    print("\n2. Creando entorno de evaluación...")
    from pokemon_env import PokemonRedEnv
    from wrappers import (GrayscaleObservation, ResizeObservation, 
                         NormalizeObservation, FrameStack, SkipFrame)
    base_eval = PokemonRedEnv(rom_path=rom_path, render_mode=None, max_steps=10000)
    eval_env = SkipFrame(base_eval, skip=4)
    eval_env = GrayscaleObservation(eval_env)
    eval_env = ResizeObservation(eval_env, shape=(84, 84))
    eval_env = NormalizeObservation(eval_env)
    eval_env = FrameStack(eval_env, num_stack=4)
    eval_env = Monitor(eval_env, filename=f"{log_dir}/eval_monitor.csv")
    print("   ✓ Entorno de evaluación creado")
    
    # Crear modelo
    print("\n3. Creando modelo PPO...")
    model = create_model(
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,  # Steps por env antes de actualizar
        batch_size=64,
        device=device,
        verbose=1
    )
    print("   ✓ Modelo creado")
    
    print_model_info(model)
    
    # Información adicional
    print("\n" + "=" * 80)
    print("CONFIGURACIÓN PARALELA")
    print("=" * 80)
    print(f"Entornos paralelos:    {n_envs}")
    print(f"Steps por env:         {2048}")
    print(f"Steps totales/update:  {2048 * n_envs} ({n_envs}x más datos)")
    print(f"Updates por epoch:     ~{total_timesteps // (2048 * n_envs)}")
    print("=" * 80)
    
    # Callbacks
    print("\n4. Configurando callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Ajustar por número de envs
        save_path=checkpoint_dir,
        name_prefix=model_name,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=eval_freq // n_envs,  # Ajustar por número de envs
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    print("   ✓ Callbacks configurados")
    
    # Entrenar
    print("\n5. Iniciando entrenamiento paralelo...")
    print("=" * 80)
    print("TIP: Monitorea con TensorBoard:")
    print(f"     tensorboard --logdir={log_dir}/tensorboard")
    print("=" * 80)
    print(f"📊 Con {n_envs} entornos, cada iteración recolecta {n_envs}x más experiencia")
    print(f"⏱️  Tiempo estimado: ~{total_timesteps // (n_envs * 50) // 60} minutos")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=model_name,
            progress_bar=True
        )
        
        print("\n" + "=" * 80)
        print("✓ ENTRENAMIENTO COMPLETADO")
        print("=" * 80)
        
        # Guardar modelo final
        print("\n6. Guardando modelo final...")
        save_model(model, checkpoint_dir, f"{model_name}_final")
        
        # Cerrar entornos
        env.close()
        eval_env.close()
        
        print("\n" + "=" * 80)
        print("RESUMEN")
        print("=" * 80)
        print(f"Timesteps entrenados: {total_timesteps:,}")
        print(f"Instancias usadas: {n_envs}")
        print(f"Aceleración: ~{n_envs}x vs. entrenamiento normal")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Mejor modelo: {log_dir}/best_model")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido")
        print("Guardando modelo...")
        save_model(model, checkpoint_dir, f"{model_name}_interrupted")
        env.close()
        eval_env.close()
        print("✓ Modelo guardado")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar con múltiples instancias paralelas")
    
    parser.add_argument("--rom", type=str, default="POKEMON_RED.gb", help="Ruta al ROM")
    parser.add_argument("--envs", type=int, default=4, help="Número de instancias paralelas (4-8 recomendado)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total de timesteps")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Frecuencia de guardado")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Frecuencia de evaluación")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--name", type=str, default="pokemon_parallel", help="Nombre del modelo")
    
    args = parser.parse_args()
    
    train_parallel(
        rom_path=args.rom,
        n_envs=args.envs,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.lr,
        device=args.device,
        model_name=args.name
    )
