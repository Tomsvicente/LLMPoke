"""
Script de entrenamiento principal para el agente de Pokémon Red.
"""
import os
import argparse
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import make_training_env
from model import create_model, save_model, print_model_info, RewardLogger


def train(
    rom_path="POKEMON_RED.gb",
    total_timesteps=1_000_000,
    save_freq=50_000,
    eval_freq=10_000,
    n_eval_episodes=5,
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    model_name="pokemon_ppo",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    device='auto',
    load_checkpoint=None
):
    """
    Entrenar agente de RL para jugar Pokémon Red.
    
    Args:
        rom_path: Ruta al ROM de Pokémon Red
        total_timesteps: Total de steps de entrenamiento
        save_freq: Frecuencia de guardado de checkpoints
        eval_freq: Frecuencia de evaluación
        n_eval_episodes: Episodios por evaluación
        checkpoint_dir: Directorio para checkpoints
        log_dir: Directorio para logs
        model_name: Nombre del modelo
        learning_rate: Learning rate del optimizador
        n_steps: Steps por actualización
        batch_size: Tamaño del batch
        device: 'cuda', 'cpu' o 'auto'
        load_checkpoint: Ruta a checkpoint para continuar entrenamiento
    """
    
    print("=" * 80)
    print("ENTRENAMIENTO DE AGENTE RL - POKÉMON RED")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Checkpoints: cada {save_freq:,} steps")
    print(f"Evaluación: cada {eval_freq:,} steps")
    print("=" * 80)
    
    # Crear directorios
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
    
    # Crear entorno de entrenamiento
    print("\n1. Creando entorno de entrenamiento...")
    env = make_training_env(rom_path=rom_path, max_steps=10000)
    env = Monitor(env, filename=f"{log_dir}/training_monitor.csv")
    print("   ✓ Entorno creado")
    
    # Crear entorno de evaluación
    print("\n2. Creando entorno de evaluación...")
    eval_env = make_training_env(rom_path=rom_path, max_steps=10000)
    eval_env = Monitor(eval_env, filename=f"{log_dir}/eval_monitor.csv")
    print("   ✓ Entorno de evaluación creado")
    
    # Crear o cargar modelo
    if load_checkpoint:
        print(f"\n3. Cargando modelo desde checkpoint: {load_checkpoint}")
        from model import load_model
        model = load_model(load_checkpoint, env=env)
    else:
        print("\n3. Creando nuevo modelo PPO...")
        model = create_model(
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            device=device,
            verbose=1
        )
        print("   ✓ Modelo creado")
    
    # Mostrar configuración del modelo
    print_model_info(model)
    
    # Callbacks
    print("\n4. Configurando callbacks...")
    
    # Checkpoint callback - guardar modelo periódicamente
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=model_name,
        verbose=1
    )
    
    # Eval callback - evaluar modelo periódicamente
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Reward logger
    reward_logger = RewardLogger(verbose=0)
    
    callbacks = [checkpoint_callback, eval_callback, reward_logger]
    print("   ✓ Callbacks configurados")
    
    # Entrenar
    print("\n5. Iniciando entrenamiento...")
    print("=" * 80)
    print("TIP: Monitorea el progreso con TensorBoard:")
    print(f"     tensorboard --logdir={log_dir}/tensorboard")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=model_name,
            reset_num_timesteps=(load_checkpoint is None),
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
        print(f"Checkpoints guardados en: {checkpoint_dir}")
        print(f"Mejor modelo guardado en: {log_dir}/best_model")
        print(f"Logs en: {log_dir}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido por el usuario")
        print("Guardando modelo actual...")
        save_model(model, checkpoint_dir, f"{model_name}_interrupted")
        env.close()
        eval_env.close()
        print("✓ Modelo guardado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar agente de RL para Pokémon Red")
    
    parser.add_argument("--rom", type=str, default="POKEMON_RED.gb", help="Ruta al ROM")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total de timesteps")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Frecuencia de guardado")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Frecuencia de evaluación")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Cargar desde checkpoint")
    parser.add_argument("--name", type=str, default="pokemon_ppo", help="Nombre del modelo")
    
    args = parser.parse_args()
    
    train(
        rom_path=args.rom,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.lr,
        device=args.device,
        load_checkpoint=args.checkpoint,
        model_name=args.name
    )
