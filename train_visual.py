"""
Script de entrenamiento CON visualización en tiempo real.
ADVERTENCIA: Será MUCHO más lento que el entrenamiento normal.
"""
import os
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import make_preprocessed_env
from model import create_model, save_model, print_model_info, RewardLogger


def train_with_render(
    rom_path="POKEMON_RED.gb",
    total_timesteps=50_000,
    save_freq=10_000,
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    model_name="pokemon_visual",
    learning_rate=3e-4,
):
    """
    Entrenar con ventana visible (más lento pero puedes ver al agente).
    """
    
    print("=" * 80)
    print("ENTRENAMIENTO VISUAL - POKÉMON RED")
    print("=" * 80)
    print("⚠️  ADVERTENCIA: Será mucho más lento que entrenamiento normal")
    print("⚠️  Recomendado solo para demos cortas o debugging")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 80)
    
    # Crear directorios
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Crear entorno CON ventana
    print("\nCreando entorno con visualización...")
    env = make_preprocessed_env(
        rom_path=rom_path,
        render_mode="human",  # Mostrar ventana
        max_steps=5000,
        grayscale=True,
        resize=True,
        normalize=True,
        frame_stack=4,
        skip_frames=4
    )
    env = Monitor(env, filename=f"{log_dir}/visual_training.csv")
    print("✓ Entorno creado (con ventana)")
    
    # Crear modelo
    print("\nCreando modelo PPO...")
    model = create_model(
        env=env,
        learning_rate=learning_rate,
        n_steps=512,  # Menos steps para actualizar más seguido
        batch_size=64,
        verbose=1
    )
    print("✓ Modelo creado")
    
    print_model_info(model)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=model_name,
        verbose=1
    )
    
    reward_logger = RewardLogger(verbose=1)
    
    # Entrenar
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO INICIADO - Observa la ventana del juego")
    print("Presiona Ctrl+C para detener")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, reward_logger],
            tb_log_name=model_name,
            progress_bar=True
        )
        
        print("\n" + "=" * 80)
        print("✓ ENTRENAMIENTO COMPLETADO")
        print("=" * 80)
        
        # Guardar modelo final
        save_model(model, checkpoint_dir, f"{model_name}_final")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido")
        print("Guardando modelo...")
        save_model(model, checkpoint_dir, f"{model_name}_interrupted")
    
    finally:
        env.close()
        print("\n✓ Ventana cerrada")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar con visualización")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total timesteps")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Frecuencia de guardado")
    
    args = parser.parse_args()
    
    train_with_render(
        total_timesteps=args.timesteps,
        save_freq=args.save_freq
    )
