"""
Script para probar el entorno de Pokémon Red.
Ejecuta acciones aleatorias para verificar que todo funciona correctamente.
"""
from pokemon_env import PokemonRedEnv
import numpy as np


def test_random_actions(num_episodes=2, steps_per_episode=100):
    """
    Probar el entorno con acciones aleatorias.
    """
    print("=" * 60)
    print("PROBANDO POKEMON RED ENVIRONMENT")
    print("=" * 60)
    
    # Crear entorno
    print("\n1. Creando entorno...")
    env = PokemonRedEnv(rom_path="POKEMON_RED.gb", render_mode="human")
    print(f"   ✓ Entorno creado!")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    
    try:
        for episode in range(num_episodes):
            print(f"\n{'=' * 60}")
            print(f"EPISODIO {episode + 1}/{num_episodes}")
            print(f"{'=' * 60}")
            
            # Reset del entorno
            observation, info = env.reset()
            print(f"\n2. Entorno reseteado!")
            print(f"   - Observation shape: {observation.shape}")
            print(f"   - Info: {info}")
            
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Acción aleatoria
                action = env.action_space.sample()
                
                # Ejecutar acción
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Mostrar progreso cada 20 steps
                if (step + 1) % 20 == 0:
                    print(f"\n   Step {step + 1}/{steps_per_episode}")
                    print(f"   - Acción: {info['action_taken']}")
                    print(f"   - Reward acumulado: {episode_reward:.4f}")
                    print(f"   - Observation shape: {observation.shape}")
                
                # Verificar si terminó
                if terminated:
                    print(f"\n   ¡Episodio terminado en step {step + 1}!")
                    break
                    
                if truncated:
                    print(f"\n   ¡Episodio truncado en step {step + 1}!")
                    break
            
            print(f"\n{'=' * 60}")
            print(f"RESUMEN EPISODIO {episode + 1}")
            print(f"{'=' * 60}")
            print(f"Reward total: {episode_reward:.4f}")
            print(f"Steps completados: {step + 1}")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Prueba interrumpida por el usuario")
    
    finally:
        print("\n" + "=" * 60)
        print("CERRANDO ENTORNO")
        print("=" * 60)
        env.close()
        print("\n✓ Prueba completada!")


if __name__ == "__main__":
    test_random_actions(num_episodes=2, steps_per_episode=100)
