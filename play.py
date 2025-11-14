"""
Script para evaluar y visualizar el agente entrenado jugando.
"""
import argparse
import time
from wrappers import make_eval_env
from model import load_model


def play(model_path, rom_path="POKEMON_RED.gb", n_episodes=5, 
         max_steps=10000, delay=0.05, deterministic=True):
    """
    Ver al agente entrenado jugando Pokémon Red.
    
    Args:
        model_path: Ruta al modelo entrenado (.zip)
        rom_path: Ruta al ROM
        n_episodes: Número de episodios a jugar
        max_steps: Máximo de steps por episodio
        delay: Delay entre frames (segundos)
        deterministic: Usar política determinista o estocástica
    """
    
    print("=" * 80)
    print("EVALUACIÓN DEL AGENTE - POKÉMON RED")
    print("=" * 80)
    print(f"Modelo: {model_path}")
    print(f"Episodios: {n_episodes}")
    print(f"Modo: {'Determinista' if deterministic else 'Estocástico'}")
    print("=" * 80)
    
    # Crear entorno con ventana
    print("\nCreando entorno...")
    env = make_eval_env(rom_path=rom_path, max_steps=max_steps)
    print("✓ Entorno creado")
    
    # Cargar modelo
    print("\nCargando modelo...")
    model = load_model(model_path, env=env)
    
    # Jugar episodios
    for episode in range(n_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODIO {episode + 1}/{n_episodes}")
        print(f"{'='*80}")
        
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        
        print(f"Estado inicial:")
        print(f"  Posición: ({info['player_x']}, {info['player_y']})")
        print(f"  Mapa: {info['map_id']}")
        print(f"  Badges: {info['badges']}/8")
        print(f"  Level: {info['pokemon_level']}")
        
        while not (done or truncated):
            # Predecir acción con el modelo
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Ejecutar acción
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            # Mostrar progreso cada 100 steps
            if step % 100 == 0:
                print(f"\n[Step {step}]")
                print(f"  Pos: ({info['player_x']}, {info['player_y']}) | Mapa: {info['map_id']}")
                print(f"  Badges: {info['badges']}/8 | Level: {info['pokemon_level']}")
                print(f"  Reward acumulado: {total_reward:.4f}")
            
            # Delay para visualizar
            time.sleep(delay)
            
            if done:
                print(f"\n💀 Game Over en step {step}")
                break
            
            if truncated:
                print(f"\n⏱️ Límite de steps alcanzado ({step})")
                break
        
        # Resumen del episodio
        stats = env.get_progress_stats()
        print(f"\n{'='*80}")
        print(f"RESUMEN EPISODIO {episode + 1}")
        print(f"{'='*80}")
        print(f"Steps: {step}")
        print(f"Reward total: {total_reward:.4f}")
        print(f"Reward promedio: {total_reward/step if step > 0 else 0:.4f}")
        print(f"Mapas explorados: {stats['maps_visited']}")
        print(f"Badges: {stats['badges']}/8")
        print(f"Level máximo: {stats['level']}")
        print(f"Money: ${stats['money']}")
        print(f"{'='*80}")
        
        if episode < n_episodes - 1:
            print("\nPresiona Ctrl+C para detener...")
            time.sleep(2)
    
    env.close()
    print("\n✓ Evaluación completada!")


def evaluate(model_path, rom_path="POKEMON_RED.gb", n_episodes=10):
    """
    Evaluar modelo sin visualización (solo métricas).
    
    Args:
        model_path: Ruta al modelo
        rom_path: Ruta al ROM
        n_episodes: Número de episodios
    """
    from wrappers import make_training_env
    
    print("=" * 80)
    print("EVALUACIÓN RÁPIDA (Sin visualización)")
    print("=" * 80)
    
    # Crear entorno sin ventana
    env = make_training_env(rom_path=rom_path, max_steps=10000)
    model = load_model(model_path, env=env)
    
    episode_rewards = []
    episode_lengths = []
    badges_collected = []
    maps_explored = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
        
        stats = env.get_progress_stats()
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        badges_collected.append(stats['badges'])
        maps_explored.append(stats['maps_visited'])
        
        print(f"Episodio {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Steps={step}, Badges={stats['badges']}")
    
    env.close()
    
    # Estadísticas finales
    import numpy as np
    print(f"\n{'='*80}")
    print("ESTADÍSTICAS FINALES")
    print(f"{'='*80}")
    print(f"Reward promedio: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Length promedio: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Badges promedio: {np.mean(badges_collected):.2f} ± {np.std(badges_collected):.2f}")
    print(f"Mapas promedio: {np.mean(maps_explored):.2f} ± {np.std(maps_explored):.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar agente entrenado")
    
    parser.add_argument("model", type=str, help="Ruta al modelo (.zip)")
    parser.add_argument("--rom", type=str, default="POKEMON_RED.gb", help="Ruta al ROM")
    parser.add_argument("--episodes", type=int, default=5, help="Número de episodios")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay entre frames")
    parser.add_argument("--stochastic", action="store_true", help="Usar política estocástica")
    parser.add_argument("--eval-only", action="store_true", help="Solo métricas, sin visualización")
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate(args.model, args.rom, args.episodes)
    else:
        play(
            args.model, 
            args.rom, 
            args.episodes, 
            delay=args.delay,
            deterministic=not args.stochastic
        )
