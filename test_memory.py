"""
Script para verificar la lectura de memoria RAM del juego.
Muestra información en tiempo real del estado del juego.
"""
from pokemon_env import PokemonRedEnv
import time


def test_memory_reading():
    """
    Probar la lectura de memoria mostrando el estado del juego en tiempo real.
    """
    print("=" * 70)
    print("TEST DE LECTURA DE MEMORIA - POKÉMON RED")
    print("=" * 70)
    
    # Crear entorno
    print("\nCreando entorno...")
    env = PokemonRedEnv(rom_path="POKEMON_RED.gb", render_mode="human", max_steps=1000)
    print("✓ Entorno creado!\n")
    
    # Reset
    observation, info = env.reset()
    
    print("Estado inicial del juego:")
    print("-" * 70)
    print(f"Posición: ({info['player_x']}, {info['player_y']})")
    print(f"Mapa ID: {info['map_id']}")
    print(f"Badges: {info['badges']}/8")
    print(f"Pokémon en party: {info['party_size']}")
    print(f"HP: {info['hp_current']}/{info['hp_max']} ({info['hp_ratio']*100:.1f}%)")
    print(f"Nivel: {info['pokemon_level']}")
    print(f"Dinero: ${info['money']}")
    print("-" * 70)
    
    print("\nPresiona Ctrl+C para detener...")
    print("\nEjecutando acciones aleatorias y monitoreando cambios...\n")
    
    steps = 0
    last_state = info.copy()
    
    try:
        while steps < 500:
            # Acción aleatoria
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Detectar cambios importantes en el estado
            changes = []
            
            if info['player_x'] != last_state['player_x'] or info['player_y'] != last_state['player_y']:
                changes.append(f"Movimiento: ({last_state['player_x']},{last_state['player_y']}) → ({info['player_x']},{info['player_y']})")
            
            if info['map_id'] != last_state['map_id']:
                changes.append(f"⭐ NUEVO MAPA: {last_state['map_id']} → {info['map_id']}")
            
            if info['badges'] != last_state['badges']:
                changes.append(f"🏆 NUEVA BADGE! {last_state['badges']} → {info['badges']}")
            
            if info['pokemon_level'] != last_state['pokemon_level']:
                changes.append(f"📈 LEVEL UP! {last_state['pokemon_level']} → {info['pokemon_level']}")
            
            if info['hp_current'] != last_state['hp_current']:
                hp_diff = info['hp_current'] - last_state['hp_current']
                emoji = "❤️" if hp_diff > 0 else "💔"
                changes.append(f"{emoji} HP: {last_state['hp_current']} → {info['hp_current']} ({hp_diff:+d})")
            
            if info['party_size'] != last_state['party_size']:
                changes.append(f"⚡ Party size: {last_state['party_size']} → {info['party_size']}")
            
            if info['money'] != last_state['money']:
                money_diff = info['money'] - last_state['money']
                changes.append(f"💰 Dinero: ${last_state['money']} → ${info['money']} ({money_diff:+d})")
            
            # Mostrar cambios detectados
            if changes:
                print(f"\n[Step {steps}] Acción: {info['action_taken']}")
                for change in changes:
                    print(f"  {change}")
            
            # Mostrar progreso cada 50 steps
            if steps % 50 == 0:
                stats = env.get_progress_stats()
                print(f"\n{'='*70}")
                print(f"PROGRESO - Step {steps}/500")
                print(f"{'='*70}")
                print(f"Badges: {stats['badges']}/8 | Level: {stats['level']} | Mapas visitados: {stats['maps_visited']}")
                print(f"HP: {info['hp_current']}/{info['hp_max']} ({stats['hp_ratio']*100:.1f}%)")
                print(f"Dinero: ${stats['money']} | Party: {stats['party_size']} Pokémon")
                print(f"{'='*70}\n")
            
            last_state = info.copy()
            
            if terminated:
                print("\n💀 GAME OVER - Todos los Pokémon debilitados!")
                break
            
            if truncated:
                print("\n⏱️ Límite de steps alcanzado!")
                break
            
            time.sleep(0.05)  # Pequeña pausa para ver mejor
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Prueba interrumpida por el usuario")
    
    finally:
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        stats = env.get_progress_stats()
        print(f"Steps totales: {steps}")
        print(f"Badges obtenidas: {stats['badges']}/8")
        print(f"Nivel máximo: {stats['level']}")
        print(f"Mapas explorados: {stats['maps_visited']}")
        print(f"Dinero final: ${stats['money']}")
        print(f"Reward acumulado: {info['episode_reward']:.4f}")
        print("=" * 70)
        
        env.close()
        print("\n✓ Test completado!")


if __name__ == "__main__":
    test_memory_reading()
