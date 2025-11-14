"""
Test visual para ver el agente jugando en tiempo real.
"""
from pokemon_env import PokemonRedEnv
import time

print("=" * 70)
print("TEST VISUAL - VER AL AGENTE JUGANDO")
print("=" * 70)
print("\nPresiona Ctrl+C para detener...\n")

# Crear entorno CON ventana para ver
env = PokemonRedEnv(rom_path="POKEMON_RED.gb", render_mode="human", max_steps=5000)
print("✓ Entorno creado con ventana SDL2!")

try:
    # Reset
    observation, info = env.reset()
    
    print("\n" + "=" * 70)
    print("ESTADO INICIAL")
    print("=" * 70)
    print(f"Posición: ({info['player_x']}, {info['player_y']})")
    print(f"Mapa ID: {info['map_id']}")
    print(f"Badges: {info['badges']}/8")
    print(f"Party size: {info['party_size']}")
    print(f"HP: {info['hp_current']}/{info['hp_max']}")
    print(f"Level: {info['pokemon_level']}")
    print(f"Money: ${info['money']}")
    print("=" * 70)
    
    print("\n🎮 Jugando con acciones aleatorias...")
    print("(Observa la ventana del juego)\n")
    
    steps = 0
    total_reward = 0
    
    while steps < 500:  # 500 acciones
        # Acción aleatoria
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        steps += 1
        total_reward += reward
        
        # Mostrar progreso cada 50 steps
        if steps % 50 == 0:
            print(f"\n[Step {steps}/500]")
            print(f"  Posición: ({info['player_x']}, {info['player_y']}) | Mapa: {info['map_id']}")
            print(f"  HP: {info['hp_current']}/{info['hp_max']} | Level: {info['pokemon_level']}")
            print(f"  Mapas visitados: {len(env.visited_maps)}")
            print(f"  Reward acumulado: {total_reward:.4f}")
        
        # Pequeña pausa para que sea visible
        time.sleep(0.05)
        
        if terminated:
            print(f"\n💀 GAME OVER en step {steps}!")
            break
        
        if truncated:
            print(f"\n⏱️ Límite de steps alcanzado!")
            break

except KeyboardInterrupt:
    print("\n\n⚠️ Detenido por el usuario")

finally:
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    stats = env.get_progress_stats()
    print(f"Steps totales: {steps}")
    print(f"Mapas explorados: {stats['maps_visited']}")
    print(f"Badges: {stats['badges']}/8")
    print(f"Level máximo: {stats['level']}")
    print(f"Money: ${stats['money']}")
    print(f"Reward total: {total_reward:.4f}")
    print("=" * 70)
    
    env.close()
    print("\n✓ Test visual completado!")
