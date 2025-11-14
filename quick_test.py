"""
Test rápido para verificar que el entorno funciona.
"""
from pokemon_env import PokemonRedEnv

print("Probando PokemonRedEnv...")

# Crear entorno sin ventana (headless)
env = PokemonRedEnv(rom_path="POKEMON_RED.gb", render_mode=None, max_steps=50)
print("✓ Entorno creado!")

# Reset
obs, info = env.reset()
print(f"✓ Reset exitoso - Observación shape: {obs.shape}")
print(f"  Estado inicial: Pos({info['player_x']}, {info['player_y']}), Mapa: {info['map_id']}")

# Ejecutar algunos steps
print("\nEjecutando 10 acciones aleatorias...")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {i+1}: Acción={info['action_taken']}, Reward={reward:.4f}, Terminated={terminated}")
    
    if terminated or truncated:
        break

print("\n✓ Test completado exitosamente!")
env.close()
