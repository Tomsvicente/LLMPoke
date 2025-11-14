"""
Test simple: crear 4 instancias de PyBoy con ventanas visibles.
"""

from pokemon_env import PokemonRedEnv
import time

print("🎮 Creando 4 instancias de PyBoy...")

envs = []

for i in range(4):
    print(f"  Creando instancia {i+1}...")
    env = PokemonRedEnv(rom_path="POKEMON_RED.GB", render_mode="human")
    envs.append(env)
    time.sleep(0.5)

print("✅ ¡4 ventanas creadas!")
print("💡 Deberías ver 4 ventanas de Game Boy abiertas")
print()
print("Presiona Ctrl+C para cerrar...")

try:
    # Mantener vivas las ventanas
    while True:
        for i, env in enumerate(envs):
            # Ejecutar una acción aleatoria en cada env
            action = env.action_space.sample()
            env.step(action)
        time.sleep(0.016)  # ~60 FPS
except KeyboardInterrupt:
    print("\n🛑 Cerrando...")
    for env in envs:
        env.close()
    print("✅ Cerrado")
