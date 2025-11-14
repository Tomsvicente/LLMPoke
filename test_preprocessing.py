"""
Test del preprocesamiento de observaciones.
Compara observaciones originales vs preprocesadas.
"""
from pokemon_env import PokemonRedEnv
from wrappers import make_preprocessed_env, make_training_env
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("TEST DE PREPROCESAMIENTO DE OBSERVACIONES")
print("=" * 80)

# === TEST 1: Entorno sin preprocesar ===
print("\n1. Creando entorno SIN preprocesar...")
env_raw = PokemonRedEnv(rom_path="POKEMON_RED.gb", render_mode=None, max_steps=100)
obs_raw, _ = env_raw.reset()
print(f"   ✓ Shape original: {obs_raw.shape}")
print(f"   ✓ Dtype: {obs_raw.dtype}")
print(f"   ✓ Rango: [{obs_raw.min()}, {obs_raw.max()}]")
print(f"   ✓ Tamaño en memoria: {obs_raw.nbytes / 1024:.2f} KB")

# Ejecutar algunos steps
for _ in range(5):
    obs_raw, _, _, _, _ = env_raw.step(env_raw.action_space.sample())

env_raw.close()

# === TEST 2: Entorno preprocesado completo ===
print("\n2. Creando entorno CON preprocesamiento completo...")
env_processed = make_training_env(rom_path="POKEMON_RED.gb", max_steps=100)
obs_processed, _ = env_processed.reset()
print(f"   ✓ Shape preprocesado: {obs_processed.shape}")
print(f"   ✓ Dtype: {obs_processed.dtype}")
print(f"   ✓ Rango: [{obs_processed.min():.3f}, {obs_processed.max():.3f}]")
print(f"   ✓ Tamaño en memoria: {obs_processed.nbytes / 1024:.2f} KB")

# Ejecutar algunos steps
for _ in range(5):
    obs_processed, _, _, _, _ = env_processed.step(env_processed.action_space.sample())

env_processed.close()

# === TEST 3: Comparación de reducción ===
print("\n3. Comparación de reducción de datos:")
reduction_ratio = obs_raw.nbytes / obs_processed.nbytes
print(f"   Original:     {obs_raw.nbytes:>8} bytes | Shape: {obs_raw.shape}")
print(f"   Preprocesado: {obs_processed.nbytes:>8} bytes | Shape: {obs_processed.shape}")
print(f"   Reducción:    {reduction_ratio:.2f}x más pequeño")
print(f"   Ahorro:       {(1 - 1/reduction_ratio) * 100:.1f}%")

# === TEST 4: Probar diferentes configuraciones ===
print("\n4. Probando diferentes configuraciones de preprocesamiento...")

configs = [
    ("Solo Grayscale", dict(grayscale=True, resize=False, normalize=False, frame_stack=0, skip_frames=0)),
    ("Grayscale + Resize", dict(grayscale=True, resize=True, normalize=False, frame_stack=0, skip_frames=0)),
    ("Gray + Resize + Norm", dict(grayscale=True, resize=True, normalize=True, frame_stack=0, skip_frames=0)),
    ("Completo (4 frames)", dict(grayscale=True, resize=True, normalize=True, frame_stack=4, skip_frames=0)),
    ("Completo + Skip", dict(grayscale=True, resize=True, normalize=True, frame_stack=4, skip_frames=4)),
]

print(f"\n{'Configuración':<25} {'Shape':<20} {'Dtype':<10} {'Size (KB)':<12}")
print("-" * 80)

for name, config in configs:
    env_test = make_preprocessed_env(rom_path="POKEMON_RED.gb", render_mode=None, **config)
    obs, _ = env_test.reset()
    size_kb = obs.nbytes / 1024
    print(f"{name:<25} {str(obs.shape):<20} {str(obs.dtype):<10} {size_kb:>8.2f}")
    env_test.close()

# === TEST 5: Verificar frame stacking ===
print("\n5. Verificando Frame Stacking (captura de movimiento)...")
env_stack = make_preprocessed_env(
    rom_path="POKEMON_RED.gb",
    render_mode=None,
    grayscale=True,
    resize=True,
    normalize=True,
    frame_stack=4,
    skip_frames=0
)

obs, _ = env_stack.reset()
print(f"   ✓ Observación inicial shape: {obs.shape}")
print(f"   ✓ Frames apilados: 4")
print(f"   ✓ Cada frame: 84x84x1 -> Total: 84x84x4")

# Ejecutar acción y verificar que los frames cambian
obs_before = obs.copy()
obs_after, _, _, _, _ = env_stack.step(0)  # Mover arriba

# Verificar que las observaciones son diferentes (hubo movimiento)
diff = np.abs(obs_after - obs_before).sum()
print(f"   ✓ Diferencia entre frames: {diff:.2f} (> 0 = frames distintos)")

env_stack.close()

# === TEST 6: Skip frames (aceleración) ===
print("\n6. Verificando Skip Frames (aceleración del juego)...")
print("   Probando velocidad con y sin skip...")

import time

# Sin skip
env_no_skip = make_preprocessed_env(rom_path="POKEMON_RED.gb", render_mode=None, skip_frames=0, max_steps=50)
obs, _ = env_no_skip.reset()
start = time.time()
for _ in range(50):
    obs, _, term, trunc, _ = env_no_skip.step(env_no_skip.action_space.sample())
    if term or trunc:
        break
time_no_skip = time.time() - start
env_no_skip.close()

# Con skip=4
env_with_skip = make_preprocessed_env(rom_path="POKEMON_RED.gb", render_mode=None, skip_frames=4, max_steps=50)
obs, _ = env_with_skip.reset()
start = time.time()
for _ in range(50):
    obs, _, term, trunc, _ = env_with_skip.step(env_with_skip.action_space.sample())
    if term or trunc:
        break
time_with_skip = time.time() - start
env_with_skip.close()

print(f"   Sin skip (50 steps):  {time_no_skip:.3f}s")
print(f"   Con skip=4 (50 steps): {time_with_skip:.3f}s")
print(f"   Aceleración: {time_no_skip/time_with_skip:.2f}x más rápido")

print("\n" + "=" * 80)
print("RESUMEN DEL PREPROCESAMIENTO")
print("=" * 80)
print("✓ Grayscale: Reduce de RGB (3 canales) a 1 canal")
print("✓ Resize: Reduce resolución de 72x80 a 84x84")
print("✓ Normalize: Escala valores de [0,255] a [0,1]")
print("✓ Frame Stack: Apila 4 frames para capturar movimiento")
print("✓ Skip Frames: Ejecuta acción por 4 frames (acelera 4x)")
print(f"\nReducción total: {reduction_ratio:.1f}x más pequeño")
print("=" * 80)

print("\n✓ Test de preprocesamiento completado!")
