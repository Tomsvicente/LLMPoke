"""
Test rápido del modelo PPO sin entrenar.
Verifica que la configuración funcione correctamente.
"""
from wrappers import make_training_env
from model import create_model, print_model_info
import torch

print("=" * 80)
print("TEST DE CONFIGURACIÓN DEL MODELO PPO")
print("=" * 80)

# Verificar disponibilidad de CUDA
print(f"\n1. Verificando hardware...")
cuda_available = torch.cuda.is_available()
print(f"   CUDA disponible: {cuda_available}")
if cuda_available:
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Device recomendado: 'cuda'")
else:
    print(f"   Device recomendado: 'cpu'")

# Crear entorno
print(f"\n2. Creando entorno preprocesado...")
env = make_training_env(rom_path="POKEMON_RED.gb", max_steps=100)
print(f"   ✓ Observation space: {env.observation_space.shape}")
print(f"   ✓ Action space: {env.action_space.n} acciones")

# Crear modelo
print(f"\n3. Creando modelo PPO...")
device = 'cuda' if cuda_available else 'cpu'
model = create_model(env, device=device, verbose=0)
print(f"   ✓ Modelo creado")

# Mostrar info
print_model_info(model)

# Test rápido de inferencia
print(f"\n4. Test de inferencia...")
obs, _ = env.reset()
print(f"   Observación inicial shape: {obs.shape}")

# Predecir 10 acciones
print(f"   Prediciendo 10 acciones...")
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(f"   Step {i+1}: Acción={action}, Reward={reward:.4f}")
    
    if done or truncated:
        obs, _ = env.reset()

print(f"\n   ✓ Inferencia exitosa!")

# Cerrar
env.close()

print(f"\n{'='*80}")
print("RESUMEN")
print(f"{'='*80}")
print(f"✓ Entorno funciona correctamente")
print(f"✓ Modelo PPO configurado")
print(f"✓ Inferencia funciona")
print(f"✓ Device: {model.device}")
print(f"\nTodo listo para entrenar!")
print(f"Ejecuta: python train.py --timesteps 100000")
print(f"{'='*80}")
