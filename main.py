from pyboy import PyBoy
import random
import time

# Ruta al ROM de Pokémon Rojo
ROM_PATH = "POKEMON_RED.gb"  # Asegúrate que esté en el mismo folder

# Crear instancia del emulador
print("Inicializando PyBoy...")
pyboy = PyBoy(ROM_PATH, window="SDL2")  # usa SDL2 para ver ventana
print("PyBoy inicializado correctamente!")
pyboy.set_emulation_speed(0)  # 1 = velocidad normal, 0 = máxima velocidad

# Acciones posibles (botones)
BUTTONS = [
    "up",
    "down",
    "left",
    "right",
    "a",
    "b",
    "start",
    "select",
]

# Avanzar unos frames iniciales
print("Iniciando frames iniciales...")
for _ in range(10000):  # Frames suficientes para que el juego arranque bien
    pyboy.tick()

print("Entrando al loop principal...")
print("Presiona Ctrl+C para detener el juego")

# Contador para limitar el tiempo de ejecución
step_count = 0
max_steps = 1000  # El juego correrá por 1000 acciones (~3-5 minutos)

# Loop principal
try:
    while step_count < max_steps:
        # Elegir acción random
        action = random.choice(BUTTONS)
        
        # Usar la forma correcta de obtener botones (de la clase PyBoy, no del objeto)
        boton = getattr(PyBoy, f"BUTTON_{action.upper()}")

        # Mostrar progreso cada 50 acciones
        if step_count % 50 == 0:
            print(f"Paso {step_count}/{max_steps} - Acción: {action}")

        # Presionar botón
        pyboy.send_input(boton)

        # Avanzar frames mientras está presionado
        for _ in range(10):
            pyboy.tick()
            
        # Soltar botón
        pyboy.send_input(boton, press=False)

        # Avanzar algunos frames más después de soltar
        for _ in range(5):
            pyboy.tick()

        step_count += 1
        time.sleep(0.2)  # Delay para ver mejor las acciones

except KeyboardInterrupt:
    print("Interrumpido por el usuario...")
finally:
    print(f"Juego terminado después de {step_count} acciones")
    pyboy.stop()
    print("Emulador cerrado correctamente")
