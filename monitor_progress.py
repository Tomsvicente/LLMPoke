"""
Monitor de progreso en tiempo real para las 4 instancias.
Muestra estadísticas actualizadas cada 5 segundos.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

def clear_screen():
    """Limpia la consola."""
    os.system('cls' if os.name == 'nt' else 'clear')

def parse_monitor_file(filepath):
    """Lee el último episodio del archivo de monitor."""
    try:
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            
            # Última línea con datos
            last_line = lines[-1].strip()
            if not last_line:
                return None
            
            data = json.loads(last_line)
            return data
    except:
        return None

def get_instance_stats(instance_num):
    """Obtiene estadísticas de una instancia."""
    monitor_dir = f"./logs/monitor_visual"
    
    if not os.path.exists(monitor_dir):
        return None
    
    # Buscar archivo de monitor más reciente
    monitor_files = list(Path(monitor_dir).glob("*.monitor.csv"))
    if not monitor_files:
        return None
    
    # Leer el último
    latest_file = max(monitor_files, key=lambda f: f.stat().st_mtime)
    data = parse_monitor_file(latest_file)
    
    if data:
        return {
            'reward': data.get('r', 0),
            'length': data.get('l', 0),
            'time': data.get('t', 0)
        }
    
    return None

def format_time(seconds):
    """Formatea segundos a HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def main():
    """Loop principal del monitor."""
    print("Monitor de Progreso - 4 Instancias de Pokemon RL")
    print("=" * 80)
    print()
    print("Iniciando monitoreo...")
    print("Presiona Ctrl+C para salir")
    print()
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            
            clear_screen()
            
            print("=" * 80)
            print(f"  MONITOR DE PROGRESO - POKEMON RL (4 INSTANCIAS)")
            print("=" * 80)
            print(f"  Tiempo transcurrido: {format_time(elapsed)}")
            print(f"  Actualización #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)
            print()
            
            # Estadísticas globales
            total_reward = 0
            total_episodes = 0
            instances_active = 0
            
            # Mostrar cada instancia
            for i in range(1, 5):
                print(f"  INSTANCIA {i}:")
                print("  " + "-" * 76)
                
                stats = get_instance_stats(i)
                
                if stats:
                    instances_active += 1
                    reward = stats['reward']
                    length = stats['length']
                    ep_time = stats['time']
                    
                    total_reward += reward
                    total_episodes += 1
                    
                    print(f"    Estado: ENTRENANDO")
                    print(f"    Reward episodio: {reward:.2f}")
                    print(f"    Duración episodio: {length} pasos")
                    print(f"    Tiempo episodio: {format_time(ep_time)}")
                else:
                    print(f"    Estado: INICIANDO... (esperando datos)")
                
                print()
            
            # Resumen global
            print("=" * 80)
            print("  RESUMEN GLOBAL:")
            print("  " + "-" * 76)
            print(f"    Instancias activas: {instances_active}/4")
            if total_episodes > 0:
                print(f"    Reward promedio: {total_reward/total_episodes:.2f}")
            print(f"    Tiempo total: {format_time(elapsed)}")
            print("=" * 80)
            print()
            print("  Actualizando cada 5 segundos... (Ctrl+C para salir)")
            
            # Esperar 5 segundos
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitor detenido.")
        print(f"Tiempo total de monitoreo: {format_time(elapsed)}")

if __name__ == "__main__":
    main()
