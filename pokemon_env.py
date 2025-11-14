"""
Entorno de Pokémon Red compatible con Gymnasium para Reinforcement Learning.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import time


class PokemonRedEnv(gym.Env):
    """
    Environment personalizado para Pokémon Red usando PyBoy.
    Compatible con Gymnasium API para usar con Stable-Baselines3.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, rom_path="POKEMON_RED.gb", render_mode=None, max_steps=10000):
        super().__init__()
        
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Inicializar PyBoy
        window_type = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=window_type)
        self.pyboy.set_emulation_speed(0)  # Velocidad máxima
        
        # Definir espacio de acciones (8 botones)
        # 0: up, 1: down, 2: left, 3: right, 4: a, 5: b, 6: start, 7: select
        self.action_space = spaces.Discrete(8)
        
        # Mapeo de acciones a nombres de botones (string)
        self.action_to_button = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "a",
            5: "b",
            6: "start",
            7: "select",
        }
        
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
        
        # Definir espacio de observaciones
        # Usaremos la pantalla del Game Boy (160x144 pixels, RGB)
        # La reduciremos a 80x72 para hacerla más manejable
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(72, 80, 3),  # Height, Width, Channels (RGB)
            dtype=np.uint8
        )
        
        # Variables para tracking del progreso
        self.previous_state = {}
        self.episode_reward = 0
        
        # Estadísticas de progreso del episodio
        self.max_map_progress = 0
        self.max_level_reached = 0
        self.visited_maps = set()
        
        # Inicializar el juego (skip intro)
        self._skip_intro()
    
    def _skip_intro(self, frames=10000):
        """
        Avanzar frames iniciales y presionar botones para saltar la intro del juego.
        Presiona START y A varias veces para pasar los menús iniciales.
        """
        print("Saltando intro del juego...")
        
        # Avanzar frames iniciales
        for _ in range(1000):
            self.pyboy.tick()
        
        # Presionar START y A varias veces para pasar menús
        for _ in range(20):
            # Presionar START
            self.pyboy.button_press("start")
            for _ in range(10):
                self.pyboy.tick()
            self.pyboy.button_release("start")
            
            for _ in range(20):
                self.pyboy.tick()
            
            # Presionar A
            self.pyboy.button_press("a")
            for _ in range(10):
                self.pyboy.tick()
            self.pyboy.button_release("a")
            
            for _ in range(30):
                self.pyboy.tick()
        
        # Avanzar más frames para estabilizar
        for _ in range(500):
            self.pyboy.tick()
            
        print("Intro completada!")
    
    def _get_screen(self):
        """
        Obtener la pantalla actual del juego.
        Retorna imagen reducida de 80x72 en RGB.
        """
        # Obtener screen array de PyBoy
        screen = np.array(self.pyboy.screen.ndarray)
        
        # Asegurarnos que tiene 3 canales RGB
        if len(screen.shape) == 2:  # Si es escala de grises
            screen = np.stack([screen] * 3, axis=-1)
        elif screen.shape[-1] == 4:  # Si tiene canal alpha (RGBA)
            screen = screen[:, :, :3]  # Tomar solo RGB
        
        # Reducir resolución a la mitad para hacerla más manejable
        # De 160x144 a 80x72
        reduced_screen = screen[::2, ::2, :]
        
        return reduced_screen.astype(np.uint8)
    
    def _execute_action(self, action):
        """
        Ejecutar una acción en el emulador.
        Presiona el botón por varios frames y luego lo suelta.
        """
        # Convertir a int si es numpy array (por compatibilidad con SB3)
        if hasattr(action, 'item'):
            action = action.item()
        
        button_name = self.action_to_button[action]
        
        # Presionar botón usando button_press
        self.pyboy.button_press(button_name)
        
        # Mantener presionado por 10 frames
        for _ in range(10):
            self.pyboy.tick()
        
        # Soltar botón usando button_release
        self.pyboy.button_release(button_name)
        
        # Avanzar algunos frames adicionales
        for _ in range(5):
            self.pyboy.tick()
    
    def _read_memory(self, address):
        """Leer un byte de memoria en la dirección especificada."""
        return self.pyboy.memory[address]
    
    def _read_bit(self, address, bit_position):
        """Leer un bit específico de un byte en memoria."""
        byte_value = self._read_memory(address)
        return (byte_value >> bit_position) & 1
    
    def _get_game_state(self):
        """
        Extraer información del estado actual del juego desde la memoria RAM.
        
        Direcciones de memoria para Pokémon Red (USA):
        Fuente: https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        """
        
        try:
            # Posición del jugador
            player_x = self._read_memory(0xD362)  # X position
            player_y = self._read_memory(0xD361)  # Y position  
            map_id = self._read_memory(0xD35E)     # Map ID
            
            # Badges (8 gimnasios)
            badges_byte = self._read_memory(0xD356)
            badges_count = bin(badges_byte).count('1')
            
            # Party info
            party_size = self._read_memory(0xD163)
            
            # HP del primer Pokémon en el party
            # HP actual (2 bytes, big endian)
            hp_current_high = self._read_memory(0xD16C)
            hp_current_low = self._read_memory(0xD16D)
            hp_current = (hp_current_high << 8) | hp_current_low
            
            # HP máximo (2 bytes, big endian)
            hp_max_high = self._read_memory(0xD16E)
            hp_max_low = self._read_memory(0xD16F)
            hp_max = (hp_max_high << 8) | hp_max_low
            
            # Nivel del primer Pokémon
            pokemon_level = self._read_memory(0xD18C)
            
            # Dinero (3 bytes en formato BCD)
            money_byte1 = self._read_memory(0xD347)
            money_byte2 = self._read_memory(0xD348)
            money_byte3 = self._read_memory(0xD349)
            
            # Convertir BCD a decimal
            money = (
                ((money_byte1 >> 4) * 10 + (money_byte1 & 0x0F)) * 10000 +
                ((money_byte2 >> 4) * 10 + (money_byte2 & 0x0F)) * 100 +
                ((money_byte3 >> 4) * 10 + (money_byte3 & 0x0F))
            )
            
            return {
                'step': self.current_step,
                'player_x': player_x,
                'player_y': player_y,
                'map_id': map_id,
                'badges': badges_count,
                'party_size': party_size,
                'hp_current': hp_current,
                'hp_max': hp_max,
                'hp_ratio': hp_current / hp_max if hp_max > 0 else 0,
                'pokemon_level': pokemon_level,
                'money': money,
            }
        except Exception as e:
            # Si hay error leyendo memoria, retornar estado seguro
            return {
                'step': self.current_step,
                'player_x': 0,
                'player_y': 0,
                'map_id': 0,
                'badges': 0,
                'party_size': 0,
                'hp_current': 1,  # Evitar división por cero
                'hp_max': 1,
                'hp_ratio': 1.0,
                'pokemon_level': 1,
                'money': 0,
            }
    
    def _calculate_reward(self, current_state, previous_state):
        """
        Calcular la recompensa basada en el progreso del juego.
        
        Sistema de recompensas:
        - Exploración: +1.0 por descubrir nuevo mapa
        - Progreso: +5.0 por obtener nueva badge (gimnasio)
        - Nivel: +2.0 por subir de nivel
        - Dinero: +0.01 por cada $100 ganados
        - Party: +3.0 por capturar nuevo Pokémon
        - HP: -0.5 por perder HP, +0.1 por recuperar HP
        - Estancamiento: -0.01 por quedarse en la misma posición
        - Supervivencia: +0.001 por cada step (incentivo base)
        """
        reward = 0.0
        
        # Recompensa base por sobrevivir
        reward += 0.001
        
        # === EXPLORACIÓN ===
        # Recompensa grande por descubrir nuevo mapa
        if current_state['map_id'] != previous_state.get('map_id', -1):
            if current_state['map_id'] not in self.visited_maps or len(self.visited_maps) == 1:
                reward += 1.0
                
        # === BADGES (Gimnasios) ===
        # Recompensa MUY grande por obtener nueva badge
        badges_gained = current_state['badges'] - previous_state.get('badges', 0)
        if badges_gained > 0:
            reward += 5.0 * badges_gained
            
        # === NIVEL ===
        # Recompensa por subir de nivel
        level_gained = current_state['pokemon_level'] - previous_state.get('pokemon_level', 0)
        if level_gained > 0:
            reward += 2.0 * level_gained
            
        # === DINERO ===
        # Recompensa por ganar dinero (indica batallas ganadas)
        money_gained = current_state['money'] - previous_state.get('money', 0)
        if money_gained > 0:
            reward += money_gained * 0.0001  # $100 = 0.01 reward
            
        # === PARTY SIZE ===
        # Recompensa por capturar nuevos Pokémon
        party_gained = current_state['party_size'] - previous_state.get('party_size', 0)
        if party_gained > 0:
            reward += 3.0 * party_gained
            
        # === HP (SALUD) ===
        # Penalización por perder HP, recompensa por recuperar
        if current_state['hp_max'] > 0 and previous_state.get('hp_max', 0) > 0:
            hp_change = current_state['hp_current'] - previous_state.get('hp_current', 0)
            if hp_change < 0:  # Perdió HP
                reward -= 0.5 * abs(hp_change) / current_state['hp_max']
            elif hp_change > 0:  # Recuperó HP
                reward += 0.1 * hp_change / current_state['hp_max']
                
        # === MOVIMIENTO ===
        # Pequeña penalización por quedarse en la misma posición (anti-estancamiento)
        if (current_state['player_x'] == previous_state.get('player_x', -1) and 
            current_state['player_y'] == previous_state.get('player_y', -1) and
            current_state['map_id'] == previous_state.get('map_id', -1)):
            reward -= 0.01
        else:
            # Pequeña recompensa por moverse
            reward += 0.005
            
        return reward
    
    def get_progress_stats(self):
        """
        Obtener estadísticas del progreso del episodio actual.
        """
        current_state = self._get_game_state()
        return {
            'badges': current_state['badges'],
            'level': current_state['pokemon_level'],
            'maps_visited': len(self.visited_maps),
            'money': current_state['money'],
            'party_size': current_state['party_size'],
            'hp_ratio': current_state['hp_ratio'],
            'steps': self.current_step,
        }
    
    def reset(self, seed=None, options=None):
        """
        Reiniciar el entorno al estado inicial.
        """
        super().reset(seed=seed)
        
        # Cerrar y reiniciar PyBoy
        self.pyboy.stop()
        
        window_type = "SDL2" if self.render_mode == "human" else "null"
        self.pyboy = PyBoy(self.rom_path, window=window_type)
        self.pyboy.set_emulation_speed(0)
        
        # Saltar intro nuevamente
        self._skip_intro()
        
        # Reiniciar contadores
        self.current_step = 0
        self.episode_reward = 0
        self.max_map_progress = 0
        self.max_level_reached = 0
        self.visited_maps = set()
        self.previous_state = self._get_game_state()
        
        # Obtener observación inicial
        observation = self._get_screen()
        info = self._get_game_state()
        
        return observation, info
    
    def step(self, action):
        """
        Ejecutar una acción y retornar el resultado.
        
        Returns:
            observation: Estado actual del juego (pantalla)
            reward: Recompensa obtenida
            terminated: Si el episodio terminó (game over)
            truncated: Si se alcanzó el límite de steps
            info: Información adicional del estado del juego
        """
        # Ejecutar acción
        self._execute_action(action)
        self.current_step += 1
        
        # Obtener nuevo estado
        current_state = self._get_game_state()
        observation = self._get_screen()
        
        # Trackear mapas visitados
        self.visited_maps.add(current_state['map_id'])
        
        # Actualizar máximos
        if current_state['pokemon_level'] > self.max_level_reached:
            self.max_level_reached = current_state['pokemon_level']
        
        # Calcular recompensa
        reward = self._calculate_reward(current_state, self.previous_state)
        self.episode_reward += reward
        
        # Determinar si el episodio terminó
        # Game over si todos los Pokémon están debilitados (HP = 0)
        # Solo verificar si realmente tenemos datos válidos
        terminated = False
        if current_state['hp_max'] > 0:  # Solo verificar si tenemos datos válidos
            terminated = current_state['hp_current'] == 0
        
        truncated = self.current_step >= self.max_steps
        
        # Información adicional
        info = current_state.copy()
        info['episode_reward'] = self.episode_reward
        info['action_taken'] = self.action_names[action]
        
        # Actualizar estado anterior
        self.previous_state = current_state
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Renderizar el entorno.
        En modo 'human', PyBoy ya muestra la ventana.
        """
        if self.render_mode == "rgb_array":
            return self._get_screen()
        elif self.render_mode == "human":
            # PyBoy ya renderiza automáticamente en modo SDL2
            pass
    
    def close(self):
        """
        Cerrar el entorno y liberar recursos.
        """
        if hasattr(self, 'pyboy'):
            self.pyboy.stop()
        print("Entorno cerrado correctamente")
