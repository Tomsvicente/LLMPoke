"""
Test del sistema de recompensas.
Muestra en detalle cómo se calculan las recompensas por diferentes acciones.
"""
from pokemon_env import PokemonRedEnv
import time

print("=" * 80)
print("TEST DEL SISTEMA DE RECOMPENSAS")
print("=" * 80)

# Crear entorno con ventana
env = PokemonRedEnv(rom_path="POKEMON_RED.gb", render_mode="human", max_steps=5000)
print("✓ Entorno creado!\n")

# Reset
observation, info = env.reset()

print("=" * 80)
print("ESTADO INICIAL")
print("=" * 80)
print(f"Posición: ({info['player_x']}, {info['player_y']}) | Mapa: {info['map_id']}")
print(f"Badges: {info['badges']}/8 | Party: {info['party_size']} Pokémon")
print(f"HP: {info['hp_current']}/{info['hp_max']} | Level: {info['pokemon_level']}")
print(f"Money: ${info['money']}")
print("=" * 80)

print("\n🎮 Observa cómo cambian las recompensas según las acciones...")
print("Presiona Ctrl+C para detener\n")

steps = 0
total_reward = 0
rewards_breakdown = {
    'exploration': 0,
    'badges': 0,
    'level': 0,
    'money': 0,
    'party': 0,
    'hp_loss': 0,
    'hp_gain': 0,
    'movement': 0,
    'survival': 0,
}

last_state = info.copy()

try:
    while steps < 1000:
        # Acción aleatoria
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        steps += 1
        total_reward += reward
        
        # Analizar qué causó la recompensa
        reward_reasons = []
        
        # Exploración
        if info['map_id'] != last_state['map_id']:
            reward_reasons.append(f"🗺️  NUEVO MAPA ({last_state['map_id']} → {info['map_id']})")
            rewards_breakdown['exploration'] += 1.0
            
        # Badges
        if info['badges'] > last_state['badges']:
            badge_gain = info['badges'] - last_state['badges']
            reward_reasons.append(f"🏆 BADGE! ({last_state['badges']} → {info['badges']})")
            rewards_breakdown['badges'] += 5.0 * badge_gain
            
        # Level up
        if info['pokemon_level'] > last_state['pokemon_level']:
            level_gain = info['pokemon_level'] - last_state['pokemon_level']
            reward_reasons.append(f"📈 LEVEL UP! ({last_state['pokemon_level']} → {info['pokemon_level']})")
            rewards_breakdown['level'] += 2.0 * level_gain
            
        # Dinero
        if info['money'] > last_state['money']:
            money_gain = info['money'] - last_state['money']
            reward_reasons.append(f"💰 +${money_gain}")
            rewards_breakdown['money'] += money_gain * 0.0001
            
        # Party
        if info['party_size'] > last_state['party_size']:
            party_gain = info['party_size'] - last_state['party_size']
            reward_reasons.append(f"⚡ Nuevo Pokémon! ({last_state['party_size']} → {info['party_size']})")
            rewards_breakdown['party'] += 3.0 * party_gain
            
        # HP
        if info['hp_max'] > 0 and last_state['hp_max'] > 0:
            hp_change = info['hp_current'] - last_state['hp_current']
            if hp_change < 0:
                reward_reasons.append(f"💔 Perdió HP ({last_state['hp_current']} → {info['hp_current']})")
                rewards_breakdown['hp_loss'] += -0.5 * abs(hp_change) / info['hp_max']
            elif hp_change > 0:
                reward_reasons.append(f"❤️  Recuperó HP ({last_state['hp_current']} → {info['hp_current']})")
                rewards_breakdown['hp_gain'] += 0.1 * hp_change / info['hp_max']
        
        # Movimiento
        if (info['player_x'] != last_state['player_x'] or 
            info['player_y'] != last_state['player_y']):
            rewards_breakdown['movement'] += 0.005
        else:
            rewards_breakdown['movement'] -= 0.01
            
        rewards_breakdown['survival'] += 0.001
        
        # Mostrar eventos importantes
        if reward_reasons:
            print(f"\n{'='*80}")
            print(f"[Step {steps}] Acción: {info['action_taken']} | Reward: {reward:.4f}")
            print(f"{'='*80}")
            for reason in reward_reasons:
                print(f"  {reason}")
            print(f"  📍 Posición: ({info['player_x']}, {info['player_y']}) | Mapa: {info['map_id']}")
            print(f"  💰 Money: ${info['money']} | HP: {info['hp_current']}/{info['hp_max']}")
            
        # Resumen cada 100 steps
        if steps % 100 == 0:
            print(f"\n{'='*80}")
            print(f"RESUMEN - Step {steps}/1000")
            print(f"{'='*80}")
            stats = env.get_progress_stats()
            print(f"Reward Total: {total_reward:.4f}")
            print(f"Mapas visitados: {stats['maps_visited']}")
            print(f"Badges: {stats['badges']}/8 | Level: {stats['level']}")
            print(f"Money: ${stats['money']} | Party: {stats['party_size']}")
            print(f"\nDesglose de Rewards:")
            print(f"  Exploración:  {rewards_breakdown['exploration']:>8.4f}")
            print(f"  Badges:       {rewards_breakdown['badges']:>8.4f}")
            print(f"  Level Up:     {rewards_breakdown['level']:>8.4f}")
            print(f"  Dinero:       {rewards_breakdown['money']:>8.4f}")
            print(f"  Party:        {rewards_breakdown['party']:>8.4f}")
            print(f"  HP Perdido:   {rewards_breakdown['hp_loss']:>8.4f}")
            print(f"  HP Ganado:    {rewards_breakdown['hp_gain']:>8.4f}")
            print(f"  Movimiento:   {rewards_breakdown['movement']:>8.4f}")
            print(f"  Supervivencia:{rewards_breakdown['survival']:>8.4f}")
            print(f"{'='*80}\n")
        
        last_state = info.copy()
        time.sleep(0.05)  # Pausa para ver mejor
        
        if terminated or truncated:
            break

except KeyboardInterrupt:
    print("\n\n⚠️ Test detenido por el usuario")

finally:
    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")
    stats = env.get_progress_stats()
    print(f"Steps totales: {steps}")
    print(f"Reward total: {total_reward:.4f}")
    print(f"Reward promedio por step: {total_reward/steps if steps > 0 else 0:.4f}")
    print(f"\nProgreso del juego:")
    print(f"  Mapas explorados: {stats['maps_visited']}")
    print(f"  Badges: {stats['badges']}/8")
    print(f"  Level máximo: {stats['level']}")
    print(f"  Money: ${stats['money']}")
    print(f"  Party size: {stats['party_size']}")
    print(f"\nDesglose de Rewards:")
    for key, value in rewards_breakdown.items():
        print(f"  {key.replace('_', ' ').title():.<20} {value:>10.4f} ({value/total_reward*100 if total_reward > 0 else 0:>5.1f}%)")
    print(f"{'='*80}")
    
    env.close()
    print("\n✓ Test de recompensas completado!")
