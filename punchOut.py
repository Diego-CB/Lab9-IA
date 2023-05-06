# ANTES DE CORRER
# pip install gymnasium[atari]
# pip install gymnasium[accept-rom-license]

import gymnasium as gym

env = gym.make("ALE/Boxing-v5", render_mode='human')
observation, info = env.reset()

score = 0
goal = 10
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if reward > 0:
        print(f'El agente acertó un golpe de {reward} puntos')
        score += reward

    if terminated or truncated:
        observation, info = env.reset()

    if score > 9:
        print(f'Tu jugador ha acertado {goal} golpes')
        break
env.close()
