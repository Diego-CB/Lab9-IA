import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

map = gym.make(
    'FrozenLake-v1',
    desc=generate_random_map(size=4),
    map_name="4x4",
    is_slippery=True,
    render_mode='human'
)

map.reset()
map.render()

'''
Acciones
    - left: 0
    - down: 1
    - right: 2
    - up: 3
'''

'''
step Return
    - observation: object
    - Reward: float
    - Terminated: goal or obstacule
    - Truncated: 
    - info
'''
MAX_ITERATIONS = 30
EPOCHS = 5

for e in range(EPOCHS):
    print('Epoch:', e+1)
    for i in range(MAX_ITERATIONS):
        map.render()
        action = map.action_space.sample()
        observation, reward, terminated, truncated, info = map.step(action)
        print(
            f'Iteration {i+1} action {action}'
        )

        if terminated:
            break


map.close()
