#Imports
import time
import gymnasium as gym 
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# Video de apoyo https://www.youtube.com/watch?v=Vrro7W7iW2w
#Enviroment
environment=gym.make("FrozenLake-v1",render_mode='human',is_slippery=True,desc=generate_random_map(size=4))
environment.reset()
environment.render()

iteraciones = 1000
print('Iniciando entrenamiento del programa')
for i in range(iteraciones):
    #0 - iaquierda, 1 - abajo, 2 - derecha, 3 - arriba
    movimiento = environment.action_space.sample()
    resultado = environment.step(movimiento)
    
    #Done simboliza otro elemento que no sea un hielo para caminar y reward determina si se llego a la meta o no 
    siguiente_estado, reward, done, info = resultado[:4] 
    environment.render()
    print('Iteracion: {}, movimiento {}'.format(i+1, movimiento),'y meta',reward)
    time.sleep(2)
    
    # Si el muñeco cae en un hielo lo reinicia y si el mismo llega a la meta termina con la ejecucion de este 
    if done is True and reward != 1:
        environment.reset()
        environment.render()
    elif done is True and reward == 1:
        break
    