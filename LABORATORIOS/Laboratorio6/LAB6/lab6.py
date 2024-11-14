import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

def discretizar(observation):
    resized = cv2.resize(observation, (5, 5))
    discretized = (resized / 255 * 5).astype(np.int32)
    return tuple(discretized.flatten())

def train(episodes):
    # Parámetros de Q-learning
    learning_rate = 0.3
    discount_factor = 0.95
    epsilon = 1
    epsilon_decay_rate = 0.01
    rng = np.random.default_rng()
    
    # Inicializar arrays y Q-table
    rewards_per_episode = np.zeros(episodes)
    q_table = defaultdict(lambda: np.zeros(18))  # 18 acciones en Boxing
    
    # Entrenar primero sin renderización
    env_train = gym.make("ALE/Boxing-v5", render_mode=None, obs_type="grayscale")
    
    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Decidir qué entorno usar basado en el episodio
        if i >= episodes - 2:  # Últimos 5 episodios
            if 'env_render' not in locals():  # Si aún no existe el entorno de renderización
                env_train.close()  # Cerrar el entorno de entrenamiento
                env_render = gym.make("ALE/Boxing-v5", render_mode="human", obs_type="grayscale")
                env = env_render
            else:
                env = env_render
        else:
            env = env_train
        
        state = env.reset()[0]
        state = discretizar(state)
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            # Decisión de acción: explorar o explotar
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            # Realizar acción
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = discretizar(new_state)
            
            # Actualizar la tabla Q
            old_value = q_table[state][action]
            next_max = np.max(q_table[new_state])
            
            # Fórmula Q-learning
            new_value = old_value + learning_rate * (
                reward + discount_factor * next_max - old_value
            )
            q_table[state][action] = new_value
            
            state = new_state
            total_reward += reward
        
        # Registrar recompensa total del episodio
        rewards_per_episode[i] = total_reward
        
        # Reducir epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # Imprimir progreso
        if (i + 1) % 250 == 0 or i >= episodes - 5:
            print(f"Episodio: {i + 1} - Recompensa total: {total_reward}")
    
    # Cerrar los entornos
    env_train.close()
    if 'env_render' in locals():
        env_render.close()
    
    # Mostrar algunos valores de la Q-table
    print("\nMuestra de la Q-table:")
    for idx, (state, actions) in enumerate(list(q_table.items())[:10]):
        print(f"Estado: {state[:10]}..., Acciones: {actions}")
    
    # Graficar rendimiento
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas en 100 episodios')
    plt.title('Rendimiento acumulado en el entorno de Boxing')
    plt.show()

# Ejecutar el entrenamiento
if __name__ == "__main__":
    train(100)