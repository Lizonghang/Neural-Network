import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000

RENDER = False

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)

for i_episode in range(1000):

    observation = env.reset()

    while True:

        if RENDER:
            env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)  # reward = -1

        RL.store_transition(observation, action, reward)

        if done:

            rewards_sum = sum(RL.experience_rewards)

            running_reward = running_reward*0.99 + rewards_sum*0.01 if 'running_reward' in globals() else rewards_sum

            # if running_reward > DISPLAY_REWARD_THRESHOLD:
            #     RENDER = True

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            # if i_episode == 30:
            #     plt.plot(vt)
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()

            break

        observation = observation_
