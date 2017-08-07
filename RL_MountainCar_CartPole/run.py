import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
DISPLAY_REWARD_THRESHOLD = -2000
# env = gym.make('CartPole-v0')
# DISPLAY_REWARD_THRESHOLD = 400

env.seed(1)
env = env.unwrapped

RENDER = False

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

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:

            rewards_sum = sum(RL.experience_rewards)

            running_reward = running_reward*0.995 + rewards_sum*0.005 if 'running_reward' in globals() else rewards_sum

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            # if i_episode == 30:
            #     plt.plot(vt)
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()

            break

        observation = observation_

print 'Model saved in ', RL.saver.save(RL.sess, '/tmp/train/model.ckpt')
