import os
import gym

class Node(object):
    def __init__(self):
        self.visited_cnt = 0
        self.children = []
# https://zhuanlan.zhihu.com/p/30458774
def run_env(env_name, n_episode=300, m_steps=1000):
    env = gym.make(env_name)
    
    for i in range(n_episode):
        env.reset()
        root = Node()
        for j in range(m_steps):
            # https://gist.github.com/blole/dfebbec182e6b72ec16b66cc7e331110
            # line 83
    
    
if __name__ == "__main__":
    run_env("cartpole")