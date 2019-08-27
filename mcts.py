import os
import gym
import numpy as np
class Node(object):
    def __init__(self):
        self.N = 0 # visited count N in UCB exploration
        self.Q = 0 # quality Q in UCB exploration
        self.children = []
        self.parent = None
    
    def ucb_select(self, c=0.707): # c is 1/sqrt(2), expermental value
        sel = None
        maxval = 0
        for n in self.children:
            if n.N < 1e-6:
                return n
            ucb = n.Q/n.N + c * np.sqrt(2*self.N/n.N)
            if ucb>maxval:
                maxval = ucb
                sel = n
        return sel
def default_policy():
    """
    random action (uniform)
    """
def backup():
    """
    update Q, V; return to root
    """
def expand():
    """
    expand a node
    """
def tree_policy():
    """
    select and expand node
    """

def uct_search():
    """
    entry for ucb
    """# backup current env as root env
    
    # within computational budget:

        # 
    
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