import os
import gym
import numpy as np
class MCTS(object):
    class Node(object):
        def __init__(self, state, action = None, reward = 0, done = False):
            self.N = 0 # visited count N in UCB exploration
            self.Q = 0 # quality Q in UCB exploration
            self.children = []
            self.action = action # the action that leads to this node 
            self.parent = None
            self.state = state
            self.reward = reward
            
    def __init__(self, env):
        self.env = env
        
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
        
    def default_policy(self):
        """
        random action (uniform)
        """
        pass
    def backup(self):
        """
        update Q, V; return to root
        """
        pass
    def is_fully_expand(self, v):
        num_total_action = self.env.action_space.n
        return len(v.children) < num_total_action
        
    def expand(self,v):
        """
        expand a node
        """
        action_set = set([a for a in v.children.action])
        action = env.action_space.sample()
        while action in action_set:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        new_child = Node(state = next_state, action = action, reward = reward, done = done)
        v.children.append(new_child)
        ret = new_child
        return ret
    
    def tree_policy(self,v):
        """
        select and expand node
        """


    def uct_search(self):
        """
        entry for ucb
        """
        # backup current env as root env
        
        # within computational budget:

        pass
    
# https://zhuanlan.zhihu.com/p/30458774
def run_env(env_name, n_episode=300, m_steps=1000):
    env = gym.make(env_name)
    
    for i in range(n_episode):
        for j in range(m_steps):
            # https://gist.github.com/blole/dfebbec182e6b72ec16b66cc7e331110
            # line 83
            # https://github.com/tobegit3hub/ml_implementation/blob/master/monte_carlo_tree_search/mcst_example.py
    
    
if __name__ == "__main__":
    run_env("cartpole")