import os
import gym
import numpy as np
from copy import deepcopy
"""
十二方王牌大车併
https://www.youtube.com/watch?v=-aACMW5gJwo&list=RDNlPy34UgdlQ&index=2
"""
class MCTS(object):
    class Node(object):
        def __init__(self, env, action = None, reward = 0, done = False, leaf_budget = 10):
            self.N = 0 # visited count N in UCB exploration
            self.Q = 0 # quality Q in UCB exploration
            self.children = []
            self.action = action # the action that leads to this node 
            self.parent = None
            self.env = deepcopy(env)
            self.reward = reward
            self.done = done
            self.leaf_budget = leaf_budget
            
    def __init__(self, env):
        self.env = deepcopy(env)
        
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
        
    def default_policy(self, v):
        """
        random action (uniform). Could be a smarter policy, but currently this is enough...
        """
        k = self.leaf_budget
        env_leaf = deepcopy(v.env)
        total_reward = 0
        while v.done == False and (k is not None and k > 0):
            action = self.env.action_space.sample()
            next_state, reward, done, info = env_leaf.step(action)
            total_reward += reward
            if k is not None:
                k = k - 1
        
        return total_reward
            
    def backup(self, v, r):
        """
        update the Q, V of all nodes previous to this current v.
        """
        while v is not None:
            v.N = v.N +1
            v.reward = v.reward + r
            v = v.parent
        
    def is_fully_expand(self, v):
        num_total_action = self.env.action_space.n
        return len(v.children) < num_total_action
        
    def expand(self,v):
        """
        expand a node
        """
        action_set = set([a for a in v.children.action])
        action = v.env.action_space.sample()
        while action in action_set:
            action = self.env.action_space.sample()
        env_next = deepcopy(v.env)
        next_state, reward, done, info = env_next.step(action)
        new_child = Node(env = env_next, action = action, reward = reward, done = done)
        v.children.append(new_child)
        ret = new_child
        return ret
    
    def tree_policy(self,v):
        """
        select and expand node
        """
        # is nontermial: need depth judgement or not?
        while v.done == False:
            if not self.is_fully_expand(v):
                return self.expand(v)
            v = self.ucb_select()
        return v

    def mcts_search(self, search_depth = 10):
        """
        entry for ucb
        """
        v0 = Node(self.env)
        for i in range(search_depth):
            v1 = self.tree_policy(v0)
            # problem here: already execute something in tree policy, so the reward accumulation in 'expand' and 'default policy' is hot consensus???
            r = self.default_policy(v1)
            self.backup(v1, r)
        return self.ucb_select(0).action
        
# https://zhuanlan.zhihu.com/p/30458774
def run_env(env_name, n_episode=300, m_steps=1000):
    env = gym.make(env_name)
    
    for i in range(n_episode):
        state = env.reset()
        mcts = MCTS(env)
        for j in range(m_steps):
            mcts = MCTS(env)
            act = mcts.mcts_search()
            next_state, r, d, info = env.step(act)
            
            
            # https://gist.github.com/blole/dfebbec182e6b72ec16b66cc7e331110
            # line 83
            # https://github.com/tobegit3hub/ml_implementation/blob/master/monte_carlo_tree_search/mcst_example.py
    

if __name__ == "__main__":
    run_env("cartpole")