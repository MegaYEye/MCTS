import os
import gym
import numpy as np
from copy import deepcopy
import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger('mainlogger')
logger.info('main file log')
def log_in_out(func):
    
    def decorated_func(*args, **kwargs):
        logger.info("Enter "+ func.__name__)
        result = func(*args, **kwargs)
        logger.info("Leave "+ func.__name__)
        return result

    return decorated_func
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
            
            
    def __init__(self, env, leaf_budget=10):
        self.env = deepcopy(env)
        self.leaf_budget = leaf_budget
        
    # @log_in_out
    def ucb_select(self, v, c=0.707): # c is 1/sqrt(2), expermental value
        sel = None
        maxval = -1e10
        for n in v.children:
            if n.N < 1e-6:
                 return n
            ucb = n.Q/n.N + c * np.sqrt(2*v.N/n.N)
            if ucb>maxval:
                maxval = ucb
                sel = n
        if sel is None:
            logger.error("wtf")
        return sel
    
    # @log_in_out
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
    
    # @log_in_out
    def backup(self, v, r):
        """
        update the Q, V of all nodes previous to this current v.
        """
        # here I consider something called edge reward.
        edge_reward=0
        while v is not None:
            v.N = v.N + 1
            v.Q = v.Q + r + edge_reward
            edge_reward = v.reward
            v = v.parent
            
    def is_fully_expand(self, v):
        num_total_action = self.env.action_space.n
        return len(v.children) == num_total_action
    
    # @log_in_out
    def expand(self,v):
        """
        expand a node
        """
        # this place needs improvements, but at least now, it is okay..
        action_set = set([a.action for a in v.children])
        action = v.env.action_space.sample()
        while action in action_set:
            action = self.env.action_space.sample()
        env_next = deepcopy(v.env)
        next_state, reward, done, info = env_next.step(action)
        new_child = self.Node(env = env_next, action = action, reward = reward, done = done)
        v.children.append(new_child)
        ret = new_child
        return ret
    
    # @log_in_out
    def tree_policy(self,v):
        """
        select and expand node
        """
        # is nontermial: need depth judgement or not?
        while v.done == False:
            if not self.is_fully_expand(v):
                return self.expand(v)

            v_bak = v
            v = self.ucb_select(v)
            
        return v
    
    # @log_in_out
    def mcts_search(self, search_depth = 30):
        """
        entry for ucb
        """
        v0 = self.Node(self.env)
        for i in range(search_depth):
            v1 = self.tree_policy(v0)
            # problem here: already execute something in tree policy, so the reward accumulation in 'expand' and 'default policy' is hot consensus???
            r = self.default_policy(v1)
            self.backup(v1, r)
        return self.ucb_select(v0, 0).action
        
# https://zhuanlan.zhihu.com/p/30458774
def run_env(env_name, n_episode=1, m_steps=1000):
    env = gym.make(env_name)
   
    for i in range(n_episode):
        total_reward = 0
        state = env.reset()
        mcts = MCTS(env)
        for j in range(m_steps):
            # logger.info("step start")
            print("step %d, reward %d" % (j, total_reward) )
            mcts = MCTS(env)
            act = mcts.mcts_search()
            next_state, r, d, info = env.step(act)
            total_reward += r
            # logger.info(next_state)
            if d:
                logger.info("episode done!")
                logger.info(total_reward)
                break
            # logger.info(total_reward)
            

    
# why the performance is highly random????
if __name__ == "__main__":
    run_env("CartPole-v1")