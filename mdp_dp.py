### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np
from numpy import argmax

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy for all the differnt states.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    # First Part of Policy Iteration
    # Intilize state value function size of possible states
    #State Value function give you the value of each state under a specific policy,
    # assuming the agent continues to act according to that policy.
    V_Expected = np.zeros(nS)


    while True:
        delta=0# change in expected value
        for s in range(nS):#iterate over all states
            V=0
            for a in range(aS): #iterate over all the actions per state
                for prob, next_state, reward, done in P[s][a]: #iterate over all possible probs for the action ie account for slippery tiles
                 V += policy[s][a] * prob*(reward+gama*V_Expected[next_state])
                 # Calculate the expected return for this transition,
                 # then add it to the total value of the current state-action pair
                 #p(s',r|s,a),  probability of ending up in next state s' with reward r, given that the agent was in state s and took action a

            delta = max(delta, V_Expected(s) - V_Expected(s - 1))
            V_Expected[s]= V #Update the array with current value


        if delta < tol: # ie value has converged dif is soo soo ssmall thus break out of loop as we found expected value for state action pair
            break


        return V_Expected




def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """
    # Action Value Function give a value for each possible action in each state

    new_policy = np.ones([nS, nA]) / nA

    policy_stable==True

    for s in range(nS):
        Prev_action_Value=np.copy(policy[s])
        #  given a state s and an action a, the value of the action is the expected
        #  future reward the agent will receive if it takes action a in state s and then follows its policy for all future actions.
            q=np.zeros(nA)
            for a in range(aS):
                for prob, next_state, reward, done  in P[s][a]:
                q[a]+= prob * (reward + gama * V_Expected[next_state])

            Sick_Action =np.argmax(q) #tells you value of each action
            policy[s] = np.eye(nA)[Sick_Action]  # creates identity matrix and vector where the best action is a 1

            if np.all(policy[s]) != np.all(Prev_action_Value):
                policy_stable== False

            elif:
                policy_stable == True
                 break
    return policy



#Argmax used to return the the class with the largest predicted probality



def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
"""
    Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
"""
while True:
    new_policy = policy.copy()
    value_function = policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8)
    new_policy = policy_improvement(P, nS, nA, value_from_policy, gamma=0.9)
    if new_policy == True:
        break
    return value_function, new_policy

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
"""
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #

    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
"""
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
 """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            
    return total_rewards



