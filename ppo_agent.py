import torch
import torch.nn.functional as F
import numpy as np
import random
import math

from ppo_model import PolicyNetwork, CriticNetwork
gamma = 0.99
lamda = 0.95
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_TD_residuals(rewards, values, gamma=gamma):
    TD_residuals = torch.zeros(rewards.size()[0], dtype=torch.float, device=device)
    GAMMA = torch.tensor(gamma, dtype=torch.float, device=device) 
    for k in range(rewards.size()[0]-1):    
        TD_residuals[k] = rewards[k] + GAMMA * values[k+1].detach() - values[k].detach()
    return TD_residuals

def get_adv_and_returns(rewards, values, gamma=gamma, lamda=lamda):
    R = get_TD_residuals(rewards, values)
    GAMMA = torch.tensor(gamma, dtype=torch.float, device=device)
    LAMDA = torch.full_like(GAMMA, lamda)
    advantages = torch.zeros_like(R)
    returns = torch.zeros_like(R)
    advantages[-1] = R[-1]
    returns[-1] = rewards[-1]
    for k in range(rewards.size()[0]-2,-1,-1):
            advantages[k] = R[k].to(device) + GAMMA * LAMDA * advantages[k+1].to(device)
            returns[k] = rewards[k] + GAMMA * returns[k+1].to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
    return advantages, returns
            

def get_log_probs(actions, means, stds):
    var = torch.pow(stds,2)
    components_log_probs = -(actions - means)**2 / (2 * var) - 0.5 * (math.log(2 * math.pi) + torch.log(stds))
    actions_log_probs = torch.sum(components_log_probs, 1)
    return actions_log_probs

    
def get_best_action(policy, state):
    policy.eval()
    with torch.no_grad():
        dist = policy(state)[0]
        best_action = dist.sample()
    policy.train()
    return torch.clamp(best_action,-1,1)
    

def collect_trajectory(env, brain_name, agent):
    env_info = env.reset(train_mode=True)[brain_name]
    STATES = torch.empty(agent.buffer_size, agent.state_size).to(device, torch.float)
    ACTIONS = torch.empty(agent.buffer_size, agent.action_size).to(device, torch.float)
    REWARDS = torch.zeros(agent.buffer_size).to(device, torch.float)
   
    for t in range(agent.buffer_size):
        STATES[t] = torch.from_numpy(env_info.vector_observations).to(device, torch.float)
        action = get_best_action(agent.policy, STATES[t].unsqueeze(0))
        ACTIONS[t] = action.to(torch.float)
        env_info = env.step(action.cpu().detach().numpy())[brain_name]                           
        REWARDS[t] = torch.from_numpy(np.asarray(env_info.rewards)).to(device, torch.float)
        if np.any(env_info.local_done):
            env_info = env.reset(train_mode=True)[brain_name]
            
    means = agent.policy(STATES)[1]
    stds = agent.policy.stds[0]
    LOG_PROBS = get_log_probs(ACTIONS, means, stds)
    VALUES = agent.critic(STATES)
    ADVANT, RETURNS = get_adv_and_returns(REWARDS, VALUES)
    score = torch.sum(REWARDS)
        
    return STATES, ACTIONS, LOG_PROBS.detach(), RETURNS, ADVANT, VALUES.clone(), score

class Agent():
    def __init__(self, state_size, action_size, buffer_size, batch_size, num_epochs, hidden_units, learn_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.policy = PolicyNetwork(state_size, action_size, hidden_units).to(device) 
        self.critic = CriticNetwork(state_size, hidden_units).to(device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=learn_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learn_rate)
        
    def perform_learning_with(self, STATES, ACTIONS, LOG_PROBS, RETURNS, ADVANT, VALUES, epsilon):
        
        buffer_size = STATES.size()[0]
        
        STATES[:] = STATES[:].to(device)
        ACTIONS[:] = ACTIONS[:].to(device)
        LOG_PROBS[:] = LOG_PROBS[:].to(device)
        RETURNS[:] = RETURNS[:].to(device)
        ADVANT[:] = ADVANT[:].to(device)
        VALUES[:] = VALUES[:].to(device)
        
        T = np.arange(buffer_size)
        for epoch in range(self.num_epochs):
            np.random.shuffle(T)
            for i in range(buffer_size//self.batch_size):    
                batch_index = T[self.batch_size*i : self.batch_size*(i+1)]
                batch_index = torch.LongTensor(batch_index).to(device)
                batch_states = STATES[batch_index]
                batch_actions = ACTIONS[batch_index]
                batch_advant = ADVANT[batch_index]
                batch_returns = RETURNS[batch_index] 
                batch_old_values = VALUES[batch_index].detach()
                 
                means = self.policy(batch_states)[1]
                stds = self.policy.stds[0]
                new_log_probs = get_log_probs(batch_actions, means.to(device), stds.to(device))
                old_log_probs = LOG_PROBS[batch_index]
       
                ratios = torch.exp(new_log_probs-old_log_probs)
                clipped_ratios = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
                actor_losses = ratios * batch_advant
                clipped_actor_losses = clipped_ratios * batch_advant
                losses = -torch.min(actor_losses, clipped_actor_losses).to(device)
                surrogate_loss = losses.mean()
       
                batch_new_values = self.critic(batch_states)
                batch_clipped_values = batch_old_values + torch.clamp(batch_new_values - batch_old_values, -epsilon, epsilon)
                mse = torch.nn.MSELoss(reduce=False)
                MSE_losses = mse(batch_new_values, batch_returns.unsqueeze(1))
                clipped_MSE_losses = mse(batch_clipped_values, batch_returns.unsqueeze(1))
                losses = torch.max(MSE_losses,clipped_MSE_losses).to(device)
                critic_loss = losses.mean()
         
        
                total_loss = surrogate_loss + 0.5 * critic_loss
            
                self.critic_optim.zero_grad()
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
                self.critic_optim.step()

                self.policy_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm(self.policy.parameters(),5)
                self.policy_optim.step()