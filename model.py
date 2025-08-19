import os
import torch
import torch.nn as nn

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()
    
    def train_step(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        pred = self.model(states)

        target = pred.clone()

        non_final_mask = ~dones
        non_final_next_states = next_states[non_final_mask]

        q_new = rewards.clone()

        if len(non_final_next_states) > 0:
            next_q_values = self.model(non_final_next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            q_new[non_final_mask] += self.gamma * max_next_q

        action_indices = torch.argmax(actions, dim=1)
        target[torch.arange(len(states)), action_indices] = q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def save(self, file_name='model.pth', **kwargs):
        
        model_folder_path = 'SnakeAI/model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_score': kwargs.get('total_score', 0),
            'n_games': kwargs.get('n_games', 0),
            'epsilon': kwargs.get('epsilon', 1.0),
            'record': kwargs.get('record', 0),
            'plot_scores': kwargs.get('plot_scores', []),
            'plot_mean_scores': kwargs.get('plot_mean_scores', [])
        }
        torch.save(state, file_path)