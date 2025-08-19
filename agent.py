import os
import torch
import random
from collections import deque
from snake_game import SnakeGame
from model import QTrainer, Linear_QNet

class Agent:
    
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.9
        self.epsilon = 1.0  
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.995 
        self.n_games = 0 
        self.memory = deque(maxlen = 10000)  
        self.batch_size = 1000
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(model=self.model, lr=self.lr, gamma=self.gamma)


    def get_state(self, game: SnakeGame):
        head = game.snake[0]
        curr_dir = game.direction
        dir_idx = game.DIRECTIONS.index(curr_dir)

        point_straight = (head[0] + curr_dir[0], head[1] + curr_dir[1])
        
        dir_right = game.DIRECTIONS[(dir_idx + 1) % 4]
        point_right = (head[0] + dir_right[0], head[1] + dir_right[1])
        
        dir_left = game.DIRECTIONS[(dir_idx - 1 + 4) % 4]
        point_left = (head[0] + dir_left[0], head[1] + dir_left[1])

        # identifying the danger
        danger_straight = game._is_collision(point_straight)
        danger_right = game._is_collision(point_right)
        danger_left = game._is_collision(point_left)

        # current direction
        dir_r = curr_dir == (1, 0)
        dir_d = curr_dir == (0, 1)   
        dir_l = curr_dir == (-1, 0)
        dir_u = curr_dir == (0, -1)  

        # food location 
        food_up = game.food[1] < head[1]    
        food_down = game.food[1] > head[1]  
        food_left = game.food[0] < head[0]
        food_right = game.food[0] > head[0]

        state = [
            danger_straight, danger_right, danger_left,
            dir_r, dir_d, dir_l, dir_u,
            food_up, food_down, food_left, food_right
        ]
        
        return state


    def get_action(self, state):
       
        final_move = [0, 0, 0] 

        # exploration 
        if random.random() < self.epsilon: 
            move = random.randint(0, 2)
            final_move[move] = 1
        # exploitation
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def remember(self, experience):
        self.memory.append(experience)
        
    def train_short_memory(self, experience):
        self.trainer.train_step([experience])

    def train_long_memory(self):

        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        self.trainer.train_step(mini_sample)

    def load(self, file_name='model.pth'):
        model_folder_path = 'SnakeAI/model'
        file_path = os.path.join(model_folder_path, file_name)
        
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.n_games = checkpoint.get('n_games', self.n_games)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)

            plot_scores = checkpoint.get('plot_scores', [])
            plot_mean_scores = checkpoint.get('plot_mean_scores', [])

            self.model.train()
            print(f"Model and training state loaded from {file_path}")
            return True, plot_scores, plot_mean_scores
        else:
            print(f"Model file not found at {file_path}")
            return False, [], []
        