import random
import pygame 

BLOCK_SIZE = 20
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class SnakeGame:
    
    DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up
        
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height

        pygame.init()
        self.display = pygame.display.set_mode((self.width * BLOCK_SIZE, self.height * BLOCK_SIZE))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 25)

        self.reset() 

    def reset(self):
        self.direction = (1, 0)
        self.snake = [(self.width // 2, self.height // 2)]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        while True:
            food_x = random.randint(0, self.width - 1)
            food_y = random.randint(0, self.height - 1)
            self.food = (food_x, food_y)
            if self.food not in self.snake:
                break
                
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        if pt[0] < 0 or pt[0] >= self.width or pt[1] < 0 or pt[1] >= self.height:
            return True
        if pt in self.snake[1:]:
            return True
        return False
        
    def play_step(self, action):

        self.frame_iteration += 1
        reward = 0
        game_over = False

        turn = action.index(1)
        dir_idx = self.DIRECTIONS.index(self.direction)

        if turn == 0:
            new_dir_idx = dir_idx
        elif turn == 1:
            new_dir_idx = (dir_idx + 1) % 4
        elif turn == 2:
            new_dir_idx = (dir_idx - 1 + 4) % 4
        
        self.direction = self.DIRECTIONS[new_dir_idx]
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, new_head)
        
        if self._is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if new_head == self.food:
            reward = 10
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(20) 

        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0] * BLOCK_SIZE, pt[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0] * BLOCK_SIZE + 4, pt[1] * BLOCK_SIZE + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0] * BLOCK_SIZE, self.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()
