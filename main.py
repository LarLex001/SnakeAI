import sys
import pygame
from agent import Agent
from snake_game import SnakeGame
from plotter import plot_and_save

LOAD_EXISTING_MODEL = True
MAX_NUM_GAMES = 550

def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    game = SnakeGame()
    agent = Agent()
    loaded, loaded_scores, loaded_mean_scores = agent.load('model.pth') 

    if loaded and LOAD_EXISTING_MODEL:
        plot_scores = loaded_scores
        plot_mean_scores = loaded_mean_scores
        total_score = sum(plot_scores)
        record = max(plot_scores) if plot_scores else 0

        print(f"Continuing training. Start game: {agent.n_games}, Record: {record}")
    else:
        print("Model not found. Starting new training.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        old_state = agent.get_state(game)

        # for reward shaping
        head = game.snake[0]
        food = game.food
        old_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])

        action = agent.get_action(old_state)
        reward, game_over, score = game.play_step(action)

        new_head = game.snake[0]
        new_distance = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])

        if new_distance < old_distance:
            reward += 0.1 
        else:
            reward -= 0.1 

        new_state = agent.get_state(game)

        agent.train_short_memory((old_state, action, reward, new_state, game_over))
        agent.remember((old_state, action, reward, new_state, game_over))

        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            if score > record:
                record = score
                agent.trainer.save(
                    epsilon=agent.epsilon, 
                    n_games=agent.n_games, 
                    record=record, 
                    total_score=total_score,
                    plot_scores=plot_scores, 
                    plot_mean_scores=plot_mean_scores
                )

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            if agent.n_games >= MAX_NUM_GAMES:
                break

    return plot_scores, plot_mean_scores

if __name__ == '__main__':
    scores, mean_scores = train()
    plot_and_save(scores, mean_scores)