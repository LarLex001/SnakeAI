import os
import matplotlib.pyplot as plt

def plot_and_save(scores, mean_scores, filename='training_progress.png'):
    plt.clf()
    
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}")

    plot_folder = 'SnakeAI/plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    file_path = os.path.join(plot_folder, filename)
    plt.savefig(file_path)
    print(f"Graph saved to {file_path}")