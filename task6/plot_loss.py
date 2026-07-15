import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():
    # Read the log file generated during training
    try:
        df = pd.read_csv('training_log.csv')
    except FileNotFoundError:
        print("Could not find training_log.csv")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot Train and Val Loss
    plt.plot(df['step'], df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(df['step'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    
    plt.title('CLIP Model Training Loss over Steps', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save the graph
    plt.savefig('training_curve.png', bbox_inches='tight', dpi=300)
    print("Successfully saved training_curve.png!")

if __name__ == '__main__':
    plot_metrics()