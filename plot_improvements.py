import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tuning_improvement():
    """
    Generates a bar chart to visualize the gradual improvement
    of the model's RMSE score across different tuning stages.
    """
    # Data from our tuning runs (best RMSE from each stage)
    # These values are taken from the results you generated.
    stages = [
        'Initial Tune\n(n_factors/epochs)', 
        'Coarse Tune\n(lr/reg)', 
        'Fine Tune\n(lr/reg)'
    ]
    rmse_scores = [0.8465, 0.8350, 0.8182]

    # Create a pandas DataFrame for plotting
    data = pd.DataFrame({
        'Tuning Stage': stages,
        'Best RMSE Score': rmse_scores
    })

    # Create the plot
    plt.figure(figsize=(10, 7))
    barplot = sns.barplot(x='Tuning Stage', y='Best RMSE Score', data=data, palette='viridis')

    # Add labels and title
    plt.title('SVD Model RMSE Improvement Through Hyperparameter Tuning', fontsize=16)
    plt.xlabel('Tuning Stage', fontsize=12)
    plt.ylabel('Root Mean Squared Error (RMSE) - Lower is Better', fontsize=12)
    
    # Set the y-axis limit to make the differences more visible
    min_score = min(rmse_scores)
    plt.ylim(min_score - 0.01, max(rmse_scores) + 0.01)

    # Add the score value on top of each bar for clarity
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.4f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')

    # Save the plot to a file
    output_filename = 'improvement_barchart.png'
    plt.savefig(output_filename)
    print(f"Bar chart saved to '{output_filename}'")

if __name__ == "__main__":
    plot_tuning_improvement()

