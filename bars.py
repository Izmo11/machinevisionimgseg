import matplotlib
matplotlib.use('Agg') # This tells it to render without a graphical display
import pandas as pd
import os
import matplotlib.pyplot as plt


paths = {
    'Data': 'data/CSVs/train.csv',
    'Train': 'data/CSVs/train.csv',
    'Validation': 'data/CSVs/validation.csv',
    'Test': 'data/CSVs/test.csv'
}

counts = {}
for split, path in paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        counts[split] = len(df)
    else:
        print(f"Warning: {path} not found.")
        counts[split] = 0

plt.figure(figsize=(8, 5))
bars = plt.bar(counts.keys(), counts.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Add labels, title, and adjust y-axis limit
plt.title('Dataset Sample Counts', fontsize=14)
plt.ylabel('Number of Images', fontsize=12)
plt.ylim(0, max(counts.values()) + 20)

# Add exact counts above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Adjust layout and save the chart as a PNG image
plt.tight_layout()
plt.savefig('dataset_splits_chart.png')
print("Chart successfully saved as dataset_splits_chart.png")