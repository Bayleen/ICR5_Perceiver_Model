import pandas as pd

data_path = r'E:\UTK\Reseach\Publication\Workshop\ICRA2025_workshop_dataset_HumanRobotCorr-main\softmax.csv'
prob_df = pd.read_csv(data_path)

class_labels = ['reaching', 'grasping', 'lifting', 'holding', 'transporting', 'placing', 'releasing', 'nothing']

prob_df.columns = class_labels

avg_probs = prob_df.mean().values

for label, prob in zip(class_labels, avg_probs):
    print(f"{label}: {prob:.6f}")