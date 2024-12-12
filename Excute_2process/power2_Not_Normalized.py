from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Function to categorize chlorophyll-a concentrations
def categorize_chl_a(values):
    return np.where(values < 30, 0, np.where(values < 40, 1, 2))

# Load data
data = pd.read_csv('power_2process.csv')

# Initialize dictionaries to save MSEs for each model across iterations
individual_model_results_all = {}
stacked_model_results_all = {}

# Run 10 iterations
for _ in range(3):
    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # # 最適そうな変数
    explanatory_vars = ['layers_1', 'layers_2', 'batch_size_1','batch_size_2','model1_Param', 'model2_Param', 'model1_FLP', 'model2_FLP']

    # Setup the experiment with selected explanatory variables
    exp1 = setup(train_data[explanatory_vars + ['Power']], target='Power')


    # Compare models and get the top 15
    models_comparison = compare_models(sort='MAE', n_select=18)
    
    # Collect MSE for individual models
    for model in models_comparison:
        predictions = predict_model(model, data=test_data)
        mae = abs(predictions['Power'] - predictions['prediction_label']).mean()

        model_name = model.__class__.__name__
        if model_name not in individual_model_results_all:
            individual_model_results_all[model_name] = []
        individual_model_results_all[model_name].append(mae)

    # Select models for stacking
    models_list1 = compare_models(sort='MAE', n_select=3)
    models_list2 = compare_models(sort='MAE', n_select=5)
    models_list3 = compare_models(sort='MAE', n_select=10)
    
    
    
    # Create stacked models
    for i, models_list in enumerate([models_list1, models_list2, models_list3], 1):
        stacked_model = stack_models(estimator_list=models_list, meta_model=LinearRegression())
        predictions = predict_model(stacked_model, data=test_data)
        mae = abs(predictions['Power'] - predictions['prediction_label']).mean()

        model_name = f'Stacked_Model_{i}'
        if model_name not in stacked_model_results_all:
            stacked_model_results_all[model_name] = []
        stacked_model_results_all[model_name].append(mae)

        # Confusion matrix for the first stacked model
        if i == 1:
            y_test_category = categorize_chl_a(test_data['Power'])
            pred_category = categorize_chl_a(predictions['prediction_label'])
            cm = confusion_matrix(y_test_category, pred_category)            
                
        # Collect MSE for individual models
    for model in models_comparison:
        predictions = predict_model(model, data=test_data)
        mae = abs(predictions['Power'] - predictions['prediction_label']).mean()
        model_name = model.__class__.__name__
        if model_name not in individual_model_results_all:
            individual_model_results_all[model_name] = []
        individual_model_results_all[model_name].append(mae)

    # Stacked model predictions
    for i, models_list in enumerate([models_list1, models_list2, models_list3], 1):
        stacked_model = stack_models(estimator_list=models_list, meta_model=LinearRegression())
        predictions = predict_model(stacked_model, data=test_data)
        mae = abs(predictions['Power'] - predictions['prediction_label']).mean()
        model_name = f'Stacked_Model_{i}'
        if model_name not in stacked_model_results_all:
            stacked_model_results_all[model_name] = []
        stacked_model_results_all[model_name].append(mae)

# Calculate average MAE for each model
average_mae_results = {
    model: np.mean(mae_list) for model, mae_list in {**individual_model_results_all, **stacked_model_results_all}.items()
}

# Convert the results into a DataFrame for plotting
average_mae_df = pd.DataFrame(list(average_mae_results.items()), columns=['Model', 'Average MAE'])

# Plot MAE comparison
plt.figure(figsize=(12, 6))
bars = plt.barh(average_mae_df['Model'], average_mae_df['Average MAE'], color='skyblue')
plt.xlabel('Average MAE')
plt.ylabel('Model')
# plt.title('Average Model Comparison on Test Data (MSE) over 10 iterations')
plt.gca().invert_yaxis()
# 余白の調整（右側にスペースを確保）

plt.subplots_adjust(left=0.15)  # 左側に余白を追加


# 右側に余白を追加
plt.xlim(0, max(average_mae_df['Average MAE']) * 1.1)

# 各バーにMSEの値を表示
for bar in bars:
    width = bar.get_width()
    # 右に少し余白を持たせてMSE値を表示
    plt.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
             f'{width:.2f}', va='center', 
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0'))  # bboxで余白追加


plt.show()

print('exp1:',exp1)