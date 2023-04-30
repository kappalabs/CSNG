from matplotlib import pyplot as plt
import numpy as np


def scatter_with_minimals(x, y, color, label):
    x = np.array(x)
    y = np.array(y)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # Plot the X and Y values as points
    plt.scatter(x, y, color=color, s=15, label=label)

    # create a variable to keep track of the minimum y value so far
    min_y = np.inf

    # create a list to store the coordinates of the points to be connected
    points_to_connect = []

    # iterate over the data and find the points with the minimum y value so far
    for i in range(len(x)):
        if y[i] < min_y:
            min_y = y[i]
            points_to_connect.append((x[i], y[i]))

    # create the plot and connect the points with the minimum y value so far
    # fig, ax = plt.subplots()
    # ax.scatter(x, y)
    for i in range(len(points_to_connect) - 1):
        plt.plot([points_to_connect[i][0], points_to_connect[i + 1][0]],
                 [points_to_connect[i][1], points_to_connect[i + 1][1]], color=color)


# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.07302, 0.07311, 0.07561, 0.07863, 0.08019]
values_mse = [0.008122, 0.00809, 0.008632, 0.0094, 0.009794]
values_ssim = [0.2229, 0.193, 0.1968, 0.205, 0.1977]
values_msssim = [16.372, 16.285, 16.729, 17.267, 17.513]

X = np.arange(len(models))
plt.bar(X + 0.00, values_l1, color='b', width=0.25)
plt.bar(X + 0.25, values_mse, color='g', width=0.25)
plt.bar(X + 0.50, values_ssim, color='r', width=0.25)
# plt.bar(X + 0.75, values_msssim, color='y', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_l1):
    plt.text(i - 0.05, v + 0.001, str(v), color='black', fontweight='bold')
for i, v in enumerate(values_mse):
    plt.text(i + 0.2, v + 0.001, str(v), color='black', fontweight='bold')
for i, v in enumerate(values_ssim):
    plt.text(i + 0.45, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_ten_trials.png', dpi=300, bbox_inches='tight')
plt.close()

# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.07302, 0.07311, 0.07561, 0.07863, 0.08019]
values_mse = [0.008122, 0.00809, 0.008632, 0.0094, 0.009794]
values_ssim = [0.2229, 0.193, 0.1968, 0.205, 0.1977]
values_msssim = [16.372, 16.285, 16.729, 17.267, 17.513]

X = np.arange(len(models))
plt.bar(X, values_l1, color='b', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_l1):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_ten_trials_l1.png', dpi=300, bbox_inches='tight')
plt.close()

# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.07302, 0.07311, 0.07561, 0.07863, 0.08019]
values_mse = [0.008122, 0.00809, 0.008632, 0.0094, 0.009794]
values_ssim = [0.2229, 0.193, 0.1968, 0.205, 0.1977]
values_msssim = [16.372, 16.285, 16.729, 17.267, 17.513]

X = np.arange(len(models))
plt.bar(X, values_mse, color='g', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
plt.legend(['MSE', 'SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_mse):
    plt.text(i, v + 0.0002, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_ten_trials_mse.png', dpi=300, bbox_inches='tight')
plt.close()

# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.07302, 0.07311, 0.07561, 0.07863, 0.08019]
values_mse = [0.008122, 0.00809, 0.008632, 0.0094, 0.009794]
values_ssim = [0.2229, 0.193, 0.1968, 0.205, 0.1977]
values_msssim = [16.372, 16.285, 16.729, 17.267, 17.513]

X = np.arange(len(models))
plt.bar(X, values_ssim, color='r', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
plt.legend(['SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_ssim):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_ten_trials_ssim.png', dpi=300, bbox_inches='tight')
plt.close()

# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.07302, 0.07311, 0.07561, 0.07863, 0.08019]
values_mse = [0.008122, 0.00809, 0.008632, 0.0094, 0.009794]
values_ssim = [0.2229, 0.193, 0.1968, 0.205, 0.1977]
values_msssim = [16.372, 16.285, 16.729, 17.267, 17.513]

X = np.arange(len(models))
plt.bar(X, values_msssim, color='y', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
plt.legend(['MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_msssim):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_ten_trials_msssim.png', dpi=300, bbox_inches='tight')
plt.close()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_export_2023-04-22T10_28_28.799+02_00.csv')

df = df[df['dataset_limit_train'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_train']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'r', None)

# show y axis grid
plt.grid(axis='y')
plt.grid(axis='x')
# show smaller grid lines
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# Set the title and axis labels
plt.title('Dataset Size vs L1 Loss')
plt.xlabel('Dataset Size')
plt.ylabel('L1 Loss')

# Display the plot
plt.savefig('linear_regression_ten_trials.png', dpi=500, bbox_inches='tight')
plt.close()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.09481, 0.07105, 0.06878, 0.06655, 0.06862]
values_mse = [0.0143, 0.007875, 0.007441, 0.006964, 0.007404]
values_ssim = [0.3451, 0.2049, 0.1992, 0.1906, 0.1921]
values_msssim = [19.876, 15.961, 15.555, 15.116, 15.468]

X = np.arange(len(models))
plt.bar(X + 0.00, values_l1, color='b', width=0.25)
plt.bar(X + 0.25, values_mse, color='g', width=0.25)
plt.bar(X + 0.50, values_ssim, color='r', width=0.25)
plt.bar(X + 0.75, values_msssim, color='y', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_l1):
    plt.text(i - 0.05, v + 0.001, str(v), color='black', fontweight='bold')
for i, v in enumerate(values_mse):
    plt.text(i + 0.2, v + 0.001, str(v), color='black', fontweight='bold')
for i, v in enumerate(values_ssim):
    plt.text(i + 0.45, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_one_trial.png', dpi=300, bbox_inches='tight')
plt.close()

# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.09481, 0.07105, 0.06878, 0.06655, 0.06862]
values_mse = [0.0143, 0.007875, 0.007441, 0.006964, 0.007404]
values_ssim = [0.3451, 0.2049, 0.1992, 0.1906, 0.1921]
values_msssim = [19.876, 15.961, 15.555, 15.116, 15.468]

X = np.arange(len(models))
plt.bar(X, values_l1, color='b', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_l1):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_one_trial_l1.png', dpi=300, bbox_inches='tight')
plt.close()


# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.09481, 0.07105, 0.06878, 0.06655, 0.06862]
values_mse = [0.0143, 0.007875, 0.007441, 0.006964, 0.007404]
values_ssim = [0.3451, 0.2049, 0.1992, 0.1906, 0.1921]
values_msssim = [19.876, 15.961, 15.555, 15.116, 15.468]

X = np.arange(len(models))
plt.bar(X, values_mse, color='g', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
# plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
plt.legend(['MSE', 'SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_mse):
    plt.text(i, v + 0.0002, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_one_trial_mse.png', dpi=300, bbox_inches='tight')
plt.close()

# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.09481, 0.07105, 0.06878, 0.06655, 0.06862]
values_mse = [0.0143, 0.007875, 0.007441, 0.006964, 0.007404]
values_ssim = [0.3451, 0.2049, 0.1992, 0.1906, 0.1921]
values_msssim = [19.876, 15.961, 15.555, 15.116, 15.468]

X = np.arange(len(models))
plt.bar(X, values_ssim, color='r', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
# plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
plt.legend(['SSIM', 'MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_ssim):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_one_trial_ssim.png', dpi=300, bbox_inches='tight')
plt.close()

# %%%
models = ['LR', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4']
values_l1 = [0.09481, 0.07105, 0.06878, 0.06655, 0.06862]
values_mse = [0.0143, 0.007875, 0.007441, 0.006964, 0.007404]
values_ssim = [0.3451, 0.2049, 0.1992, 0.1906, 0.1921]
values_msssim = [19.876, 15.961, 15.555, 15.116, 15.468]

X = np.arange(len(models))
plt.bar(X, values_msssim, color='y', width=0.25)
plt.xticks(X, models)
plt.title('Models Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Model')
# plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
plt.legend(['MSSSIM'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_msssim):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('models_evaluation_one_trial_msssim.png', dpi=300, bbox_inches='tight')
plt.close()


# %%%
losses = ['L1', 'MSE', 'SSIM', 'MSSSIM', 'MIX', 'Adversarial']
values_l1 = [0.06943, 0.07141, 0.06904, 0.06655, 0.06845, 0.06903]
values_mse = [0.007379, 0.007632, 0.007226, 0.006964, 0.007227, 0.00781]
values_ssim = [0.1944, 0.1968, 0.1919, 0.1906, 0.1938, 0.2097]
values_msssim = [15.618, 15.947, 15.541, 15.116, 15.44, 15.577]

X = np.arange(len(losses))
plt.bar(X, values_l1, color='b', width=0.25)
plt.xticks(X, losses)
plt.title('Loss Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Training loss')
# plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
plt.legend(['L1'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_l1):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('model_loss_one_trial.png', dpi=300, bbox_inches='tight')
plt.close()


# %%%
losses = ['L1', 'MSE', 'SSIM', 'MSSSIM', 'MIX', 'Adversarial']
values_l1 = [0.06943, 0.07141, 0.06904, 0.06655, 0.06845, 0.06903]
values_mse = [0.007379, 0.007632, 0.007226, 0.006964, 0.007227, 0.00781]
values_ssim = [0.1944, 0.1968, 0.1919, 0.1906, 0.1938, 0.2097]
values_msssim = [15.618, 15.947, 15.541, 15.116, 15.44, 15.577]

X = np.arange(len(losses))
plt.bar(X, values_l1, color='b', width=0.25)
plt.xticks(X, losses)
plt.title('Loss Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Training loss')
# plt.legend(['L1', 'MSE', 'SSIM', 'MSSSIM'])
plt.legend(['L1'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_l1):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('model_loss_one_trial_l1.png', dpi=300, bbox_inches='tight')
plt.close()


# %%%
losses = ['L1', 'MSE', 'SSIM', 'MSSSIM', 'MIX', 'Adversarial']
values_l1 = [0.06943, 0.07141, 0.06904, 0.06655, 0.06845, 0.06903]
values_mse = [0.007379, 0.007632, 0.007226, 0.006964, 0.007227, 0.00781]
values_ssim = [0.1944, 0.1968, 0.1919, 0.1906, 0.1938, 0.2097]
values_msssim = [15.618, 15.947, 15.541, 15.116, 15.44, 15.577]

X = np.arange(len(losses))
plt.bar(X, values_mse, color='g', width=0.25)
plt.xticks(X, losses)
plt.title('Loss Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Training loss')
plt.legend(['MSE'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_mse):
    plt.text(i, v + 0.0002, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('model_loss_one_trial_mse.png', dpi=300, bbox_inches='tight')
plt.close()


# %%%
losses = ['L1', 'MSE', 'SSIM', 'MSSSIM', 'MIX', 'Adversarial']
values_l1 = [0.06943, 0.07141, 0.06904, 0.06655, 0.06845, 0.06903]
values_mse = [0.007379, 0.007632, 0.007226, 0.006964, 0.007227, 0.00781]
values_ssim = [0.1944, 0.1968, 0.1919, 0.1906, 0.1938, 0.2097]
values_msssim = [15.618, 15.947, 15.541, 15.116, 15.44, 15.577]

X = np.arange(len(losses))
plt.bar(X, values_ssim, color='r', width=0.25)
plt.xticks(X, losses)
plt.title('Loss Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Training loss')
plt.legend(['MSE'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_ssim):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('model_loss_one_trial_ssim.png', dpi=300, bbox_inches='tight')
plt.close()


# %%%
losses = ['L1', 'MSE', 'SSIM', 'MSSSIM', 'MIX', 'Adversarial']
values_l1 = [0.06943, 0.07141, 0.06904, 0.06655, 0.06845, 0.06903]
values_mse = [0.007379, 0.007632, 0.007226, 0.006964, 0.007227, 0.00781]
values_ssim = [0.1944, 0.1968, 0.1919, 0.1906, 0.1938, 0.2097]
values_msssim = [15.618, 15.947, 15.541, 15.116, 15.44, 15.577]

X = np.arange(len(losses))
plt.bar(X, values_msssim, color='y', width=0.25)
plt.xticks(X, losses)
plt.title('Loss Evaluation')
plt.ylabel('Loss value')
plt.xlabel('Training loss')
plt.legend(['MSE'])
# show y axis grid
# plt.grid(axis='y')
# show smaller grid lines
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# show the values on top of the bars
for i, v in enumerate(values_msssim):
    plt.text(i, v + 0.001, str(v), color='black', fontweight='bold')

# plt.show()
# Save the figure
plt.savefig('model_loss_one_trial_msssim.png', dpi=300, bbox_inches='tight')
plt.close()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_response_size_one_trial.csv')

df = df[df['dataset_limit_responses'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'r', None)

# show y axis grid
plt.grid(axis='y')
plt.grid(axis='x')
# show smaller grid lines
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# Set the title and axis labels
plt.title('Number of Neurons vs L1 Loss')
plt.xlabel('Number of Neurons')
plt.ylabel('L1 Loss')

# Display the plot
plt.savefig('response_size_one_trial.png', dpi=500, bbox_inches='tight')
plt.close()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_export_2023-04-22T16_04_05.638+02_00.csv')

df = df[df['dataset_limit_train'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_train']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'r', None)

# show y axis grid
plt.grid(axis='y')
plt.grid(axis='x')
# show smaller grid lines
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# Set the title and axis labels
plt.title('Dataset Size vs L1 Loss')
plt.xlabel('Dataset Size')
plt.ylabel('L1 Loss')

# Display the plot
plt.savefig('dataset_size_one_trial.png', dpi=500, bbox_inches='tight')
plt.close()

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the JSON file
with open('results/loss_avg_for_each_augmentation_and_criterion.json', 'r') as f:
    data = json.load(f)

# Define colors for each metric
colors = {
    'L1': 'b',
    'MSE': 'r',
    'SSIM': 'g',
    'MSSSIM': 'y',
    'MIX': 'purple'
}

augmentations = {
    'blur_65_10': 'Gauss blur',
    'noise_0.1': 'Gauss noise',
    'value_noise_0.07': 'Value noise',
    'value_shift_0.07': 'Intensity',
    'image_shift_x_9': 'Translation',
}

# Plot bars for each metric
for metric in data:
    fig, ax = plt.subplots()
    names = list(data[metric].keys())[1:]
    names = [augmentations[name] for name in names]
    values = list(data[metric].values())[1:]
    ax.bar(names, values, color=colors[metric])
    ax.set_xlabel('Noise Type')
    ax.set_ylabel('Metric Value')
    ax.set_title(metric)
    for i, v in enumerate(values):
        ax.text(i, v, str(round(v, 4)), ha='center', va='bottom', color='black')
    # save the figure
    plt.savefig(f'augmentation_{metric}.png', dpi=500, bbox_inches='tight')
    plt.close()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# ALL

df = pd.read_csv('results/wandb_response_size_one_trial.csv')

df = df[df['dataset_limit_responses'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'r', 'All')

# L4

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_l4_num_neurons_cnnv3.csv')

df = df[df['dataset_limit_responses'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'g', 'L4')

# L2/3

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_l23_num_neurons_cnnv3.csv')

df = df[df['dataset_limit_responses'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'b', 'L2/3')

# show y axis grid
plt.grid(axis='y')
plt.grid(axis='x')
# show smaller grid lines
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# add labels for each scatter
# plt.legend(['All', 'L4', 'L2/3'])
plt.legend()

# Set the title and axis labels
plt.title('Number of Neurons vs L1 Loss')
plt.xlabel('Number of Neurons')
plt.ylabel('L1 Loss')

# Display the plot
plt.savefig('responses_size_groups_one_trial.png', dpi=500, bbox_inches='tight')
plt.close()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ALL

df = pd.read_csv('results/wandb_response_size_one_trial.csv')

df = df[df['dataset_limit_responses'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses'].tolist()
y = df['test.L1_central'].tolist()

scatter_with_minimals(x, y, 'r', 'All')

# L4

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_l4_num_neurons_cnnv3.csv')

df = df[df['dataset_limit_responses'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses']
y = df['test.L1_central']

scatter_with_minimals(x, y, 'g', 'L4')

# L4 Exc

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_l4_exc_num_responses.csv')

df = df[df['dataset_limit_responses'] > 200]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'y', 'L4 Exc')

# L4 Inh

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('results/wandb_l4_inh_num_responses.csv')

df = df[df['dataset_limit_responses'] > 200]
df = df[df['test.L1_central'] < 0.3]

# Extract the X and Y values from the DataFrame
x = df['dataset_limit_responses']
y = df['test.L1_central']

# Plot the X and Y values as points
scatter_with_minimals(x, y, 'purple', 'L4 Inh')

# show y axis grid
plt.grid(axis='y')
plt.grid(axis='x')
# show smaller grid lines
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# add labels for each scatter
plt.legend()

# Set the title and axis labels
plt.title('Number of Neurons vs L1 Loss')
plt.xlabel('Number of Neurons')
plt.ylabel('L1 Loss')

# Display the plot
plt.savefig('responses_size_l4_groups_one_trial.png', dpi=500, bbox_inches='tight')
plt.close()
