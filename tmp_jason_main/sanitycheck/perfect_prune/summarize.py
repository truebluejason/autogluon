import ast
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

df = pd.read_csv('sanitycheck/perfect_prune/result.csv')
datasets = df['dataset'].unique()
models = df['model'].unique()
tasks = [task for task in df['task_type'].unique() if task != 'original']

# print average rank of noise configuration per model across datasets
model_winrate_dicts = []
for model in models:
    model_winrate_dict = {'model': model}
    for task in tasks:
        val_wins, val_ties, val_losses = [], [], []
        test_wins, test_ties, test_losses = [], [], []
        for dataset in datasets:
            original_row = df[(df['task_type'] == 'original') & (df['model'] == model) & (df['dataset'] == dataset)]
            row = df[(df['task_type'] == task) & (df['model'] == model) & (df['dataset'] == dataset)]
            original_val, original_test = original_row['val_score'].item(), original_row['test_score'].item()
            val, test = row['val_score'].item(), row['test_score'].item()
            if val < original_val:
                val_wins.append(dataset)
            elif val == original_val:
                val_ties.append(dataset)
            else:
                val_losses.append(dataset)
            if test < original_test:
                test_wins.append(dataset)
            elif test == original_test:
                test_ties.append(dataset)
            else:
                test_losses.append(dataset)
        print(f"{model} {task}: {val_losses} {test_losses}")
        val_winrate = (len(val_wins) + 0.5 * len(val_ties)) / (len(val_wins) + len(val_ties) + len(val_losses))
        test_winrate = (len(test_wins) + 0.5 * len(test_ties)) / (len(test_wins) + len(test_ties) + len(test_losses))
        model_winrate_dict[f'val_{task}'] = val_winrate
        model_winrate_dict[f'test_{task}'] = test_winrate
    model_winrate_dicts.append(model_winrate_dict)

winrate_df = pd.DataFrame(model_winrate_dicts).set_index('model')
winrate_df.to_csv('sanitycheck/perfect_prune/summarized_score.csv')

# summarize f1 score based on feature importance
# TODO: Group results by how many synthetic features were added
fi_info = []
for model in models:
    f1_info = {}
    for task in tasks:
        for dataset in datasets:
            original_row = df[(df['task_type'] == 'original') & (df['model'] == model) & (df['dataset'] == dataset)]
            row = df[(df['task_type'] == task) & (df['model'] == model) & (df['dataset'] == dataset)]

            features = ast.literal_eval(row.features.item())
            fi_mean = ast.literal_eval(row.fi_mean.item())
            fi_pval = ast.literal_eval(row.fi_pval.item())
            features_info = [el for el in zip(features, fi_mean, fi_pval)]
            k = len([feature for feature in features if 'noise_' not in feature])
            top_k_fi_mean = list(map(lambda e: e[0], sorted(features_info, key=lambda e: e[1])[::-1][:k]))
            top_k_fi_pval = list(map(lambda e: e[0], sorted(features_info, key=lambda e: e[2])[:k]))

            truth = [1 if 'noise_' not in feature else 0 for feature in features]
            pred_fi_mean = [1 if feature in top_k_fi_mean else 0 for feature in features]
            pred_fi_pval = [1 if feature in top_k_fi_pval else 0 for feature in features]
            mean_key, mean_score = f"mean_{task}", f1_score(truth, pred_fi_mean)
            # pval_key, pval_score = f"pval_{task}", f1_score(truth, pred_fi_pval)
            # mean_key, mean_score = dataset, f1_score(truth, pred_fi_mean)
            if mean_key in f1_info:
                f1_info[mean_key].append(mean_score)
                # f1_info[pval_key].append(pval_score)
            else:
                f1_info[mean_key] = [mean_score]
                # f1_info[pval_key] = [pval_score]
    # print(f1_info)
    f1_info = {key: np.mean(val) for key, val in f1_info.items()}
    fi_info.append({'model': model, **f1_info})

fi_df = pd.DataFrame(fi_info).set_index('model')
fi_df.to_csv('sanitycheck/perfect_prune/summarized_fi.csv')
print(fi_df)
