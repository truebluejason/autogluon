"""
Runs on debug_info.csv
1. Show on how many datasets pruning was triggered
2. Compute mean ratio between fit time vs fi time
3. Show mean model score improvement from proxy vs no proxy
4. Show mean validation score increase per pruning iteration
"""
import ast
import pandas as pd

df = pd.read_csv('~/Downloads/debug_info(170).csv')
successful_rows = df[(df['pruned']) & (df['score_improvement_from_proxy_yes'] > 0)]
n_prune_successful = len(successful_rows)
n_prune_failed = len(df[(df['pruned']) & (df['score_improvement_from_proxy_yes'] == 0)])
n_prune_aborted = len(df[(~df['pruned']) & (df['total_prune_time'] > 0)])
n_prune_untriggered = len(df[(~df['pruned']) & (df['total_prune_time'] == 0)])
assert(len(df) == n_prune_successful + n_prune_failed + n_prune_aborted + n_prune_untriggered)
print("==================================================================")
print(f"Number of Successful/Failed/Aborted/Untriggered Prune: {n_prune_successful}/{n_prune_failed}/{n_prune_aborted}/{n_prune_untriggered}")

mean_fit_ratio = (successful_rows['total_prune_fit_time'] / successful_rows['total_prune_time']).mean()
mean_fi_ratio = (successful_rows['total_prune_fi_time'] / successful_rows['total_prune_time']).mean()
print("==================================================================")
print(f"Mean Fit/Feature Importance Time Ratio: {round(mean_fit_ratio, 4)} vs {round(mean_fi_ratio, 4)}")

num_improved = successful_rows['score_improvement_from_proxy_yes']
num_not_improved = successful_rows['score_improvement_from_proxy_no']
print("==================================================================")
print(f"Model Improvement Ratio From Pruning: {round((num_improved / (num_improved + num_not_improved)).mean(), 4)}")

index_trajectories = successful_rows['index_trajectory']
index_trajectories = [ast.literal_eval(trajectory) for trajectory in index_trajectories.tolist()]
max_len = max([len(trajectory) for trajectory in index_trajectories])
padded_trajectories = [trajectory + [False] * (max_len - len(trajectory)) for trajectory in index_trajectories]
trajectories_df = pd.DataFrame(padded_trajectories).replace(True, 1).replace(False, 0)
index_improvement_ratio = [round(val, 4) for val in (trajectories_df.sum(axis=0) / n_prune_successful).tolist()]
print("==================================================================")
print(f"Mean Validation Score Improvement Per Index: {index_improvement_ratio}")

mean_pruned_ratio = 1. - successful_rows['kept_ratio'].mean()
print("==================================================================")
print(f"Mean Ratio of Features Pruned: {mean_pruned_ratio}")

unique_exceptions = set(df['exceptions'])
print("==================================================================")
print(f"Unique Exceptions ({len(unique_exceptions)}): {unique_exceptions}")
print("==================================================================")
