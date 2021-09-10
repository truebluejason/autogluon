"""
1. See what percentage of models have successfully been pruned
2. For each task, determine whether pruned vs unpruned models had lower test error
3. For each task, determine whether pruned vs unpruned models had lower generalization error
"""

import numpy as np
import pandas as pd

# full results
# df_original = pd.read_csv('result/baseline/4h/results_ag_leaderboard_4h8c_autogluon_bestquality_nochildoof.ag.4h8c.aws.20210813T013449.csv')
# df = pd.read_csv('result/best/4h/results_ag_leaderboard_4h8c_autogluon_autoprune_power.ag.4h8c.aws.20210813T161310.csv')
# df = pd.read_csv('~/Downloads/results_ag_leaderboard_4h8c_autogluon_autoprune_power.ag.4h8c.aws.20210820T041916.csv')
# df = pd.read_csv('result/best/4h/results_ag_leaderboard_4h8c_autogluon_autoprune_power.ag.4h8c.aws.20210814T072519.csv')

# 4hr results
# df_original = pd.read_csv('result/baseline/4h/results_ag_leaderboard_4h8c_autogluon_bestquality.ag.4h8c.aws.20210816T211408.csv')
# df = pd.read_csv('~/Downloads/results_ag_leaderboard_4h8c_prune.ag.4h8c.aws.20210822T053718.csv')

# 1hr results
# df_original = pd.read_csv('result/baseline/1h/results_ag_leaderboard_1h8c_autogluon_bestquality.ag.1h8c.aws.20210816T211413.csv')
# df = pd.read_csv('~/Downloads/results_ag_leaderboard_1h8c_prune.ag.1h8c.aws.20210822T053721.csv')
# df = pd.read_csv('~/Downloads/results_ag_leaderboard_1h8c_.csv')

# norepeat results
# df_original = pd.read_csv('result/baseline/4hnorepeat/results_ag_leaderboard_4h8c_autogluon_bestquality_norepeat.ag.4h8c.aws.20210813T013445.csv')
# df = pd.read_csv('result/best/4hnorepeat/results_ag_leaderboard_4h8c_autogluon_autoprune_power_norepeat.ag.4h8c.aws.20210813T013500.csv')


def add_dataset_info(results_raw: pd.DataFrame, task_metadata: pd.DataFrame):
    results_raw['tid'] = [int(x.split('/')[-1]) for x in results_raw['id']]
    task_metadata['ClassRatio'] = task_metadata['MinorityClassSize'] / task_metadata['NumberOfInstances']
    results_raw = results_raw.merge(task_metadata, on=['tid'])
    return results_raw


def filter_samples(df, samples=1000, lower=True):
    return df[df['NumberOfInstances'] < samples] if lower else df[df['NumberOfInstances'] >= samples]


task_metadata = pd.read_csv('result/task_metadata.csv')
df_original = filter_samples(add_dataset_info(df_original, task_metadata), samples=10000, lower=False)
df = filter_samples(add_dataset_info(df, task_metadata), samples=10000, lower=False)

SEARCH_KEYWORD = '_Prune'
print("==================================================================")
print(f"Percentage of Pruned Models: {round(len(df[df['model'].str.contains(SEARCH_KEYWORD)])/len(df), 3)}")

pruned_best_win, unpruned_best_win, best_tie = 0, 0, 0
pruned_test_win, unpruned_test_win, test_tie = 0, 0, 0
pruned_gen_win, unpruned_gen_win, gen_tie = 0, 0, 0

pruned_auc_gen, unpruned_auc_gen = [], []
pruned_nll_gen, unpruned_nll_gen = [], []
pruned_nrmse_gen, unpruned_nrmse_gen = [], []

pruned_best_counts = {}
unpruned_best_counts = {}
for task in df['task'].unique():
    rows = df[df['task'] == task]
    original_rows = df_original[df_original['task'] == task]
    pruned_all = rows[rows['model'].str.contains(SEARCH_KEYWORD)]
    unpruned_all = original_rows[original_rows['model'].isin([model[:len(model)-6] for model in pruned_all['model'].tolist()])]

    for fold in rows['fold'].unique():
        # compare selected model's test score (highest val)
        pruned_run_rows = rows[rows['fold'] == fold]
        unpruned_run_rows = original_rows[original_rows['fold'] == fold]

        # # layer = 1
        # # pruned_run_rows = pruned_run_rows[((pruned_run_rows['model'].str.contains(f'L{layer}'))
        # #                                   & ~(pruned_run_rows['model'] == f'WeightedEnsemble_L{layer}'))
        # #                                   | (pruned_run_rows['model'] == f'WeightedEnsemble_L{layer+1}')]
        # # unpruned_run_rows = unpruned_run_rows[((unpruned_run_rows['model'].str.contains(f'L{layer}'))
        # #                                       & ~(unpruned_run_rows['model'] == f'WeightedEnsemble_L{layer}'))
        # #                                       | (unpruned_run_rows['model'] == f'WeightedEnsemble_L{layer+1}')]

        # if len(pruned_run_rows) > 0 and len(unpruned_run_rows) > 0:
        #     pruned_best_row = pruned_run_rows.loc[pruned_run_rows['score_val'].idxmax()]
        #     unpruned_best_row = unpruned_run_rows.loc[unpruned_run_rows['score_val'].idxmax()]
        #     pruned_best_test = pruned_best_row.score_test
        #     unpruned_best_test = unpruned_best_row.score_test
        #     if pruned_best_test > unpruned_best_test:
        #         pruned_best_win += 1
        #     elif pruned_best_test < unpruned_best_test:
        #         unpruned_best_win += 1
        #     else:
        #         best_tie += 1
        #     pruned_best_counts[pruned_best_row.model] = pruned_best_counts.get(pruned_best_row.model, 0) + 1
        #     unpruned_best_counts[unpruned_best_row.model] = unpruned_best_counts.get(unpruned_best_row.model, 0) + 1

    for fold in pruned_all['fold'].unique():
        # compare selected model's test score (highest val)
        pruned_run_rows = rows[rows['fold'] == fold]
        unpruned_run_rows = original_rows[original_rows['fold'] == fold]

        # layer = 1
        # pruned_run_rows = pruned_run_rows[((pruned_run_rows['model'].str.contains(f'L{layer}'))
        #                                   & ~(pruned_run_rows['model'] == f'WeightedEnsemble_L{layer}'))
        #                                   | (pruned_run_rows['model'] == f'WeightedEnsemble_L{layer+1}')]
        # unpruned_run_rows = unpruned_run_rows[((unpruned_run_rows['model'].str.contains(f'L{layer}'))
        #                                       & ~(unpruned_run_rows['model'] == f'WeightedEnsemble_L{layer}'))
        #                                       | (unpruned_run_rows['model'] == f'WeightedEnsemble_L{layer+1}')]

        if len(pruned_run_rows) > 0 and len(unpruned_run_rows) > 0:
            pruned_best_row = pruned_run_rows.loc[pruned_run_rows['score_val'].idxmax()]
            unpruned_best_row = unpruned_run_rows.loc[unpruned_run_rows['score_val'].idxmax()]
            pruned_best_test = pruned_best_row.score_test
            unpruned_best_test = unpruned_best_row.score_test
            if pruned_best_test > unpruned_best_test:
                pruned_best_win += 1
            elif pruned_best_test < unpruned_best_test:
                unpruned_best_win += 1
            else:
                best_tie += 1
            pruned_best_counts[pruned_best_row.model] = pruned_best_counts.get(pruned_best_row.model, 0) + 1
            unpruned_best_counts[unpruned_best_row.model] = unpruned_best_counts.get(unpruned_best_row.model, 0) + 1

        pruned_fold = pruned_all[pruned_all['fold'] == fold]
        unpruned_fold = unpruned_all[unpruned_all['fold'] == fold]
        for model in pruned_fold['model'].unique():
            pruned = pruned_fold[pruned_fold['model'] == model]
            unpruned = unpruned_fold[unpruned_fold['model'] == model[:len(model)-6]]
            if len(pruned) == 0 or len(unpruned) == 0:
                continue

            # record test error wins
            pruned_test = pruned['score_test'].item()
            unpruned_test = unpruned['score_test'].item()
            if pruned_test > unpruned_test:
                pruned_test_win += 1
            elif pruned_test < unpruned_test:
                unpruned_test_win += 1
            else:
                test_tie += 1

            # record generalization error wins
            pruned_gen = pruned['score_val'].item() - pruned_test
            unpruned_gen = unpruned['score_val'].item() - unpruned_test
            if pruned_gen < unpruned_gen:
                pruned_gen_win += 1
            elif pruned_gen > unpruned_gen:
                unpruned_gen_win += 1
            else:
                gen_tie += 1
            metric = rows.iloc[0]['metric']

            # log generalization error
            if metric == 'auc':
                pruned_auc_gen.append(pruned_gen)
                unpruned_auc_gen.append(unpruned_gen)
            elif metric == 'neg_logloss':
                pruned_nll_gen.append(pruned_gen)
                unpruned_nll_gen.append(unpruned_gen)
            else:
                pruned_nrmse_gen.append(pruned_gen)
                unpruned_nrmse_gen.append(unpruned_gen)

print("==================================================================")
print(f"Best Model Test Score Wins (Pruned/Unpruned/Tie): {pruned_best_win}/{unpruned_best_win}/{best_tie}")
print(f"Number of Test Score Wins Per Task (Pruned/Unpruned/Tie): {pruned_test_win}/{unpruned_test_win}/{test_tie}")
print(f"Number of Generalization Error Wins Per Task (Pruned/Unpruned/Tie): {pruned_gen_win}/{unpruned_gen_win}/{gen_tie}")
# print(f"Mean Binary Generalization Error (Pruned/Unpruned): {round(np.mean(pruned_auc_gen),4)}/{round(np.mean(unpruned_auc_gen),4)}")
# print(f"Mean Multiclass Generalization Error (Pruned/Unpruned): {round(np.mean(pruned_nll_gen),4)}/{round(np.mean(unpruned_nll_gen),4)}")
# print(f"Mean Regression Generalization Error (Pruned/Unpruned): {round(np.mean(pruned_nrmse_gen),4)}/{round(np.mean(unpruned_nrmse_gen),4)}")
# print(f"PRUNED BEST COUNT: {pruned_best_counts}")
# print(f"UNPRUNED BEST COUNT: {unpruned_best_counts}")
print("==================================================================")
