import numpy as np
import pandas as pd
import sys

from result_stats import compare_dfs, add_dataset_info

args = sys.argv[1:]
assert len(args) == 2 or len(args) == 3, "first, second, and optionally third argument should be path to result files."


def task_score(df, task):
    if isinstance(df, pd.DataFrame):
        return round(df[df['task'] == task].result.dropna().mean(), 4)
    else:
        return [round(dataframe[dataframe['task'] == task].result.dropna().mean(), 4) for dataframe in df]


def loss_tie_win_rows(base, prune):
    loss_info, tie_info, win_info = compare_dfs(base, prune)
    loss_info = [('_'.join(info.split('_')[:-1]), int(info.split('_')[-1])) for info in loss_info]
    tie_info = [('_'.join(info.split('_')[:-1]), int(info.split('_')[-1])) for info in tie_info]
    win_info = [('_'.join(info.split('_')[:-1]), int(info.split('_')[-1])) for info in win_info]
    loss = pd.concat([prune[(prune['task'] == info[0]) & (prune['fold'] == info[1])] for info in loss_info])
    tie = pd.concat([prune[(prune['task'] == info[0]) & (prune['fold'] == info[1])] for info in tie_info])
    win = pd.concat([prune[(prune['task'] == info[0]) & (prune['fold'] == info[1])] for info in win_info])
    return loss, tie, win


task_metadata = pd.read_csv('result/task_metadata.csv')
base = add_dataset_info(pd.read_csv(args[0]), task_metadata)
prune1 = add_dataset_info(pd.read_csv(args[1]), task_metadata)
loss1, tie1, win1 = loss_tie_win_rows(base, prune1)

if len(args) == 3:
    prune2 = add_dataset_info(pd.read_csv(args[2]), task_metadata)
    loss2, tie2, win2 = loss_tie_win_rows(base, prune2)
    lossc, tiec, winc = loss_tie_win_rows(prune1, prune2)

import pdb; pdb.set_trace()
"""
Plot 2D histogram of win, loss, and tie count of features vs sample size
Plot 2D histogram of difference in win count between two different configuration when comparing with baseline

plot tasks in a sorted manner from highest winrate among seed to lowest
"""
