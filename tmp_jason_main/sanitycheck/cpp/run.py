import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.models import CatBoostModel, KNNModel, LGBModel, XGBoostModel, TabularNeuralNetModel, RFModel
import os
from numpy.core.fromnumeric import trace
import pandas as pd
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_dir', help='path to cpp directory', type=str, default='dataset/cpp')
parser.add_argument('-p', '--problem', help='only run this problem if specified', type=str, default=None)
parser.add_argument('-r', '--result_path', help='file to save test set score to', type=str, default='sanitycheck/cpp/result.csv')
parser.add_argument('-m', '--mode', help='what AutoGluon setting to try', choices=['ag', 'ag-stack'], default='ag-stack')
parser.add_argument('-t', '--time_limit', help='time limit in minutes', type=int, default=60)
args = parser.parse_args()

# DATASETS = [
#     "1db99236-0601-4e03-b8bb-96b5eb236d74",
#     "20e6e8e8-a4da-4fea-a9de-c784cdf84c1f",
#     "2cbd9a22-0da1-404d-a7ba-49911840a622",
#     "3cf28e5f-886a-4ace-bebf-299e1dbde654",
#     "4dbb8031-56a6-43bf-9e03-40ea2affa163",
#     "5729f07d-8d43-463d-894b-7dfa2da63efb",
#     "5d1e3461-8b01-463c-a9db-2e4c48db1467",
#     "60c60200-2341-427d-b0ec-2fc30c4bfdd8",
# ]
TIME_LIMIT = args.time_limit * 60.
RESULT_PATH = args.result_path
EXCEPTIONS_PATH = os.path.join(os.path.dirname(args.result_path), 'exceptions.csv')
if args.problem is None:
    DATASETS = sorted([dataset for dataset in os.listdir(args.dataset_dir) if not dataset.startswith('.')])[1:]
else:
    DATASETS = [args.problem]
FEATURE_PRUNE_KWARGS = {}


def add_datapoint(result: dict, dataset: str, mode: str, val_score: float, test_score: float, time_limit: float, n_sample: int, n_feature: int):
    result['dataset'].append(dataset)
    result['mode'].append(mode)
    result['val_score'].append(round(val_score, 4))
    result['test_score'].append(round(test_score, 4))
    result['time_limit'].append(round(time_limit, 4))
    result['n_sample'].append(n_sample)
    result['n_feature'].append(n_feature)


def add_exception(exception: dict, dataset: str, type: str, error_str: str, stacktrace: str):
    exception['dataset'].append(dataset)
    exception['type'].append(type)
    exception['error_str'].append(error_str)
    exception['stacktrace'].append(stacktrace)


for dataset in DATASETS:
    train_data = pd.read_csv(os.path.join(args.dataset_dir, dataset, 'train.csv'))
    test_data = pd.merge(pd.read_csv(os.path.join(args.dataset_dir, dataset, 'testFeaturesNoLabel.csv')),
                         pd.read_csv(os.path.join(args.dataset_dir, dataset, 'testLabel.csv')), on='ID')
    y_test = test_data['label']
    presets = ['medium_quality_faster_train'] if args.mode == 'ag' else ['best_quality']
    n_sample, n_feature = len(train_data), len(train_data.columns) - 1

    result = {'dataset': [], 'mode': [], 'val_score': [], 'test_score': [], 'time_limit': [], 'n_sample': [], 'n_feature': []}
    exception = {'dataset': [], 'type': [], 'error_str': [], 'stacktrace': []}
    try:
        predictor = TabularPredictor(label='label', eval_metric='roc_auc')
        predictor = predictor.fit(train_data, presets=presets, time_limit=TIME_LIMIT, ag_args_fit=dict(num_cpu=8))
        leaderboard = predictor.leaderboard(test_data)
        best_val_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
        val_score, test_score = best_val_row['score_val'], best_val_row['score_test']
        add_datapoint(result, dataset, presets[0], val_score, test_score, TIME_LIMIT, n_sample, n_feature)
    except Exception as e:
        add_exception(exception, dataset, presets[0], str(e), traceback.format_exc())

    try:
        predictor = TabularPredictor(label='label', eval_metric='roc_auc')
        predictor = predictor.fit(train_data, presets=presets, time_limit=TIME_LIMIT, ag_args_fit=dict(num_cpu=8), feature_prune_kwargs=FEATURE_PRUNE_KWARGS)
        leaderboard = predictor.leaderboard(test_data)
        best_val_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
        val_score, test_score = best_val_row['score_val'], best_val_row['score_test']
        add_datapoint(result, dataset, presets[0] + "_prune", val_score, test_score, TIME_LIMIT, n_sample, n_feature)
    except Exception as e:
        add_exception(exception, dataset, presets[0] + "_prune", str(e), traceback.format_exc())

    result_df = pd.DataFrame(result)
    if os.path.exists(RESULT_PATH):
        original_result_df = pd.read_csv(RESULT_PATH)
        result_df = pd.concat([original_result_df, result_df], axis=0)
    result_df.to_csv(RESULT_PATH, index=False)

    exception_df = pd.DataFrame(exception)
    if os.path.exists(EXCEPTIONS_PATH):
        original_exception_df = pd.read_csv(EXCEPTIONS_PATH)
        exception_df = pd.concat([original_exception_df, exception_df], axis=0)
    exception_df.to_csv(EXCEPTIONS_PATH, index=False)
