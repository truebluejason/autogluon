import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.models import CatBoostModel, KNNModel, LGBModel, XGBoostModel, TabularNeuralNetModel, RFModel
import pandas as pd

"""
TODO
1. Update quip with prune trigger stats
2. Upload all the scripts to a branch
3. Check LGB results
"""


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--train_path', help='train dataset path', type=str, default='dataset/australian/train_data.csv')
parser.add_argument('-g', '--test_path', help='test dataset path', type=str, default='dataset/australian/test_data.csv')
parser.add_argument('-a', '--label_path', help='optional test set label dataset path', type=str, default=None)
parser.add_argument('-m', '--mode', help='what AutoGluon setting to try', choices=['model', 'model-high', 'model-stack', 'ag', 'ag-stack'], default='model')
parser.add_argument('-n', '--num_resource', help='number of resource to allocate across all features', type=int, default=None)
parser.add_argument('-l', '--label', help='name of the label column', type=str, default='class')
parser.add_argument('-p', '--prune', help='to use fit_with_prune or not', dest='prune', action='store_true')
parser.add_argument('-r', '--ratio', help='what percentage of features to prune at once', type=float, default=0.05)
args = parser.parse_args()

train_data = pd.read_csv(args.train_path).head(500000)
if args.label_path is None:
    test_data = pd.read_csv(args.test_path)
    X_test = test_data.drop(columns=[args.label])
    y_test = test_data[args.label]
else:
    X_test = pd.read_csv(args.test_path)
    y_test = pd.read_csv(args.label_path)
    test_data = pd.merge(X_test, y_test, on='ID')
    y_test = y_test[args.label]

feature_prune_kwargs = {
    'stopping_round': 10,
    'prune_ratio': args.ratio,
    'prune_threshold': 'noise',
    'n_train_subsample': None,
    'n_fi_subsample': 10000,
    #'proxy_model_class': CatBoostModel,
    'max_fits': 10,
    # 'k': 1000,
    'min_fi_samples': 10000,
    'max_fi_samples': 50000,
    # 'weighted': False,
    # 'replace_bag': False,
    # 'force_prune': True
    # 'n_evaluated_features': 12
}
feature_prune_kwargs = {'k': 1000}

time_limit = 3600
if args.mode == 'model':
    presets = ['medium_quality_faster_train']
    custom_hyperparameters = {LGBModel: {}, CatBoostModel: {}}
    extra_args = {}
elif args.mode == 'model-high':
    presets = ['high_quality_fast_inference_only_refit']
    custom_hyperparameters = {LGBModel: {}}
    extra_args = {'num_bag_sets': 2, 'num_stack_levels': 1}
elif args.mode == 'model-stack':
    presets = ['best_quality']
    # custom_hyperparameters = {KNNModel: {'ag_args': fit_with_prune_kwargs}}
    # custom_hyperparameters = {KNNModel: {}, CatBoostModel: {}, LGBModel: {}, XGBoostModel: {}}
    # custom_hyperparameters = {LGBModel: {}, CatBoostModel: {}}
    # extra_args = {'num_bag_sets': 2, 'num_stack_levels': 1}
    custom_hyperparameters = {"GBM": {}}  #{LGBModel: {}}
    extra_args = {'num_bag_sets': 1, 'num_stack_levels': 0}
elif args.mode == 'ag':
    presets = ['medium_quality_faster_train']
    custom_hyperparameters = None
    extra_args = {}
else:
    presets = ['best_quality']
    custom_hyperparameters = None
    extra_args = {'num_bag_sets': 2, 'num_stack_levels': 1}  # {'num_bag_sets': 2, 'num_stack_levels': 1}

predictor = TabularPredictor(label=args.label)
try:
    if args.prune:
        predictor = predictor.fit(train_data, presets=presets, feature_prune_kwargs=feature_prune_kwargs, time_limit=time_limit, **extra_args,
                                  hyperparameters=custom_hyperparameters)
    else:
        predictor = predictor.fit(train_data, presets=presets, **extra_args,
                                  time_limit=time_limit, hyperparameters=custom_hyperparameters)
except Exception as e:
    import pdb
    pdb.post_mortem()

try:
    y_pred = predictor.predict(test_data)
    performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(performance)
    print("=============== Validation Leaderboard ===============")
    predictor.leaderboard()
    print("=============== Test Leaderboard ===============")
    predictor.leaderboard(test_data)
except Exception as e:
    import pdb
    pdb.post_mortem()
    print(e)
