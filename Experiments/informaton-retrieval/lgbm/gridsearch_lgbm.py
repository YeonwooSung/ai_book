import lightgbm as lgb
import pandas as pd
import itertools
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from functools import partial


def function_to_get_data(csv_file):
    # Define your own get data function
    # refer to this example for data https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank 
    x_train = None
    x_val = None
    y_train = None 
    y_val = None
    
    # we need group because we need to rank a list of item not like classification or regression
    group_qids_train = None 
    group_qids_val = None
    return x_train, x_val, y_train, y_val, group_qids_train, group_qids_val


def get_experience(x_train, x_val, y_train, y_val, group_qids_train, group_qids_val, ranking_param_grid):
    params = {
        'learning_rate': ranking_param_grid[0],
        'n_estimators': ranking_param_grid[1],
        'num_leaves': ranking_param_grid[2],
        'max_depth': ranking_param_grid[3],
        'boosting_type': ranking_param_grid[4]
    }
    gbm = lgb.LGBMRanker(n_jobs=24, **params)
    
    # eval_set = [10, 50, 100] this is the define NDCG@10, NDCG@50, NDCG@100
    gbm.fit(x_train, y_train, group=group_qids_train, eval_set=[(x_val, y_val)],
            eval_metric='ndcg', eval_group=[group_qids_val], eval_at=[10, 50, 100],
            early_stopping_rounds=100, verbose=False)
        
    return gbm


def fit_lgbm_mp(ranking_param_grid: dict):
    """Train Light GBM model"""
    x_train, x_val, y_train, y_val, group_qids_train, group_qids_val = function_to_get_data()

    the_best_configs = {}
    len_ = 1

    for k in ranking_param_grid.keys():
        len_ *= len(ranking_param_grid[k])
        
    list_params = itertools.product(
        ranking_param_grid['learning_rate'],
        ranking_param_grid['n_estimators'],
        ranking_param_grid['num_leaves'],
        ranking_param_grid['max_depth'],
        ranking_param_grid['boosting_type']
    )

    # Multi processing
    process_pool = mp.Pool(processes=32)
    get_experience_each = partial(
        get_experience,
        x_train,
        x_val,
        y_train,
        y_val,
        group_qids_train,
        group_qids_val
    )
    with tqdm(total=len_) as pbar:
        for each_item in process_pool.imap(
            get_experience_each,
            list_params
        ):
            pbar.update()
            the_best_configs[list(each_item.best_score_['valid_0'].values())[
                2]] = each_item.get_params()

    df = pd.DataFrame(the_best_configs.items(), columns=['scores', 'params'])
    best_params = df[df.scores == max(df.scores)].params.to_list()
    print(best_params)


def fit_lgbm_sample():
    ranking_param_grid = {
        'learning_rate': np.arange(0.06, 0.14, 0.02),
        'n_estimators': np.arange(100, 500, 100),
        'num_leaves': range(25, 35, 1),
        'max_depth': [-1, 50, 100, 200],
        'boosting_type': ['gbdt', 'goss']
    }
    fit_lgbm_mp(ranking_param_grid)


if __name__ == "__main__":
    fit_lgbm_sample()
