import optuna
import lightgbm as lgb


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


def objective(trial):
    """Train Light GBM model"""
    x_train, x_val, y_train, y_val, group_qids_train, group_qids_val = function_to_get_data()

    ranking_param_grid = {
        'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.14, step=0.02),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
        'num_leaves': trial.suggest_int('num_leaves', 25, 35, 1),
        'max_depth': trial.suggest_int('max_depth', -1, 500, 50)
    }

    gbm = lgb.LGBMRanker(n_jobs=24, **ranking_param_grid)
    gbm.fit(x_train, y_train, group=group_qids_train, eval_set=[(x_val, y_val)],
        eval_metric='ndcg', eval_group=[group_qids_val], eval_at=[10, 50, 100],
        early_stopping_rounds=100, verbose=False
    )

    return list(gbm.best_score_['valid_0'].values())[2]


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    best_params = study.best_params
    best_trial = study.best_trial

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', best_trial.params)
    print('Best params:', best_params)
