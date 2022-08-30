import optuna
import torch

from informer.main_informer import Exp
from utils.tools import get_default_args

# ---------------------------------


def train_and_test_model(model, args):
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                                                                                                             args.data,
                                                                                                             args.features,
                                                                                                             args.seq_len,
                                                                                                             args.label_len,
                                                                                                             args.pred_len,
                                                                                                             args.d_model,
                                                                                                             args.n_heads,
                                                                                                             args.e_layers,
                                                                                                             args.d_layers,
                                                                                                             args.d_ff,
                                                                                                             args.attn,
                                                                                                             args.factor,
                                                                                                             args.embed,
                                                                                                             args.distil,
                                                                                                             args.mix,
                                                                                                             args.des,
                                                                                                             ii)

        # set experiments
        exp = model(args, k_fold=5)

        # train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        # print(data_dict)'
        exp.train(setting)

        # test
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, mse, rmse, mape, mspe = exp.test(setting)

        torch.cuda.empty_cache()

        return mse


def objective(trial):
    args = get_default_args()
    args.features = 'S'  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.freq = 'b'  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.learning_rate = 0.0001

    params = {
        # 'features': trial.suggest_categorical("features", ['S', 'M', 'MS']),
        'freq': trial.suggest_categorical("freq", ['b', 'd']),
        'seq_len': trial.suggest_categorical("seq_len", [10, 20, 40, 60, 80]),
        'label_len': trial.suggest_categorical("label_len", [5, 10, 20, 30, 40]),
        'pred_len': trial.suggest_categorical("pred_len", [2, 5, 10, 15, 20]),
        'd_model': trial.suggest_categorical("d_model", [256, 512, 1024]),
        'n_heads': trial.suggest_categorical("n_heads", [4, 8, 16]),
        'd_ff': trial.suggest_categorical("d_ff", [512, 1024, 2048]),
        'e_layers': trial.suggest_categorical("e_layers", [1, 2, 3]),
        'd_layers': trial.suggest_categorical("d_layers", [1, 2, 3]),
        'train_epochs': trial.suggest_categorical("train_epochs", [6, 10, 20, 40, 60]),
        'patience': trial.suggest_categorical("patience", [2, 3, 5]),
        'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
        'distil': trial.suggest_categorical("distil", [True, False]),
    }

    args.update(params)

    exp = Exp
    mse = train_and_test_model(exp, atgs)

    return mse


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)
