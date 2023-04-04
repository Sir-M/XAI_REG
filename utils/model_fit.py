import os
import datetime
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from lightgbm import LGBMRegressor

import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import pickle

from utils.fnn import fnn


def model_fit(model_type, data, target, modelname, config=None):
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists("model_data/{}/{}".format(date, modelname)):
        os.makedirs("model_data/{}/{}".format(date, modelname))

    block_size = "4D"

    masker = [
        pd.Series(g.index) for n, g in data.groupby(pd.Grouper(freq=block_size))
    ]
    train_mask, test_mask = train_test_split(
        masker, test_size=0.2, random_state=42
    )

    if model_type == "FNN":
        train_mask, val_mask = train_test_split(train_mask, test_size=0.2, random_state=42)

        train = data.loc[pd.concat(train_mask)]
        val = data.loc[pd.concat(val_mask)]
        test = data.loc[pd.concat(test_mask)]

        X_train = train.drop(target, axis=1)
        y_train = train[target]
        X_test = test.drop(target, axis=1)
        y_test = test[target]
        X_val = val.drop(target, axis=1)
        y_val = val[target]

        X_scaler = PowerTransformer()
        y_scaler = MinMaxScaler()

        X_train = X_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

        X_val = X_scaler.transform(X_val)
        y_val = y_scaler.transform(y_val.values.reshape(-1, 1))
        X_test = X_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

        X_data = X_scaler.transform(data.drop(target, axis=1))

        for dataset in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "X_data"]:
            with open("model_data/{}/{}/{}.pkl".format(date, modelname, dataset), 'wb') as f:
                pickle.dump(eval(dataset), f)


        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(
            dataset, num_workers=10, batch_size=128, shuffle=True
        )

        inputs_val = torch.tensor(X_val, dtype=torch.float32)
        labels_val = torch.tensor(y_val, dtype=torch.float32)

        dataset_val = TensorDataset(inputs_val, labels_val)
        dataloader_val = DataLoader(dataset_val, num_workers=10, batch_size=64)

        logger = TensorBoardLogger("model_logs", name='{}'.format(modelname))

        model = fnn(config)
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, mode="min")
        trainer = pl.Trainer(max_epochs=700, callbacks=[early_stop_callback], logger=logger, enable_progress_bar=False)
        lr_find_results = trainer.tuner.lr_find(
            model, dataloader, min_lr=0.001, max_lr=1
        )
        print(lr_find_results.suggestion())
        if lr_find_results.suggestion() is None:
            new_lr = 0.001
        else:
            new_lr = lr_find_results.suggestion()
        model.learning_rate = new_lr

        trainer.fit(model, dataloader, dataloader_val)

        torch.save(model.state_dict(), "model_data/{}/{}/model.pkl".format(date, modelname))



    elif model_type == "GBT":
        train = data.loc[pd.concat(train_mask)]
        test = data.loc[pd.concat(test_mask)]

        X_train = train.drop(target, axis=1)
        y_train = train[target]
        X_test = test.drop(target, axis=1)
        y_test = test[target]


        X_data = data.drop(target, axis=1)

        for dataset in ["X_train", "X_test", "y_train", "y_test", "X_data"]:
            with open("model_data/{}/{}/{}.pkl".format(date, modelname, dataset), 'wb') as f:
                pickle.dump(eval(dataset), f)

        param_distributions = {
            'learning_rate': sp_uniform(0.01, 0.1),
            'n_estimators': sp_randint(10, 40),
            'max_depth': sp_randint(3, 7),
            'num_leaves': sp_randint(10, 100),
            'min_child_samples': sp_randint(20, 50),
            'subsample': sp_uniform(0.5, 0.5),
            'colsample_bytree': sp_uniform(0.5, 0.5),
            'reg_alpha': sp_uniform(0, 1),
            'reg_lambda': sp_uniform(0, 1),
        }

        lgbm = LGBMRegressor(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_distributions,
            n_iter=1000,
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X_train, y_train)

        model = random_search.best_estimator_

        with open("model_data/{}/{}/model.pkl".format(date, modelname),'wb') as f:
            pickle.dump(model, f)