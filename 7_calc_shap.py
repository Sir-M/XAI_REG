import pickle
import torch
import numpy as np
import pandas as pd
import shap
from utils.fnn import fnn


def main():

    config_bzs = {
        "input_size": 7,
        "l1_size": 16,
        "l2_size": 8,
        "output_size": 1,
        "learning_rate": 0.001,
        "do": 0.2,
    }
    config_afrr = {
        "input_size": 13,
        "l1_size": 32,
        "l2_size": 16,
        "output_size": 1,
        "learning_rate": 0.001,
        "do": 0.2,
    }

    date = "2023-04-03"

    for change in ["afrr", "bzs"]:
        config = eval("config_" + change)
        for period in ["before", "after"]:
            for i in range(6):
                model = fnn(config)
                model.load_state_dict(
                    torch.load(f"model_data/{date}/{change}-{period}-{i}/model.pkl")
                )

                model.eval()

                with open(
                    f"model_data/{date}/{change}-{period}-{i}/X_data.pkl", "rb"
                ) as f:
                    X_data = pickle.load(f)

                def f(x):
                    return model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

                explainer = shap.KernelExplainer(f, shap.kmeans(X_data, 100))
                shap_values = explainer.shap_values(X_data)

                with open(
                    f"model_data/{date}/{change}-{period}-{i}/shap.pkl", "wb"
                ) as f:
                    pickle.dump(shap_values, f)

        for period in ["before", "after"]:
            for i in range(6):
                with open(
                    f"model_data/{date}/gbt-{change}-{period}-{i}/model.pkl", "rb"
                ) as f:
                    model = pickle.load(f)

                with open(
                    f"model_data/{date}/gbt-{change}-{period}-{i}/X_data.pkl", "rb"
                ) as f:
                    X_data = pickle.load(f)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_data)

                with open(
                    f"model_data/{date}/gbt-{change}-{period}-{i}/shap.pkl", "wb"
                ) as f:
                    pickle.dump(shap_values, f)


if __name__ == "__main__":
    main()
