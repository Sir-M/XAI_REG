import pickle
from utils.model_fit import model_fit


def main():
    config = {
        "input_size": 13,
        "l1_size": 32,
        "l2_size": 16,
        "output_size": 1,
        "learning_rate": 0.001,
        "do": 0.2,
    }

    for period in ["before", "after"]:

        with open("_data/afrr_{}.pkl".format(period), "rb") as f:
            data = pickle.load(f)

        for i in range(6):
            model_fit(
                "FNN",
                data[i],
                "neg_avg_auction_price",
                "afrr-{}-{}".format(period, i),
                config,
            )

    config = {
        "input_size": 7,
        "l1_size": 16,
        "l2_size": 8,
        "output_size": 1,
        "learning_rate": 0.001,
        "do": 0.2,
    }

    for period in ["before", "after"]:

        with open("_data/bzs_{}.pkl".format(period), "rb") as f:
            data = pickle.load(f)

        for i in range(6):
            model_fit(
                "FNN",
                data[i],
                "price_day_ahead_AT",
                "bzs-{}-{}".format(period, i),
                config,
            )

    for period in ["before", "after"]:

        with open("_data/afrr_{}.pkl".format(period), "rb") as f:
            data = pickle.load(f)

        for i in range(6):
            model_fit(
                "GBT",
                data[i],
                "neg_avg_auction_price",
                "gbt-afrr-{}-{}".format(period, i),
            )

    for period in ["before", "after"]:

        with open("_data/bzs_{}.pkl".format(period), "rb") as f:
            data = pickle.load(f)

        for i in range(6):
            model_fit(
                "GBT", data[i], "price_day_ahead_AT", "gbt-bzs-{}-{}".format(period, i)
            )


if __name__ == "__main__":
    main()
