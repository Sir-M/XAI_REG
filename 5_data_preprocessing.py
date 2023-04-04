import pandas as pd
import sys
import os
from scipy.stats.mstats import winsorize
import pickle

sys.path.append("../")

if not os.path.exists("data/"):
    os.makedirs("data/")


def get_window(df, n, p):
    windows = []
    l = len(df)
    start = [int(i * (p * l / (n / 2))) for i in range(n)]
    end = [(int(i * (p * l / (n / 2))) + int(0.5 * l)) % l for i in range(n)]
    for i in range(n):
        if start[i] < end[i]:
            windows.append(df.iloc[start[i] : end[i]])
        else:
            windows.append(pd.concat([df.iloc[start[i] :], df.iloc[: end[i]]]))
    return windows


def main():
    number_of_models = 6
    percent_of_data = 0.5

    ########### aFRR PRICE PREPROCESSING ###########

    X_columns = [
        "gen_biomass_DE",
        "gen_gas_DE",
        "gen_hard_coal_DE",
        "gen_lignite_DE",
        "gen_nuclear_DE",
        "gen_pumped_hydro_DE",
        "gen_reservoir_hydro_DE",
        "gen_run_off_hydro_DE",
        "gen_solar_DE",
        "gen_wind_DE",
        "load_DE",
        "weekday",
        "hour",
    ]

    target = "neg_avg_auction_price"

    afrr_price_data = (
        pd.read_pickle("../_data/regelleistungnet/neg_afrr_price_data.pkl")
        .astype("float")
        .sort_index()["2018-10-16":"2020-07-30"]
    )

    afrr_price_data["neg_factor"] = 0
    afrr_price_data["neg_factor"].loc["2018-10-16":"2018-12-31"] = 0.0607
    afrr_price_data["neg_factor"].loc["2019-1-1":"2019-3-31"] = 0.0675
    afrr_price_data["neg_factor"].loc["2019-4-1":"2018-6-30"] = 0.078
    afrr_price_data["neg_factor"].loc["2019-7-1":"2019-7-31"] = 0.0841

    afrr_price_data["neg_avg_auction_price"] = (
        afrr_price_data["neg_avg_cap_price"]
        - afrr_price_data["neg_avg_en_price"] * afrr_price_data["neg_factor"]
    )

    entsoedata = (
        pd.read_pickle("../_data/entsoe/entsoe_DE_AT_cty_engineered_1h")
        .astype("float")
        .tz_convert("CET")
        .sort_index()["2017-10-16":"2020-07-30"]
        .tz_convert(None)
    )

    # combine pumped hydro generation and consumption
    entsoedata["gen_pumped_hydro_DE"] = entsoedata.gen_pumped_hydro_DE.add(
        -entsoedata.pumped_hydro_consumption_DE, fill_value=0
    )
    entsoedata.drop(["pumped_hydro_consumption_DE"], axis=1, inplace=True)

    # combine wind to one timeseries
    entsoedata["gen_wind_DE"] = entsoedata.gen_wind_on_DE.add(
        entsoedata.gen_wind_off_DE, fill_value=0
    )
    entsoedata.drop(["gen_wind_on_DE", "gen_wind_off_DE"], axis=1, inplace=True)

    # resample entsoe data to 4h
    entsoedata = entsoedata.resample("4H").mean()

    entsoedata.weekday = entsoedata.index.weekday
    entsoedata.hour = entsoedata.index.hour

    data = pd.concat([entsoedata[X_columns], afrr_price_data[target]], axis=1).dropna()

    data_1 = data["2018-10-16":"2019-07-30"]
    data_2 = data["2019-07-31":"2020-07-30"]

    data_1[target] = winsorize(data_1[target], limits=(0, 0.1))
    data_2[target] = winsorize(data_2[target], limits=(0, 0.1))

    with open("data/afrr_before.pkl", "wb") as f:
        pickle.dump(get_window(data_1, number_of_models, percent_of_data), f)

    with open("data/afrr_after.pkl", "wb") as f:
        pickle.dump(get_window(data_2, number_of_models, percent_of_data), f)

    ########### BIDDING ZONE SPLIT PREPROCESSING ###########

    X_columns = ["DE", "AT", "CZ", "HU", "IT", "SI", "CH"]

    price_AT = pd.read_pickle(
        "../_data/entsoe/entsoe_DE_AT_cty_engineered_1h"
    ).price_day_ahead_AT
    res_loads = pd.read_pickle("../_data/entsoe/residual_loads_hydro")

    data = pd.concat([price_AT, res_loads[X_columns]], axis=1)[
        "2017-10-1":"2019-09-30"
    ].dropna()

    data_1 = data[:"2018-09-30"]
    data_2 = data["2018-10-1":]

    with open("data/bzs_before.pkl", "wb") as f:
        pickle.dump(get_window(data_1, number_of_models, percent_of_data), f)

    with open("data/bzs_after.pkl", "wb") as f:
        pickle.dump(get_window(data_2, number_of_models, percent_of_data), f)


if __name__ == "__main__":
    main()
