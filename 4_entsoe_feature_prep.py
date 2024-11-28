import pandas as pd
import numpy as np
import holidays


def rename_ramp_columns(s: str, region: str = None) -> str:
    """
    renames columns of ramp features
    """
    if "day_ahead" in s:
        ind = s.find("day_ahead")
        s = s[:ind] + "ramp_" + s[ind:]
    elif region:
        ind = s.find(region)
        s = s[:ind] + "ramp_" + s[ind:]
    else:
        ind = s.rfind("_")
        s = s[:ind] + "_ramp" + s[ind:]
    return s


def main():
    datasets = {
        "cty": ("DE",),
    }

    for set in datasets:
        data_1h = pd.read_pickle("_data/entsoe/entsoe_DE_AT_{}_1h".format(set))
        data_15min = pd.read_pickle("_data/entsoe/entsoe_DE_AT_{}_15min".format(set))

        for timedelta in ["1h", "15min"]:
            exec("data = data_{}".format(timedelta))

            data["price_day_ahead_DE"] = (
                data.fillna(0).prices_DE_AT_LU + data.fillna(0).prices_DE_LU
            )
            data["price_day_ahead_AT"] = (
                data.fillna(0).prices_DE_AT_LU + data.fillna(0).prices_AT
            )
            data.drop(
                ["prices_DE_LU", "prices_DE_AT_LU", "prices_AT"], axis=1, inplace=True
            )

            for region in datasets[set]:
                res_load = data["load_day_ahead_{}".format(region)].copy()
                for gen_type in ["gen_solar", "gen_wind_on", "gen_wind_off"]:
                    if "{}_day_ahead_{}".format(gen_type, region) in data.columns:
                        res_load -= data["{}_day_ahead_{}".format(gen_type, region)]
                data["residual_load_day_ahead_{}".format(region)] = res_load
                gen_type = "gen_run_off_hydro"
                if "{}_{}".format(gen_type, region) in data.columns:
                    res_load -= data["{}_{}".format(gen_type, region)].shift(
                        {"1h": 24, "15min": 96}[timedelta]
                    )
                    data["residual_load_hydro_day_ahead_{}".format(region)] = res_load

                res_load = data["load_{}".format(region)].copy()
                for gen_type in ["gen_solar", "gen_wind_on", "gen_wind_off"]:
                    if "{}_{}".format(gen_type, region) in data.columns:
                        res_load -= data["{}_{}".format(gen_type, region)]
                data["residual_load_{}".format(region)] = res_load
                gen_type = "gen_run_off_hydro"
                if "{}_{}".format(gen_type, region) in data.columns:
                    res_load -= data["{}_{}".format(gen_type, region)].shift(
                        {"1h": 24, "15min": 96}[timedelta]
                    )
                    data["residual_load_hydro_{}".format(region)] = res_load

                data["gen_total_{}".format(region)] = (
                    data.loc[:, data.columns.map(lambda x: "day_ahead" not in x)]
                    .filter(regex="^gen")
                    .filter(regex="{}$".format(region))
                    .sum(axis=1)
                )

            diff_data = data.diff()

            col_names = {}
            for region in datasets[set]:
                col_names.update(
                    {
                        i: j
                        for i, j in zip(
                            diff_data.columns[diff_data.columns.str.endswith(region)],
                            diff_data.columns[
                                diff_data.columns.str.endswith(region)
                            ].map(lambda x: rename_ramp_columns(x, region=region)),
                        )
                    }
                )
            col_names.update(
                {
                    i: j
                    for i, j in zip(
                        diff_data.columns[
                            ~diff_data.columns.str.endswith(datasets[set])
                        ],
                        diff_data.columns[
                            ~diff_data.columns.str.endswith(datasets[set])
                        ].map(rename_ramp_columns),
                    )
                }
            )
            diff_data.rename(columns=col_names, inplace=True)

            data = data.join(diff_data)

            for col_name in (
                data.loc[:, data.columns.map(lambda x: "price" not in x)]
                .filter(regex="day_ahead")
                .columns.values
            ):
                data["forecast_error_" + col_name.replace("_day_ahead", "")] = (
                    data[col_name] - data[col_name.replace("_day_ahead", "")]
                )

            data["month"] = data.index.month
            data["weekday"] = data.index.weekday
            data["hour"] = data.index.hour

            data["holiday"] = [int(i in holidays.DE()) for i in data.index]
            data["holiday_at"] = [int(i in holidays.AT()) for i in data.index]

            data.loc[data["holiday"] == 0, "holiday"] = data.loc[
                data["holiday"] == 0, "holiday_at"
            ]
            data.drop("holiday_at", axis=1, inplace=True)

            data.to_pickle(
                "_data/entsoe/entsoe_DE_AT_{}_engineered_{}".format(set, timedelta)
            )

    res_load = pd.DataFrame()
    res_load_hydro = pd.DataFrame()

    for country_code in [
        "DE",
        "AT",
        "BE",
        "CZ",
        "DK",
        "FR",
        "HU",
        "IT",
        "NL",
        "NO",
        "PL",
        "SE",
        "SI",
        "CH",
    ]:
        print(country_code)
        data = pd.read_pickle("data/entsoe/data_{}".format(country_code))

        data = data.resample("1H").mean()
        res_load[country_code] = data["Forecasted Load"].copy()
        for gen_type in [
            "Solar_day_ahead",
            "Wind Offshore_day_ahead",
            "Wind Onshore_day_ahead",
        ]:
            if gen_type in data.columns:
                res_load[country_code] -= data[gen_type].replace(np.nan, 0)
                # TODO: check validity of nan replacement

        res_load_hydro[country_code] = res_load[country_code].copy()

        if ("Hydro Run-of-river and poundage", "Actual Aggregated") in data.columns:
            res_load_hydro[country_code] -= (
                data[("Hydro Run-of-river and poundage", "Actual Aggregated")]
                .replace(np.nan, 0)
                .rolling(72)
                .mean()
                .shift(24)
            )
        elif "Hydro Run-of-river and poundage" in data.columns:
            res_load_hydro[country_code] -= (
                data["Hydro Run-of-river and poundage"]
                .replace(np.nan, 0)
                .rolling(72)
                .mean()
                .shift(24)
            )

    res_load.to_pickle("_data/entsoe/residual_loads")
    res_load_hydro.to_pickle("_data/entsoe/residual_loads_hydro")


if __name__ == "__main__":
    main()
