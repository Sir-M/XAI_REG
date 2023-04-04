import pandas as pd


def main():
    start = pd.Timestamp("20150101", tz="EUROPE/BERLIN")
    end = pd.Timestamp("20220101", tz="EUROPE/BERLIN")

    df = pd.DataFrame(index=pd.date_range(start, end, freq="1H", tz="EUROPE/BERLIN"))
    df = (
        df.join(
            pd.read_pickle("../_data/entsoe/raw_data/prices_DE_AT_LU").rename(
                "prices_DE_AT_LU"
            )
        )
        .join(
            pd.read_pickle("../_data/entsoe/raw_data/prices_DE_LU").rename(
                "prices_DE_LU"
            )
        )
        .join(pd.read_pickle("../_data/entsoe/raw_data/prices_AT").rename("prices_AT"))
    )
    df = df.tz_convert(tz="EUROPE/BERLIN").dropna(how="all")

    datasets = {"cty": ["DE"]}

    for set in datasets:
        data_1h = pd.DataFrame(
            index=pd.date_range(start, end, freq="1H", tz="EUROPE/BERLIN")
        )
        data_15min = pd.DataFrame(
            index=pd.date_range(start, end, freq="15t", tz="EUROPE/BERLIN")
        )
        for region in datasets[set]:
            print("processing {}".format(region))

            data_temp = pd.read_pickle(
                "../_data/entsoe/raw_data/data_{}".format(region)
            )

            rename_dict = {
                "Forecasted Load": "load_day_ahead",
                "Actual Load": "load",
                "Actual Aggregated": "gen_total_day_ahead",
                "Solar_day_ahead": "gen_solar_day_ahead",
                "Wind Offshore_day_ahead": "gen_wind_off_day_ahead",
                "Wind Onshore_day_ahead": "gen_wind_on_day_ahead",
                "Biomass": "gen_biomass",
                "Fossil Brown coal/Lignite": "gen_lignite",
                "Fossil Coal-derived gas": "gen_coal_gas",
                "Fossil Gas": "gen_gas",
                "Fossil Hard coal": "gen_hard_coal",
                "Fossil Oil": "gen_oil",
                "Geothermal": "gen_geothermal",
                "Hydro Run-of-river and poundage": "gen_run_off_hydro",
                "Hydro Water Reservoir": "gen_reservoir_hydro",
                "Nuclear": "gen_nuclear",
                "Other": "gen_other",
                "Other renewable": "gen_other_renewable",
                "Solar": "gen_solar",
                "Waste": "gen_waste",
                "Wind Offshore": "gen_wind_off",
                "Wind Onshore": "gen_wind_on",
                ("Biomass", "Actual Aggregated"): "gen_biomass",
                ("Fossil Brown coal/Lignite", "Actual Aggregated"): "gen_lignite",
                ("Fossil Coal-derived gas", "Actual Aggregated"): "gen_coal_gas",
                ("Fossil Gas", "Actual Aggregated"): "gen_gas",
                ("Fossil Hard coal", "Actual Aggregated"): "gen_hard_coal",
                ("Fossil Oil", "Actual Aggregated"): "gen_oil",
                ("Geothermal", "Actual Aggregated"): "gen_geothermal",
                ("Hydro Pumped Storage", "Actual Aggregated"): "gen_pumped_hydro",
                (
                    "Hydro Pumped Storage",
                    "Actual Consumption",
                ): "pumped_hydro_consumption",
                (
                    "Hydro Run-of-river and poundage",
                    "Actual Aggregated",
                ): "gen_run_off_hydro",
                ("Hydro Water Reservoir", "Actual Aggregated"): "gen_reservoir_hydro",
                ("Nuclear", "Actual Aggregated"): "gen_nuclear",
                ("Other", "Actual Aggregated"): "gen_other",
                ("Other renewable", "Actual Aggregated"): "gen_other_renewable",
                ("Solar", "Actual Aggregated"): "gen_solar",
                ("Waste", "Actual Aggregated"): "gen_waste",
                ("Wind Offshore", "Actual Aggregated"): "gen_wind_off",
                ("Wind Onshore", "Actual Aggregated"): "gen_wind_on",
            }

            data_temp = data_temp.rename(columns=rename_dict).filter(
                regex="^(?!.*Consumption).*$", axis=1
            )

            if region in ["DE", "DE_TRANSNET"]:
                data_temp.loc["2018":, "gen_total_day_ahead"] = data_temp.loc[
                    "2018":, 0
                ]
                data_temp.drop(0, axis=1, inplace=True)
            data_temp.columns = data_temp.columns.map(
                lambda x: str(x) + "_{}".format(region)
            )

            data_15min = data_15min.join(data_temp)
            data_15min["gen_total_day_ahead_{}".format(region)] = (
                data_15min["gen_total_day_ahead_{}".format(region)]
                .dropna()
                .resample("15T")
                .interpolate(limit=3)
            )

            data_temp = data_temp.resample("1H").mean().dropna(how="all")
            data_temp = data_temp.groupby(level=0, axis=1).sum(min_count=1)

            data_1h = data_1h.join(data_temp)

        data_1h.join(df).to_pickle("../_data/entsoe/entsoe_DE_AT_{}_1h".format(set))
        data_15min.join(df.resample("15t").interpolate(limit=3)).to_pickle(
            "../_data/entsoe/entsoe_DE_AT_{}_15min".format(set)
        )


if __name__ == "__main__":
    main()
