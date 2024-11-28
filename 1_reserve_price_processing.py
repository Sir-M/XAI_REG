
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import holidays
import glob
import numpy as np

# Following data files have to be downloaded from regelleistung.net to ../_data/regelleistungnet:
#
# RESULT_OVERVIEW_aFRR_2017.csv
# ERGEBNISLISTE_ANONYM_SRL_2018-*.csv (multiple files)
# RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2018-01-01_2018-12-31.xlsx
# RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2019-01-01_2019-12-31.xlsx
# RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2020-01-01_2020-12-31.xlsx
# RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2021-01-01_2021-12-31.xlsx


def main():
    pos_afrr_price_data = pd.DataFrame(
        columns=[
            "pos_avg_cap_price",
            "pos_margin_cap_price",
            "pos_avg_en_price",
            "pos_margin_en_price",
        ]
    )
    neg_afrr_price_data = pd.DataFrame(
        columns=[
            "neg_avg_cap_price",
            "neg_margin_cap_price",
            "neg_avg_en_price",
            "neg_margin_en_price",
        ]
    )

    ### process 2017 data
    """
    data_2017 = pd.read_csv(
        "_data/regelleistungnet/RESULT_OVERVIEW_aFRR_2017.csv",
        delimiter=";",
        decimal=",",
    )
    data_2017.DATE_FROM = pd.to_datetime(data_2017.DATE_FROM, dayfirst=True)
    data_2017.DATE_TO = pd.to_datetime(data_2017.DATE_TO, dayfirst=True)

    df_pos_17 = pd.DataFrame(columns=data_2017.filter(regex="^TOTAL").columns)
    df_neg_17 = pd.DataFrame(columns=data_2017.filter(regex="^TOTAL").columns)
    for i in pd.date_range("2017-1-1", "2017-12-31"):
        df_temp = data_2017.loc[(data_2017.DATE_FROM <= i) & (data_2017.DATE_TO >= i)]
        df_pos_17.loc[i] = (
            df_temp.loc[df_temp.PRODUCT == "POS_NT"].filter(regex="^TOTAL").values[0]
        )
        df_neg_17.loc[i] = (
            df_temp.loc[df_temp.PRODUCT == "NEG_NT"].filter(regex="^TOTAL").values[0]
        )
        if (i.weekday() in [5, 6]) or (i in holidays.DE()):
            df_pos_17.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "POS_NT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
            df_neg_17.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "NEG_NT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
        else:
            df_pos_17.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "POS_HT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
            df_neg_17.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "NEG_HT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
        df_pos_17.loc[i + pd.Timedelta("20H")] = (
            df_temp.loc[df_temp.PRODUCT == "POS_NT"].filter(regex="^TOTAL").values[0]
        )
        df_neg_17.loc[i + pd.Timedelta("20H")] = (
            df_temp.loc[df_temp.PRODUCT == "NEG_NT"].filter(regex="^TOTAL").values[0]
        )

    pos_afrr_price_data = pd.concat(
        [
            pos_afrr_price_data,
            pd.DataFrame(
                {
                    "pos_avg_cap_price": df_pos_17["TOTAL_AVG_CAP_PRICE_[EUR/MWh]"],
                    "pos_margin_cap_price": df_pos_17["TOTAL_MARG_CAP_PRICE_[EUR/MWh]"],
                    "pos_avg_en_price": df_pos_17["TOTAL_AVG_EN_PRICE_[EUR/MWh]"],
                    "pos_margin_en_price": df_pos_17["TOTAL_MARG_EN_PRICE_[EUR/MWh]"],
                }
            ),
        ]
    )
    neg_afrr_price_data = pd.concat(
        [
            neg_afrr_price_data,
            pd.DataFrame(
                {
                    "neg_avg_cap_price": df_neg_17["TOTAL_AVG_CAP_PRICE_[EUR/MWh]"],
                    "neg_margin_cap_price": df_neg_17["TOTAL_MARG_CAP_PRICE_[EUR/MWh]"],
                    "neg_avg_en_price": df_neg_17["TOTAL_AVG_EN_PRICE_[EUR/MWh]"],
                    "neg_margin_en_price": df_neg_17["TOTAL_MARG_EN_PRICE_[EUR/MWh]"],
                }
            ),
        ]
    )
    """

    ### process first half of 2018

    cols = [
        "DATE_FROM",
        "DATE_TO",
        "PRODUCT",
        "TOTAL_AVG_CAP_PRICE_[EUR/MWh]",
        "TOTAL_MARG_CAP_PRICE_[EUR/MWh]",
        "TOTAL_AVG_EN_PRICE_[EUR/MWh]",
        "TOTAL_MARG_EN_PRICE_[EUR/MWh]",
    ]

    data_2018 = pd.DataFrame(columns=cols)
    for file in sorted(
        glob.glob("_data/regelleistungnet/ERGEBNISLISTE_ANONYM_SRL_2018-*.CSV")
    ):
        df_temp = pd.read_csv(
            file,
            delimiter=";",
            decimal=",",
            index_col=False,
        )
        if df_temp.empty:
            print(f"Empty file: {file}")
        df_temp.loc[
            (df_temp["AP_ZAHLUNGSRICHTUNG"] == "Netz an Anbieter")
            & (df_temp.PRODUKTNAME.isin(["NEG_HT", "NEG_NT"])),
            "ARBEITSPREIS [EUR/MWh]",
        ] = (
            -1
            * df_temp.copy().loc[
                (df_temp["AP_ZAHLUNGSRICHTUNG"] == "Netz an Anbieter")
                & (df_temp.PRODUKTNAME.isin(["NEG_HT", "NEG_NT"])),
                "ARBEITSPREIS [EUR/MWh]",
            ]
        )
        date_from = pd.to_datetime(df_temp["DATUM VON"][0], dayfirst=True)
        date_to = pd.to_datetime(df_temp["DATUM BIS"][0], dayfirst=True)
        for product in ["NEG_HT", "NEG_NT", "POS_HT", "POS_NT"]:
            marg_cap = df_temp.loc[
                (df_temp.PRODUKTNAME == product) & (df_temp.ANGEBOTE_AUS_AT != "X")
            ]["LEISTUNGSPREIS [EUR/MW]"].max()
            avg_cap = np.average(
                df_temp.loc[
                    (df_temp.PRODUKTNAME == product) & (df_temp.ANGEBOTE_AUS_AT != "X")
                ]["LEISTUNGSPREIS [EUR/MW]"],
                weights=df_temp.loc[
                    (df_temp.PRODUKTNAME == product) & (df_temp.ANGEBOTE_AUS_AT != "X")
                ]["BEZUSCHLAGTE_LEISTUNG [MW]"],
            )
            if product in ["NEG_HT", "NEG_NT"]:
                marg_en = df_temp.loc[
                    (df_temp.PRODUKTNAME == product) & (df_temp.ANGEBOTE_AUS_AT != "X")
                ]["ARBEITSPREIS [EUR/MWh]"].min()
            elif product in ["POS_HT", "POS_NT"]:
                marg_en = df_temp.loc[
                    (df_temp.PRODUKTNAME == product) & (df_temp.ANGEBOTE_AUS_AT != "X")
                ]["ARBEITSPREIS [EUR/MWh]"].max()
            avg_en = np.average(
                df_temp.loc[
                    (df_temp.PRODUKTNAME == product) & (df_temp.ANGEBOTE_AUS_AT != "X")
                ]["ARBEITSPREIS [EUR/MWh]"],
                weights=df_temp.loc[
                    (df_temp.PRODUKTNAME == product) & (df_temp.ANGEBOTE_AUS_AT != "X")
                ]["BEZUSCHLAGTE_LEISTUNG [MW]"],
            )
            data_2018 = pd.concat(
                [
                    data_2018,
                    pd.DataFrame(
                        [
                            [
                                date_from,
                                date_to,
                                product,
                                avg_cap,
                                marg_cap,
                                avg_en,
                                marg_en,
                            ]
                        ],
                        columns=cols,
                    ),
                ]
            )
    print("Hi!")
    print(data_2018.shape)
    print(data_2018.head())
    df_pos_18 = pd.DataFrame(columns=data_2018.filter(regex="^TOTAL").columns)
    print(df_pos_18.shape)
    df_neg_18 = pd.DataFrame(columns=data_2018.filter(regex="^TOTAL").columns)
    print(df_neg_18.shape)
    for i in pd.date_range("2018-1-1", "2018-7-11"):
        df_temp = data_2018.loc[(data_2018.DATE_FROM <= i) & (data_2018.DATE_TO >= i)]
        df_pos_18.loc[i] = (
            df_temp.loc[df_temp.PRODUCT == "POS_NT"].filter(regex="^TOTAL").values[0]
        )
        df_neg_18.loc[i] = (
            df_temp.loc[df_temp.PRODUCT == "NEG_NT"].filter(regex="^TOTAL").values[0]
        )
        if (i.weekday() in [5, 6]) or (i in holidays.DE()):
            df_pos_18.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "POS_NT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
            df_neg_18.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "NEG_NT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
        else:
            df_pos_18.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "POS_HT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
            df_neg_18.loc[i + pd.Timedelta("8H")] = (
                df_temp.loc[df_temp.PRODUCT == "NEG_HT"]
                .filter(regex="^TOTAL")
                .values[0]
            )
        df_pos_18.loc[i + pd.Timedelta("20H")] = (
            df_temp.loc[df_temp.PRODUCT == "POS_NT"].filter(regex="^TOTAL").values[0]
        )
        df_neg_18.loc[i + pd.Timedelta("20H")] = (
            df_temp.loc[df_temp.PRODUCT == "NEG_NT"].filter(regex="^TOTAL").values[0]
        )
    pos_afrr_price_data = pd.concat(
        [
            pos_afrr_price_data,
            pd.DataFrame(
                {
                    "pos_avg_cap_price": df_pos_18["TOTAL_AVG_CAP_PRICE_[EUR/MWh]"],
                    "pos_margin_cap_price": df_pos_18["TOTAL_MARG_CAP_PRICE_[EUR/MWh]"],
                    "pos_avg_en_price": df_pos_18["TOTAL_AVG_EN_PRICE_[EUR/MWh]"],
                    "pos_margin_en_price": df_pos_18["TOTAL_MARG_EN_PRICE_[EUR/MWh]"],
                }
            ),
        ]
    )
    neg_afrr_price_data = pd.concat(
        [
            neg_afrr_price_data,
            pd.DataFrame(
                {
                    "neg_avg_cap_price": df_neg_18["TOTAL_AVG_CAP_PRICE_[EUR/MWh]"],
                    "neg_margin_cap_price": df_neg_18["TOTAL_MARG_CAP_PRICE_[EUR/MWh]"],
                    "neg_avg_en_price": df_neg_18["TOTAL_AVG_EN_PRICE_[EUR/MWh]"],
                    "neg_margin_en_price": df_neg_18["TOTAL_MARG_EN_PRICE_[EUR/MWh]"],
                }
            ),
        ]
    )

    ### process second half  of 2018

    data_2018_2 = pd.read_excel(
        "_data/regelleistungnet/RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2018-01-01_2018-12-31.xlsx"
    )
    data_2018_2.index = data_2018_2.DATE_FROM + data_2018_2.PRODUCT.str.slice(
        4, 6, 1
    ).apply(lambda x: pd.Timedelta(x + "H"))
    df_pos_18_2 = data_2018_2.loc[data_2018_2.PRODUCT.str.contains("POS")].filter(
        regex="^GERMANY"
    )
    df_neg_18_2 = data_2018_2.loc[data_2018_2.PRODUCT.str.contains("NEG")].filter(
        regex="^GERMANY"
    )

    pos_afrr_price_data = pd.concat(
        [
            pos_afrr_price_data,
            pd.DataFrame(
                {
                    "pos_avg_cap_price": df_pos_18_2[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "pos_margin_cap_price": df_pos_18_2[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "pos_avg_en_price": df_pos_18_2[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "pos_margin_en_price": df_pos_18_2[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )
    neg_afrr_price_data = pd.concat(
        [
            neg_afrr_price_data,
            pd.DataFrame(
                {
                    "neg_avg_cap_price": df_neg_18_2[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "neg_margin_cap_price": df_neg_18_2[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "neg_avg_en_price": df_neg_18_2[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "neg_margin_en_price": df_neg_18_2[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )

    ### process 2019

    data_2019 = pd.read_excel(
        "_data/regelleistungnet/RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2019-01-01_2019-12-31.xlsx"
    )
    data_2019.index = data_2019.DATE_FROM + data_2019.PRODUCT.str.slice(4, 6, 1).apply(
        lambda x: pd.Timedelta(x + "H")
    )
    df_pos_19 = data_2019.loc[data_2019.PRODUCT.str.contains("POS")].filter(
        regex="^GERMANY"
    )
    df_neg_19 = data_2019.loc[data_2019.PRODUCT.str.contains("NEG")].filter(
        regex="^GERMANY"
    )

    pos_afrr_price_data = pd.concat(
        [
            pos_afrr_price_data,
            pd.DataFrame(
                {
                    "pos_avg_cap_price": df_pos_19[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "pos_margin_cap_price": df_pos_19[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "pos_avg_en_price": df_pos_19[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "pos_margin_en_price": df_pos_19[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )
    neg_afrr_price_data = pd.concat(
        [
            neg_afrr_price_data,
            pd.DataFrame(
                {
                    "neg_avg_cap_price": df_neg_19[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "neg_margin_cap_price": df_neg_19[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "neg_avg_en_price": df_neg_19[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "neg_margin_en_price": df_neg_19[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )

    ### process 2020

    data_2020 = pd.read_excel(
        "_data/regelleistungnet/RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2020-01-01_2020-12-31.xlsx"
    )
    data_2020.index = data_2020.DATE_FROM + data_2020.PRODUCT.str.slice(4, 6, 1).apply(
        lambda x: pd.Timedelta(x + "H")
    )
    df_pos_20 = data_2020.loc[data_2020.PRODUCT.str.contains("POS")].filter(
        regex="^GERMANY"
    )
    df_neg_20 = data_2020.loc[data_2020.PRODUCT.str.contains("NEG")].filter(
        regex="^GERMANY"
    )

    pos_afrr_price_data = pd.concat(
        [
            pos_afrr_price_data,
            pd.DataFrame(
                {
                    "pos_avg_cap_price": df_pos_20[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "pos_margin_cap_price": df_pos_20[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "pos_avg_en_price": df_pos_20[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "pos_margin_en_price": df_pos_20[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )
    neg_afrr_price_data = pd.concat(
        [
            neg_afrr_price_data,
            pd.DataFrame(
                {
                    "neg_avg_cap_price": df_neg_20[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "neg_margin_cap_price": df_neg_20[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[EUR/MW]"
                    ],
                    "neg_avg_en_price": df_neg_20[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "neg_margin_en_price": df_neg_20[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )

    ### process 2021

    data_2021 = pd.read_excel(
        "_data/regelleistungnet/RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2021-01-01_2021-12-31.xlsx"
    )
    data_2021.index = data_2021.DATE_FROM + data_2021.PRODUCT.str.slice(4, 6, 1).apply(
        lambda x: pd.Timedelta(x + "H")
    )
    df_pos_21 = data_2021.loc[data_2021.PRODUCT.str.contains("POS")].filter(
        regex="^GERMANY"
    )
    df_neg_21 = data_2021.loc[data_2021.PRODUCT.str.contains("NEG")].filter(
        regex="^GERMANY"
    )

    pos_afrr_price_data = pd.concat(
        [
            pos_afrr_price_data,
            pd.DataFrame(
                {
                    "pos_avg_cap_price": df_pos_21[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"
                    ],
                    "pos_margin_cap_price": df_pos_21[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]"
                    ],
                    "pos_avg_en_price": df_pos_21[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "pos_margin_en_price": df_pos_21[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )
    neg_afrr_price_data = pd.concat(
        [
            neg_afrr_price_data,
            pd.DataFrame(
                {
                    "neg_avg_cap_price": df_neg_21[
                        "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"
                    ],
                    "neg_margin_cap_price": df_neg_21[
                        "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]"
                    ],
                    "neg_avg_en_price": df_neg_21[
                        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
                    ],
                    "neg_margin_en_price": df_neg_21[
                        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
                    ],
                }
            ),
        ]
    )

    ### save data
    pos_afrr_price_data.to_pickle("_data/regelleistungnet/pos_afrr_price_data.pkl")
    neg_afrr_price_data.to_pickle("_data/regelleistungnet/neg_afrr_price_data.pkl")


if __name__ == "__main__":
    main()
