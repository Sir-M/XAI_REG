from entsoe import EntsoePandasClient
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def main():
    api_key = "YOUR_API_KEY"

    client = EntsoePandasClient(api_key=api_key)

    tzone = "EUROPE/BERLIN"
    start = pd.Timestamp("20150101", tz=tzone)
    end = pd.Timestamp("20220101", tz=tzone)

    # get data from DE/AT:
    for country_code in ["DE", "AT"]:

        print(country_code)
        data = pd.DataFrame(
            index=pd.date_range(start, end, freq="15T", tz="EUROPE/BERLIN")
        )

        print("get load data")
        data = data.join(
            client.query_load_and_forecast(country_code, start=start, end=end)
        )

        print("get generation forecast data")
        data = data.join(
            client.query_generation_forecast(country_code, start=start, end=end)
        )

        print("get renewable forecast data")
        renewable_forecast = client.query_wind_and_solar_forecast(
            country_code, start=start, end=end, psr_type=None
        )
        renewable_forecast.columns = renewable_forecast.columns.map(
            lambda x: str(x) + "_day_ahead"
        )
        data = data.join(renewable_forecast)

        print("get generation data")
        data = data.join(
            client.query_generation(country_code, start=start, end=end, psr_type=None)
        )

        data.to_pickle("../_data/entsoe/raw_data/data_{}".format(country_code))

    # get day ahead data in neighbor countries
    for country_code in [
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
        data = pd.DataFrame(
            index=pd.date_range(start, end, freq="15T", tz="EUROPE/BERLIN")
        )

        print("get load data")
        data = data.join(
            client.query_load_and_forecast(country_code, start=start, end=end)
        )

        print("get generation forecast data")
        data = data.join(
            client.query_generation_forecast(country_code, start=start, end=end)
        )

        print("get renewable forecast data")
        renewable_forecast = client.query_wind_and_solar_forecast(
            country_code, start=start, end=end, psr_type=None
        )
        renewable_forecast.columns = renewable_forecast.columns.map(
            lambda x: str(x) + "_day_ahead"
        )
        data = data.join(renewable_forecast)

        print("get generation data")
        data = data.join(
            client.query_generation(country_code, start=start, end=end, psr_type=None)
        )

        data.to_pickle("data/entsoe/data_{}".format(country_code))

    # day-ahead price data before BZ split
    start = pd.Timestamp("20150101", tz=tzone)
    end = pd.Timestamp("20181001", tz=tzone)

    country_code = "DE_AT_LU"
    client.query_day_ahead_prices(country_code, start=start, end=end).to_pickle(
        "../_data/entsoe/raw_data/prices_{}".format(country_code)
    )

    # day-ahead price data after BZ split
    start = pd.Timestamp("20181001", tz=tzone)
    end = pd.Timestamp("20220101", tz=tzone)

    for country_code in ["DE_LU", "AT"]:
        client.query_day_ahead_prices(country_code, start=start, end=end).to_pickle(
            "../_data/entsoe/raw_data/prices_{}".format(country_code)
        )


if __name__ == "__main__":
    main()
