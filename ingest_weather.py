import argparse
from pathlib import Path

import pandas as pd

# GHCNh hourly station files are available via HTTPS.
GHCNH_BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network"

# Core NYC-area stations (well-known, high-coverage).
CORE_STATIONS = [
    "USW00094728",  # Central Park
    "USW00094789",  # JFK
    "USW00014732",  # LaGuardia
    "USW00014734",  # Newark
]

OUT_DIR = Path("data/raw/weather_hourly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_FILE_TEMPLATE = (
    GHCNH_BASE_URL + "/hourly/access/by-station/GHCNh_{station}_por.psv"
)

USE_COLS = [
    "Station_ID",
    "Station_name",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Latitude",
    "Longitude",
    "Elevation",
    "temperature",
    "dew_point_temperature",
    "station_level_pressure",
    "sea_level_pressure",
    "wind_speed",
    "wind_gust",
    "precipitation",
    "relative_humidity",
]


def parse_datetime_frame(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        dict(
            year=frame["Year"],
            month=frame["Month"],
            day=frame["Day"],
            hour=frame["Hour"],
            minute=frame["Minute"],
        ),
        errors="coerce",
    )


def read_station_hourly(
    station_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    chunksize: int,
) -> pd.DataFrame:
    url = STATION_FILE_TEMPLATE.format(station=station_id)
    print("station_url:", station_id, url)

    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(url, sep="|", usecols=USE_COLS, chunksize=chunksize):
        chunk["datetime"] = parse_datetime_frame(chunk)
        mask = (chunk["datetime"] >= start_ts) & (chunk["datetime"] <= end_ts)
        if mask.any():
            kept.append(chunk.loc[mask].copy())

    if not kept:
        return pd.DataFrame(columns=USE_COLS + ["datetime"])
    return pd.concat(kept, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NOAA GHCNh hourly weather.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--stations",
        default=",".join(CORE_STATIONS),
        help="Comma-separated station IDs (default: core NYC stations).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Rows per read chunk (keep modest to limit memory).",
    )
    args = parser.parse_args()

    start_ts = pd.to_datetime(f"{args.start} 00:00")
    end_ts = pd.to_datetime(f"{args.end} 23:59")

    station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
    if not station_ids:
        raise ValueError("No stations provided.")

    frames: list[pd.DataFrame] = []
    for station_id in station_ids:
        df_station = read_station_hourly(station_id, start_ts, end_ts, args.chunksize)
        if not df_station.empty:
            frames.append(df_station)

    if not frames:
        raise ValueError("No data found for the provided stations and date range.")

    df = pd.concat(frames, ignore_index=True)

    print("shape:", df.shape)
    print("cols:", df.columns.tolist())
    print(df.head())

    # Standardize columns
    df.columns = [c.lower() for c in df.columns]

    for col in [
        "temperature",
        "dew_point_temperature",
        "station_level_pressure",
        "sea_level_pressure",
        "wind_speed",
        "wind_gust",
        "precipitation",
        "relative_humidity",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_rain"] = (df["precipitation"] > 0).astype(int)

    print("datetime_range:", df["datetime"].min(), "to", df["datetime"].max())

    out_path = OUT_DIR / f"weather_hourly_{args.start}_to_{args.end}.parquet"
    df.to_parquet(out_path, index=False)
    print("saved:", out_path, "rows:", len(df))


if __name__ == "__main__":
    main()
