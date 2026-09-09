from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path(
    "data/processed/logs.parquet"
)


def load_logs(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the processed log dataset.

    Parameters
    ----------
    data_path:
        Path to the processed Parquet dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded log records.
    """

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Log dataset not found: {path}"
        )

    return pd.read_parquet(path)


def show_dataset_info(
    df: pd.DataFrame,
) -> None:
    """
    Display basic information about the log dataset.
    """

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())


def get_fatal_kernel_events(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return fatal kernel events.

    A fatal kernel event is a record where:
        component == 'KERNEL'
        severity == 'FATAL'
    """

    return df[
        (df["component"] == "KERNEL")
        & (df["severity"] == "FATAL")
    ]


def main() -> None:
    """
    Load the processed dataset and display
    a simple inspection of fatal kernel events.
    """

    df = load_logs()

    show_dataset_info(df)

    fatal_kernel_events = get_fatal_kernel_events(
        df
    )

    print("\nFatal kernel events:")
    print(fatal_kernel_events)


if __name__ == "__main__":
    main()