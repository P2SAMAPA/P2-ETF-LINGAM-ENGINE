"""
Data Loading Module
====================
Handles loading and caching of HuggingFace datasets.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, load_from_disk
from config import HF_CONFIG, ETF_UNIVERSE
from pathlib import Path


class ETFDataLoader:
    """Loads and manages ETF data from HuggingFace."""

    def __init__(self, cache_dir: str = None):
        """Initialize the data loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir or HF_CONFIG["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        self.raw_data = None

    def load_dataset(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the ETF dataset from HuggingFace.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with ETF data indexed by date
        """
        cache_path = self.cache_dir / "master_data"

        if use_cache and cache_path.exists():
            print(f"Loading data from cache: {cache_path}")
            self.raw_data = pd.read_parquet(cache_path)
        else:
            print(f"Loading data from HuggingFace: {HF_CONFIG['input_dataset']}")
            ds = load_dataset(HF_CONFIG["input_dataset"])
            self.raw_data = ds["train"].to_pandas()

            # Set index from __index_level_0__ column
            if "__index_level_0__" in self.raw_data.columns:
                self.raw_data["date"] = pd.to_datetime(self.raw_data["__index_level_0__"])
                self.raw_data.set_index("date", inplace=True)
                self.raw_data.drop("__index_level_0__", axis=1, inplace=True)

            # Cache the data
            self.raw_data.to_parquet(cache_path)
            print(f"Data cached to: {cache_path}")

        return self.raw_data

    def get_universe_data(self, universe: str) -> pd.DataFrame:
        """Get data for a specific ETF universe.

        Args:
            universe: 'fi_commodity' or 'equity'

        Returns:
            DataFrame with only the specified universe features
        """
        if self.raw_data is None:
            self.load_dataset()

        universe_config = ETF_UNIVERSE[universe]
        features = universe_config["all_features"]

        # Select only the required columns
        available_features = [f for f in features if f in self.raw_data.columns]
        data = self.raw_data[available_features].copy()

        return data

    def get_benchmark_data(self, universe: str) -> pd.Series:
        """Get benchmark data for a universe.

        Args:
            universe: 'fi_commodity' or 'equity'

        Returns:
            Series with benchmark returns
        """
        if self.raw_data is None:
            self.load_dataset()

        benchmark = ETF_UNIVERSE[universe]["benchmark"]

        if benchmark in self.raw_data.columns:
            return self.raw_data[benchmark].copy()
        else:
            raise ValueError(f"Benchmark {benchmark} not found in dataset")

    def get_macro_data(self, universe: str = None) -> pd.DataFrame:
        """Get macro variable data.

        Args:
            universe: Optional universe to get specific macro vars

        Returns:
            DataFrame with macro variables
        """
        if self.raw_data is None:
            self.load_dataset()

        macro_cols = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]
        available_cols = [c for c in macro_cols if c in self.raw_data.columns]

        return self.raw_data[available_cols].copy()

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data.

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with daily returns
        """
        returns = data.pct_change()
        return returns

    def get_training_windows(
        self,
        universe: str,
        start_year: int = 2008,
        end_year: int = 2026,
        min_years: int = 2
    ) -> list:
        """Generate shrinking window date ranges.

        Args:
            universe: ETF universe
            start_year: Earliest year for windows
            end_year: Latest year for windows
            min_years: Minimum years in a window

        Returns:
            List of (start_date, end_date) tuples
        """
        if self.raw_data is None:
            self.load_dataset()

        windows = []

        for start_y in range(start_year, end_year - min_years + 1):
            start_date = f"{start_y}-01-01"
            end_date = f"{end_year}-12-31"

            # Filter to available dates
            mask = (self.raw_data.index >= start_date) & (self.raw_data.index <= end_date)
            window_data = self.raw_data[mask]

            if len(window_data) > 0:
                windows.append((start_date, end_date))

        return windows


def main():
    """Test data loading."""
    loader = ETFDataLoader()
    data = loader.load_dataset()
    print(f"Loaded {len(data)} rows")
    print(f"Columns: {list(data.columns)}")
    print(data.head())


if __name__ == "__main__":
    main()
