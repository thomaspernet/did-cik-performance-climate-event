"""
sheldus_treatment_builder.py

Match firm production locations to SHELDUS disaster events by state/city/year.

This class handles only the geographic matching step. Treatment assignment
(event indicators, early-treated flags, pre/post counts) is handled by
``did_panel_builder.TreatmentAssigner``.
"""

import pandas as pd
from typing import Optional, List


class SheldusTreatmentBuilder:
    """
    Match firm locations to SHELDUS climate events.

    Takes pre-loaded and pre-filtered dataframes. All data filtering
    (thresholding, year ranges, etc.) should happen BEFORE passing to this class.

    Parameters
    ----------
    df_sheldus : pd.DataFrame
        SHELDUS event data (already filtered) with columns: state, city, year,
        and ``event_column``.
    df_location : pd.DataFrame
        Firm production locations with columns: cik, year, state, city.
    event_column : str
        Column to use for event indicator (default: 'propertydmgadj_2020').

    Example
    -------
    >>> from climate_study import SheldusTreatmentBuilder
    >>> builder = SheldusTreatmentBuilder(
    ...     df_sheldus=df_sheldus_filtered,
    ...     df_location=df_location,
    ...     event_column="propertydmgadj_2020",
    ... )
    >>> df_matched = builder.build()
    """

    def __init__(
        self,
        df_sheldus: pd.DataFrame,
        df_location: pd.DataFrame,
        event_column: str = "propertydmgadj_2020",
    ):
        self.df_sheldus = df_sheldus.copy()
        self.df_location = df_location.copy()
        self.event_column = event_column

        # Computed attributes
        self._df_merged: Optional[pd.DataFrame] = None

        # Process tracking
        self.n_sheldus = len(df_sheldus)
        self.n_locations = len(df_location)
        self.n_merged: int = 0
        self.n_firm_years: int = 0
        self.n_with_city: int = 0
        self.n_without_city: int = 0
        self.n_all_firms: int = 0

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input dataframes."""
        required_sheldus = ['state', 'city', 'year']
        missing_sheldus = set(required_sheldus) - set(self.df_sheldus.columns)
        if missing_sheldus:
            raise ValueError(f"df_sheldus missing required columns: {missing_sheldus}")

        if self.event_column not in self.df_sheldus.columns:
            raise ValueError(
                f"Event column '{self.event_column}' not found in df_sheldus. "
                f"Available columns: {list(self.df_sheldus.columns)}"
            )

        required_location = ['cik', 'year', 'state', 'city']
        missing_location = set(required_location) - set(self.df_location.columns)
        if missing_location:
            raise ValueError(f"df_location missing required columns: {missing_location}")

        # Ensure types
        self.df_sheldus['year'] = self.df_sheldus['year'].astype(str)
        self.df_location['year'] = self.df_location['year'].astype(str)
        self.df_location['cik'] = self.df_location['cik'].astype(str)

        print(f"SheldusTreatmentBuilder initialized")
        print(f"  SHELDUS observations: {len(self.df_sheldus):,}")
        print(f"  Location observations: {len(self.df_location):,}")
        print(f"  Event column: {self.event_column}")

    def merge_firm_events(self) -> pd.DataFrame:
        """
        Merge firm locations with SHELDUS events by state/city/year.

        Keeps ALL firm-year observations. Firms with missing city data
        or no matching SHELDUS events get event = 0.

        Creates ``no_city_match`` flag to identify firms that cannot be matched
        to SHELDUS because they lack valid city data.

        Returns
        -------
        pd.DataFrame
            Merged firm-event data with columns:
            - cik, year: firm-year identifiers
            - event_column: aggregated event value (0 if no match)
            - no_city_match: 1 if firm has no valid city data for any year
        """
        all_cik_years = (
            self.df_location[['cik', 'year']]
            .drop_duplicates()
            .copy()
        )

        df_with_city = self.df_location.dropna(subset=['state', 'city'])
        ciks_with_city = set(df_with_city['cik'].unique())
        ciks_all = set(self.df_location['cik'].unique())
        ciks_without_city = ciks_all - ciks_with_city

        df_events = (
            self.df_sheldus
            .merge(
                df_with_city[['cik', 'year', 'state', 'city']],
                on=['state', 'city', 'year'],
                how='inner',
            )
        )

        df_events_agg = (
            df_events
            .groupby(['cik', 'year'])
            .agg({self.event_column: 'sum'})
            .reset_index()
        )

        df_merged = (
            all_cik_years
            .merge(df_events_agg, on=['cik', 'year'], how='left')
            .fillna({self.event_column: 0})
        )

        df_merged['no_city_match'] = df_merged['cik'].isin(ciks_without_city).astype(int)

        self._df_merged = df_merged
        self.n_merged = len(df_merged)
        self.n_with_city = len(ciks_with_city)
        self.n_without_city = len(ciks_without_city)
        self.n_all_firms = len(ciks_all)

        return df_merged

    def aggregate_firm_year(self) -> pd.DataFrame:
        """
        Aggregate climate events to firm-year level.

        Returns
        -------
        pd.DataFrame
            Firm-year level data with aggregated event values.
        """
        if self._df_merged is None:
            self.merge_firm_events()
        return self._df_merged.copy()

    def build(self) -> pd.DataFrame:
        """
        Execute matching pipeline: merge locations with SHELDUS events and
        add ``has_event`` binary indicator.

        Returns
        -------
        pd.DataFrame
            Firm-year panel with columns:
            - cik, year: firm-year identifiers
            - {event_column}: aggregated damage value
            - has_event: 1 if climate event this year
            - no_city_match: 1 if firm lacks city data
        """
        self.merge_firm_events()
        df = self.aggregate_firm_year()
        self.n_firm_years = len(df)

        # Binary indicator
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['has_event'] = (df[self.event_column] > 0).astype(int)

        print(f"Matched firm-event data built")
        print(f"  All unique firms: {self.n_all_firms:,}")
        print(f"    - With valid city data: {self.n_with_city:,}")
        print(f"    - Without city data (never-treated): {self.n_without_city:,}")
        print(f"  Firm-years: {len(df):,}")
        print(f"  Unique firms in panel: {df['cik'].nunique():,}")

        return df

    def summary(self) -> pd.DataFrame:
        """
        Return yearly summary of matched events.

        Returns
        -------
        pd.DataFrame
            Summary by year with firm counts and event rates.
        """
        if self._df_merged is None:
            raise ValueError("Must call build() first")

        df = self._df_merged.copy()
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['has_event'] = (df[self.event_column] > 0).astype(int)

        yearly = df.groupby('year').agg(
            n_firms=('cik', 'nunique'),
            n_events=('has_event', 'sum'),
            event_value_mean=(self.event_column, 'mean'),
            event_value_total=(self.event_column, 'sum'),
        ).reset_index()

        yearly['event_rate'] = yearly['n_events'] / yearly['n_firms'] * 100

        n_firms = df['cik'].nunique()
        print(f"Sample Summary:")
        print(f"  Firm-years: {len(df):,}")
        print(f"  Unique firms: {n_firms:,}")
        print(f"  Treatment rate: {df['has_event'].mean()*100:.2f}%")

        return yearly

    def get_location_events(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Export firm locations merged with SHELDUS events for mapping.

        Returns location-level data (not aggregated to firm-year) with
        coordinates for geographic visualization.

        Parameters
        ----------
        years : list of int, optional
            Filter to specific years. If None, returns all years.

        Returns
        -------
        pd.DataFrame
            Location-event data with columns:
            - cik, year, state, city: identifiers
            - has_event: 1 if climate event at this location in this year
            - city_latitude, city_longitude: coordinates (if available)
            - event_column value: damage amount
        """
        df_with_city = self.df_location.dropna(subset=['state', 'city']).copy()

        if years is not None:
            years_str = [str(y) for y in years]
            df_with_city = df_with_city[df_with_city['year'].isin(years_str)]

        sheldus_cols = ['state', 'city', 'year', self.event_column]
        if 'city_latitude' in self.df_sheldus.columns:
            sheldus_cols.extend(['city_latitude', 'city_longitude'])

        df_sheldus_sub = self.df_sheldus[sheldus_cols].copy()

        if years is not None:
            df_sheldus_sub = df_sheldus_sub[df_sheldus_sub['year'].isin(years_str)]

        df_merged = df_with_city.merge(
            df_sheldus_sub,
            on=['state', 'city', 'year'],
            how='left',
        )

        df_merged[self.event_column] = df_merged[self.event_column].fillna(0)
        df_merged['has_event'] = (df_merged[self.event_column] > 0).astype(int)

        if 'city_latitude' in df_merged.columns:
            coord_lookup = (
                self.df_sheldus[['state', 'city', 'city_latitude', 'city_longitude']]
                .drop_duplicates()
            )

            df_merged = df_merged.merge(
                coord_lookup.rename(columns={
                    'city_latitude': 'lat_lookup',
                    'city_longitude': 'lon_lookup',
                }),
                on=['state', 'city'],
                how='left',
            )

            df_merged['city_latitude'] = df_merged['city_latitude'].fillna(df_merged['lat_lookup'])
            df_merged['city_longitude'] = df_merged['city_longitude'].fillna(df_merged['lon_lookup'])
            df_merged = df_merged.drop(columns=['lat_lookup', 'lon_lookup'])

        return df_merged
