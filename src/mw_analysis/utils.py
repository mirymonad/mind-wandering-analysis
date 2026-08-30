from itertools import pairwise

import h5py
import numpy as np
import pandas as pd

from pandas import DataFrame
from scipy.signal import butter, sosfiltfilt


def mw_h5_to_df(filepath, columns) -> tuple[DataFrame, DataFrame]:
    with h5py.File(filepath, "r") as h:
        return (
            DataFrame(h["stream"], columns=columns),
            DataFrame.from_records(h["events"][:])
        )


def clean_events(events: DataFrame) -> DataFrame:
    return _add_missing_rest_switch(_remove_erroneous_events(events))


def _remove_erroneous_events(events_df: DataFrame) -> DataFrame:
    keep = []
    prev_switch = ""

    for _, row in events_df.iterrows():

        if row.event == b"switch":
            keep.append(row.value != prev_switch)
            prev_switch = row.value
        else:
            keep.append(prev_switch != b"rest.html")

    # Add missing front rest label due to initial refreshes
    return events_df[keep].reset_index(drop=True)


def _add_missing_rest_switch(events_df: DataFrame) -> DataFrame:
    first_row = events_df.iloc[0]

    if first_row.value == b"rest.html":
        return events_df

    if first_row.event != b"switch":
        raise Exception("invalid dataset")

    rest = first_row.copy()
    rest["timestamp"] -= 60.0
    rest["value"] = b"rest.html"

    return pd.concat([rest, events_df])


def od(stream_df, events_df):
    """
    Converts a stream of light intensity readings to OD readings.

    :param stream_df: the nirscord stream dataframe.
    :param events_df: the nirscord events dataframe.
    :return: the OD readings dataframe.
    """

    dataframe = DataFrame()

    # Extract the rest events dataframe
    rest_events = list(events_df.loc[
        events_df["event"].eq(b"switch")
        & events_df["value"].eq(b"rest.html")
    ].itertuples())

    # Convert the rest + experiment segments to OD
    for current, baseline_end in pairwise(rest_events):
        i_baseline = stream_df.loc[
            stream_df["timestamp"].ge(current.timestamp)
            & stream_df["timestamp"].lt(current.timestamp + 60)
        ]

        block_df = stream_df.loc[
            stream_df["timestamp"].ge(current.timestamp)
            & stream_df["timestamp"].lt(baseline_end.timestamp)
        ]

        # Compute the optical density using the baseline values
        cols = ["ir_r", "red_r", "ir_l", "red_l", "ir_p", "red_p"]
        block_df[cols] = np.log(i_baseline[cols].mean() / block_df[cols])

        # Add the optical density block onto the new dataframe
        dataframe = pd.concat([dataframe, block_df], ignore_index=True)

    return dataframe


def _sci_filter(data, *, fs=10, cutoff=(0.7, 1.5)):

    # Define the SCI Butterworth filter
    sos = butter(
        N=4,
        Wn=cutoff,
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    # Apply the custom filter to the data
    return sosfiltfilt(sos, data, axis=0)


def sci(ir_l1, ir_l2):
    """
    Computes the scalp coupling index (SCI) between two signals.

    :param ir_l1: the short near-infrared signal.
    :param ir_l2: the long near-infrared signal.
    :return: the scalp coupling index (SCI).
    """

    return np.corrcoef(
        _sci_filter(ir_l1),
        _sci_filter(ir_l2)
    )[0, 1]


def mbll(ir_l1, ir_l2, d=3.0, dpf=6.0):
    """
    Computes the change in HbO and Hb concentrations (ΔC).

    Vector Equation:

    ΔA = (d * dpf) * E * ΔC
    """

    # Stack the short wavelength on the matrix top
    A = np.vstack((ir_l1, ir_l2))

    # Define the absorption coefficients matrix (E)
    E = np.array([[0.265, 0.095], [0.21, 0.21]])

    # Compute the vector equation for concentrations (ΔC)
    return np.linalg.solve(E, A / (dpf * d))


def iir_filter(data, fs=10, cutoff=0.02):

    # Define the IIR Butterworth filter
    sos = butter(
        N=6,
        Wn=cutoff,
        btype="highpass",
        fs=fs,
        output="sos"
    )

    # Apply the custom filter to the data
    return sosfiltfilt(sos, data, axis=0)
