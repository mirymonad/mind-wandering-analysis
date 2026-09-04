import h5py
import numpy as np
import pandas as pd

from pandas import DataFrame
from scipy.signal import butter, sosfiltfilt


def nirscord_h5_to_df(filepath, channels) -> tuple[DataFrame, DataFrame]:
    """
    Reads a nirscord h5 file to a stream and event dataframe.

    :param filepath: the filepath of the nirscord h5 file.
    :param channels: the channels of the nirscord stream.
    :return: a separate stream and events dataframe.
    """

    with h5py.File(filepath, "r") as h:
        return (
            DataFrame(h["stream"], columns=channels),
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

    return pd.concat([rest.to_frame().T, events_df]).reset_index(drop=True)


def find_experiment_start(events_df: DataFrame) -> float:
    events_df = events_df.loc[
        (events_df["event"].eq(b"switch")) &
        (events_df["value"].eq(b"rest.html"))
    ]

    if len(events_df) >= 3:
        return events_df.iloc[-3].timestamp
    else:
        raise Exception("invalid dataset")


def od(stream_df, events_df, cols):
    """
    Converts a stream of light intensity readings to OD readings.

    OD equation:

    OD = log(I_baseline / I)

    :param stream_df: a clean nirscord stream dataframe.
    :param events_df: a clean nirscord events dataframe.
    :param cols: a list of column names to convert.
    """

    # Extract the rest events dataframe
    switch_events = events_df.loc[events_df["event"].eq(b"switch")]

    #
    i_baseline = stream_df.loc[
        (stream_df["timestamp"].ge(switch_events.iloc[0].timestamp + 5))
        & stream_df["timestamp"].lt(switch_events.iloc[1].timestamp - 10)
    ]

    # Compute the optical density using the baseline values
    stream_df[cols] = np.log(i_baseline[cols].mean() / stream_df[cols])


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
    return np.linalg.solve(E, A / (dpf * d)) * 1000


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


def short_channel_regressor(long_channels, short_channel):
    """
    Removes the component of each long-channel HbO signal explained by the short-channel HbO signal.

    OLS equation:

    Y = β0 + β1*S + ε

    The returned long-channel signals are the OLS residuals:

    ε = Y - β0 - β1*S
    """

    X = short_channel.to_numpy()
    Y = long_channels.to_numpy()

    # Initialise the input values for ordinary least squares (OLS)
    X = np.column_stack([np.ones(len(X)), X])

    # Compute the OLS to automatically determine β0 + β1 constants
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)

    # Perform matrix multiplication on beta to calculate β0 + β1*S
    return long_channels - (X @ beta)


def detect_motion_spikes(df, cols, fs=32.25, period=20, threshold=2):
    df = df.copy()

    window = int(fs * period)

    baseline = df[cols].rolling(
        window=window,
        center=True,
        min_periods=window // 2
    ).median()

    mad = (df[cols] - baseline).abs().rolling(
        window=window,
        center=True,
        min_periods=window // 2
    ).median()

    robust_sd = 1.4826 * mad

    z = (df[cols] - baseline) / robust_sd.replace(0, np.nan)

    spikes = z.abs() > threshold

    # Replace only detected spikes
    df[cols] = df[cols].mask(spikes)

    # Interpolate them
    df[cols] = df[cols].interpolate()

    return df


def collect_epochs(datasets, segment: range):
    collected_error_epochs = []
    collected_no_error_epochs = []

    for dataset in datasets:
        no_error_epoch, error_epoch = extract_epochs(
            dataset.stream_df, dataset.events_df, segment
        )

        # Collect the epochs from every participant
        collected_error_epochs.extend(error_epoch)
        collected_no_error_epochs.extend(no_error_epoch)

    # Return the epochs in the same order as the snirf file
    return collected_no_error_epochs, collected_error_epochs


def extract_epochs(stream_df, events_df, segment: range):
    epochs = {0: [], 1: []}

    for value in epochs:
        target_trials = events_df[events_df["value"] == value]

        for index, row in target_trials.iterrows():
            epochs[value].append(stream_df[
                (stream_df["timestamp"] >= row.timestamp + segment.start) &
                (stream_df["timestamp"] < row.timestamp + segment.stop)
            ].reset_index(drop=True))

    return epochs[0], epochs[1]
