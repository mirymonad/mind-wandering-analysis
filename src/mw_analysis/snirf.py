import dataclasses

import numpy as np
from pandas import DataFrame
from snirf import Nirs, Snirf

from datetime import datetime
from dataclasses import dataclass


@dataclass
class NirscordDataset:
    subject_id: int
    stream_df: DataFrame
    events_df: DataFrame


def to_snirf(dataset: NirscordDataset, filepath: str):
    snirf = Snirf()

    snirf.nirs.appendGroup()
    nirs = snirf.nirs[0]

    snirf.formatVersion = "1.0"

    # Set the subject ID on the snirf HEHE
    nirs.metaDataTags.TimeUnit = "s"
    nirs.metaDataTags.LengthUnit = "m"
    nirs.metaDataTags.FrequencyUnit = "Hz"
    nirs.metaDataTags.SubjectID = dataset.subject_id

    now = datetime.now()

    nirs.metaDataTags.MeasurementDate = now.strftime("%Y-%m-%d")
    nirs.metaDataTags.MeasurementTime = now.strftime("%H:%M:%S")

    # Create the different sections of the SNIRF file
    _create_probe(dataset.stream_df, nirs)
    _create_data_block(dataset.stream_df, nirs)
    _create_measurement(dataset.stream_df, nirs)
    _create_events(dataset.events_df, nirs)

    snirf.save(filepath)
    snirf.close()


def _create_probe(stream_df: DataFrame, nirs: Nirs):
    source_names = []
    detector_names = []

    for name in stream_df.filter(like="hb").columns:

        # Remove HbO/HbR suffix.
        sd_name = name.rsplit(" ", 1)[0]
        source, detector = sd_name.split("_")

        if source not in source_names:
            source_names.append(source)

        if detector not in detector_names:
            detector_names.append(detector)

    probe = nirs.probe

    probe.sourceLabels = np.asarray(
        source_names,
        dtype=object,
    )

    probe.detectorLabels = np.asarray(
        detector_names,
        dtype=object,
    )

    probe.sourcePos3D = np.zeros(
        (len(source_names), 3),
        dtype=float,
    )

    probe.detectorPos3D = np.zeros(
        (len(detector_names), 3),
        dtype=float,
    )

    # Dummy wavelength as this is for HbO / Hb concentrations
    probe.wavelengths = np.array([0.0], dtype=float)


def _create_data_block(stream_df: DataFrame, nirs: Nirs):
    nirs.data.appendGroup()
    data_block = nirs.data[0]

    hbo = stream_df.filter(regex=r"hbo$")
    hb = stream_df.filter(regex=r"hb$")

    # Keep HbO/HbR together in the same order as the data matrix.
    data = np.hstack([hbo.values, hb.values])

    #
    timestamps = stream_df["timestamp"].values

    # Required: samples x channels
    data_block.dataTimeSeries = data

    # Required: time for every sample
    data_block.time = timestamps

    #
    data_block.name = "HbO_HbR"

def _create_measurement(stream_df: DataFrame, nirs: Nirs):
    data_block = nirs.data[0]

    for index, name in enumerate(stream_df.filter(like="hb").columns):
        data_block.measurementList.appendGroup()
        measurement = data_block.measurementList[-1]

        sd_name, signal = name.rsplit(" ", 1)
        source, detector = sd_name.split("_")

        # Set the extracted source and detector numbers
        measurement.sourceIndex = int(source[1:])
        measurement.detectorIndex = int(detector[1:])

        # Data type 99999 indicates the data is processed
        measurement.dataType = 99999
        measurement.dataTypeIndex = 1
        measurement.dataTypeLabel = signal

        # The default data unit set on all the measurements
        measurement.dataUnit = "M"

        # Points to the dummy wavelength to pass validation
        measurement.wavelengthIndex = 1


def _create_events(events_df: DataFrame, nirs: Nirs):
    events_df["value"] = events_df["value"].map({
        b"TS": 0,
        b"TF": 1,
        b"NS": 2,
        b"NF": 3,
        b"rest.html": 0,
        b"sart.html": 1,
    })

    for name, group in events_df.groupby("event"):
        nirs.stim.appendGroup()
        stim = nirs.stim[-1]
        stim.name = name

        onsets = group["timestamp"].values
        durations = np.zeros_like(onsets)
        values = group["value"].values

        stim.data = np.column_stack((onsets, durations, values))
        stim.dataLabels = np.asarray(["onset", "duration", "value"])
