import os
import sys
import csv
import struct
from copy import deepcopy

import pandas as pd
import numpy as np

import min_function as mf
import min_instrument as mins
import min_math as mm
import min_plot as mp


# Steady-State Spectroscopy

## Absorption Spectra

### Cary 5000 (agilent)


def ab_merge(fps):
    df_ab_all = pd.DataFrame()
    num_f = len(fps)
    for i in range(num_f):
        fp = fps[i]
        fn = mf.extract_directory_filename(fps[i])[1]
        df = pd.read_csv(
            fp,
            sep=",",
            skiprows=2,
            names=[f"wl_nm", f"{fn}"],
            usecols=[0, 1],
        )
        df_ab_all = pd.concat([df_ab_all, df], axis=1)
    return df_ab_all


def ab_merge_classify(filepaths):
    merged_classified_ab_data = pd.DataFrame()
    file_amounts = len(filepaths)
    for i in range(file_amounts):
        rawname = mf.extract_directory_filename(filepaths[i])[1]
        sample_solvent = mf.join_str(rawname.split("_")[0:2])
        simplename = mf.simplify_filename(rawname)
        header = pd.MultiIndex.from_arrays(
            [
                [f"{sample_solvent}_ab", f"{sample_solvent}_ab"],
                [f"ab_wl_nm_{2*i}", f"{rawname}_{2*i+1}"],
                [f"ab_wl_nm_{2*i}", f"{simplename}_{2*i+1}"],
            ],
            names=["sample", "condition", "name"],
        )
        current_data = np.genfromtxt(
            filepaths[i], skip_header=2, usecols=(0, 1), delimiter=","
        )
        current_data = pd.DataFrame(current_data, columns=header)
        merged_classified_ab_data = pd.concat(
            [merged_classified_ab_data, current_data], axis=1
        )
    return merged_classified_ab_data


## Photoluminescence Spectra

### Fluorolog 3 (horiba)


### FS5 (edinburgh instruments)


def em_merge(filepaths):
    merged_em_data = pd.DataFrame()
    file_amounts = len(filepaths)
    for i in range(file_amounts):
        current_filepath = filepaths[i]
        filename = mf.extract_directory_filename(filepaths[i])[1]
        current_data = pd.read_csv(
            current_filepath,
            sep=",",
            skiprows=22,
            usecols=[0, 1],
            names=[f"em_wl_nm", f"{filename}"],
        )
        merged_em_data = pd.concat([merged_em_data, current_data], axis=1)
    return merged_em_data




def em_merge_classify(filepaths):
    merged_classified_em_data = pd.DataFrame()
    file_amounts = len(filepaths)
    for i in range(file_amounts):
        rawname = mf.extract_directory_filename(filepaths[i])[1]
        sample_solvent = mf.join_str(rawname.split("_")[0:2])
        simplename = mf.simplify_filename(rawname)
        header = pd.MultiIndex.from_arrays(
            [
                [f"{sample_solvent}_em", f"{sample_solvent}_em"],
                [f"ex_wl_nm_{2*i}", f"{rawname}_{2*i+1}"],
                [f"ex_wl_nm_{2*i}", f"{simplename}_{2*i+1}"],
            ],
            names=["sample", "condition", "name"],
        )
        current_data = np.genfromtxt(
            filepaths[i], skip_header=22, usecols=(0, 1), delimiter=","
        )
        current_data = pd.DataFrame(current_data, columns=header)
        merged_classified_em_data = pd.concat(
            [merged_classified_em_data, current_data], axis=1
        )
    return merged_classified_em_data


## Excitation Spectra

### FS5 (edinburgh instruments)


def ex_merge(filepaths):
    merged_ex_data = pd.DataFrame()
    file_amounts = len(filepaths)
    for i in range(file_amounts):
        current_filepath = filepaths[i]
        filename = mf.extract_directory_filename(filepaths[i])[1]
        current_data = pd.read_csv(
            current_filepath,
            sep=",",
            skiprows=22,
            usecols=[0, 1],
            names=["ex_wl_nm", filename],
        )
        merged_ex_data = pd.concat([merged_ex_data, current_data], axis=1)
    return merged_ex_data


def ex_merge_classify(filepaths):
    merged_classified_ex_data = pd.DataFrame()
    file_amounts = len(filepaths)
    for i in range(file_amounts):
        rawname = mf.extract_directory_filename(filepaths[i])[1]
        sample_solvent = mf.join_str(rawname.split("_")[0:2])
        simplename = mf.simplify_filename(rawname)
        header = pd.MultiIndex.from_arrays(
            [
                [f"{sample_solvent}_ex", f"{sample_solvent}_ex"],
                [f"ex_wl_nm_{2*i}", f"{rawname}_{2*i+1}"],
                [f"ex_wl_nm_{2*i}", f"{simplename}_{2*i+1}"],
            ],
            names=["sample", "condition", "name"],
        )
        current_data = np.genfromtxt(
            filepaths[i], skip_header=22, usecols=(0, 1), delimiter=","
        )
        current_data = pd.DataFrame(current_data, columns=header)
        merged_classified_ex_data = pd.concat(
            [merged_classified_ex_data, current_data], axis=1
        )
    return merged_classified_ex_data


# ################### FS5 ###################

# def fs5_merge_classify(filepaths):
#     merged_classified_fs5_data = pd.DataFrame()
#     for i in range(len(filepaths)):
#         condition = mf.extract_directory_filename(filepaths[i])[1]
#         sample = mf.join_str(condition.split("_")[0:2])
#         name = mf.simplify_filename(condition)

#         probe_mode = mf.judge_probe_mode(condition)
#         if len(probe_mode) == 1:
#             if probe_mode[0] == "em":
#                 header = pd.MultiIndex.from_arrays(
#                     [
#                         [sample, sample],
#                         [f"em_wl_nm_{2*i}", f"{condition}_{2*i+1}"],
#                         [f"em_wl_nm_{2*i}", f"{name}_{2*i+1}"],
#                     ],
#                     names=["sample", "condition", "name"],
#                 )
#                 current_data = np.genfromtxt(
#                     filepaths[i], skip_header=22, usecols=(0, 1), delimiter=","
#                 )
#                 current_data = pd.DataFrame(current_data, columns=header)
#             elif probe_mode[0] == "ex":
#                 header = pd.MultiIndex.from_arrays(
#                     [
#                         [sample, sample],
#                         [f"ex_wl_nm_{2*i}", f"{condition}_{2*i+1}"],
#                         [f"ex_wl_nm_{2*i}", f"{name}_{2*i+1}"],
#                     ],
#                     names=["sample", "condition", "name"],
#                 )
#                 current_data = np.genfromtxt(
#                     filepaths[i], skip_header=22, usecols=(0, 1), delimiter=","
#                 )
#                 current_data = pd.DataFrame(current_data, columns=header)
#             elif probe_mode[0] == "tcspc":
#                 header = pd.MultiIndex.from_arrays(
#                     [
#                         [sample, sample],
#                         [f"tcspc_t_ns_{2*i}", f"{condition}_{2*i+1}"],
#                         [f"tcspc_t_ns_{2*i}", f"{name}_{2*i+1}"],
#                     ],
#                     names=["sample", "condition", "name"],
#                 )
#                 current_data = np.genfromtxt(
#                     filepaths[i], skip_header=10, usecols=(0, 1), delimiter=","
#                 )
#                 current_data = pd.DataFrame(current_data, columns=header)
#             else:
#                 print("The detected mode is not 'em', 'ex' or 'tcspc'.")
#         else:
#             print(f"Detected {len(probe_mode)} modes in {condition}.")
#         merged_classified_fs5_data = pd.concat(
#             [merged_classified_fs5_data, current_data], axis=1
#         )

#     (output_folder, sample) = mf.output_folder_sample()
#     if output_folder == "input folder":
#         output_folder = mf.extract_directory_filename(filepaths[0])[0]
#     merged_classified_fs5_data.to_csv(
#         f"{output_folder}/{sample}_fs5_merged_classified.csv",
#         header=True,
#         index=None,
#         sep=",",
#     )
#     return merged_classified_fs5_data


# Time-Resolved Spectrascopy

## General Data Process


def tailor_trspectra(df_1col, t_limit, wl_limit):
    # columns.values and index.values should be float or int
    df = df_1col.loc[
        (df_1col.index >= wl_limit[0]) & (df_1col.index <= wl_limit[1]),
        (df_1col.columns >= t_limit[0]) & (df_1col.columns <= t_limit[1]),
    ]
    return df


def extract_spectra_trspectra(
    df,
    delay_time_series,
):
    chosen_spectra = pd.DataFrame()
    for delay_time in delay_time_series:
        closest_delay_time = mf.find_closest_value(
            df.columns.values.tolist(), delay_time
        )
        chosen_spectrum = df[closest_delay_time]
        chosen_spectra = pd.concat([chosen_spectra, chosen_spectrum], axis=1)
    return chosen_spectra


def extract_kinetics_trspectra(df_1col, list_wl):
    chosen_kinetics = pd.DataFrame()
    for wavelength in list_wl:
        closest_wavelength = mf.find_closest_value(
            df_1col.index.values.tolist(), wavelength
        )
        chosen_kinetic = df_1col.loc[closest_wavelength]
        chosen_kinetics = pd.concat([chosen_kinetics, chosen_kinetic], axis=1)
    return chosen_kinetics


def extract_kinetics1col_trspectra(df, wavelengths: list):
    chosen_kinetics = pd.DataFrame()
    for wavelength in wavelengths:
        closest_wavelength = mf.find_closest_value(df.index.values.tolist(), wavelength)
        chosen_kinetic = df.loc[closest_wavelength]
        chosen_kinetics = pd.concat([chosen_kinetics, chosen_kinetic], axis=1)
    return chosen_kinetics


def extract_1colkinetics_trspectra(df, wavelengths: list):
    chosen_kinetics = pd.DataFrame()
    for wavelength in wavelengths:
        closest_wavelength = mf.find_closest_value(df.index.values.tolist(), wavelength)
        chosen_kinetic = df.loc[closest_wavelength]
        chosen_kinetics = pd.concat([chosen_kinetics, chosen_kinetic], axis=1)
    return chosen_kinetics


def extract_kinetics2col_trspectra(df_trspectra: pd.DataFrame, wavelengths: list):
    df_kinetics = pd.DataFrame()
    for wavelength in wavelengths:
        closest_wavelength = mf.find_closest_value(
            df_trspectra.index.values.tolist(), wavelength
        )
        chosen_kinetic = df_trspectra.loc[closest_wavelength]
        df_kinetic = pd.DataFrame(
            {
                "time": list(chosen_kinetic.index.values),
                f"{closest_wavelength}nm": list(chosen_kinetic.values),
            }
        )
        df_kinetics = pd.concat([df_kinetics, df_kinetic], axis=1)
    return df_kinetics


def extract_2colkinetics_trspectra(df_trspectra, wavelengths):
    df_kinetics = pd.DataFrame()
    for wavelength in wavelengths:
        closest_wavelength = mf.find_closest_value(
            df_trspectra.index.values.tolist(), wavelength
        )
        chosen_kinetic = df_trspectra.loc[closest_wavelength]
        df_kinetic = pd.DataFrame(
            {
                "time_ps": list(chosen_kinetic.index.values),
                closest_wavelength: list(chosen_kinetic.values),
            }
        )
        df_kinetics = pd.concat([df_kinetics, df_kinetic], axis=1)
    return df_kinetics


## Transient Absorption Spectra

### ns-TA

#### LP920 (Edinburgh Instruments)


def wash_nsta_v1(filepaths):
    for filepath in filepaths:
        data = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
        # display(data)
        data.dropna(axis=1, inplace=True)
        # if data.columns[-1] == "":
        #     data = data.drop(data.columns[-1], axis=1)
        # display(data)
        data.index = data.index.str.replace(",", ".").astype(float)
        # data.index = data.index.map('{:.2f}'.format)
        for col in data.columns:
            data[col] = data[col].str.replace(",", ".").astype(float)
            data[col] = data[col] * 1000
        # display(data)
        display(type(data.columns[0]))
        display(data.columns)
        try:
            data.columns = [int(col) for col in data.columns]
            data = data.sort_index(axis=1)
            # print(data)
        except ValueError:
            pass
        # display(type(data.columns[0]))
        # display(data.columns)
        # if sortcol is True:
        #     data.columns = [int(col) for col in data.columns]
        #     data = data.sort_index(axis=1)
        newfilepath = filepath.replace(".txt", ".csv")
        data.to_csv(newfilepath, sep=",", index=True)


def wash_nsta(
    filepaths,
    sortcol=True,
):
    for filepath in filepaths:
        data = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
        if data.columns[-1] == "":
            data = data.drop(data.columns[-1], axis=1)
        if sortcol is True:
            data.columns = [int(col) for col in data.columns]
            data = data.sort_index(axis=1)
        data.index = data.index.str.replace(",", ".").astype(float)
        for col in data.columns:
            data[col] = data[col].str.replace(",", ".").astype(float)
            data[col] = data[col] * 1000
        newfilepath = filepath.replace(".txt", ".csv")
        data.to_csv(newfilepath, sep=",", index=True)


def wash_nsta_spectra(
    filepaths,
    sortcol=True,
):
    for filepath in filepaths:
        data = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
        if data.columns[-1] == "":
            data = data.drop(data.columns[-1], axis=1)
        if sortcol is True:
            data.columns = [int(col) for col in data.columns]
            data = data.sort_index(axis=1)
        data.index = data.index.str.replace(",", ".").astype(float)
        for col in data.columns:
            data[col] = data[col].str.replace(",", ".").astype(float)
            data[col] = data[col] * 1000  # OD to mOD
        newfilepath = filepath.replace(".txt", ".csv")
        data.to_csv(newfilepath, sep=",", index=True)


# def wash_nsta_spectra(filepaths):
#     for filepath in filepaths:
#         data = pd.read_csv(filepath, sep="\t")  # tab separated txt file
#         newdata = pd.DataFrame()
#         for col in data.columns:
#             if "ns" in col:
#                 new_col = col.replace("ns", "")
#             elif "us" in col:
#                 new_col = float(col.replace("us", "")) * 1000
#             elif "ms" in col:
#                 new_col = float(col.replace("ms", "")) * 1000000
#             current30 = data[col].str.replace(",", ".").astype(float)

#         data.iloc[:, 1:] = data.iloc[:, 1:] * 1000  # OD to mOD
#         newfilepath = filepath.replace(".txt", ".csv")
#         data.to_csv(newfilepath, sep=",", index=False)


def wash_nsta_kinetics(filepaths, sortcol=True):
    for filepath in filepaths:
        data = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
        if data.columns[-1] == "":
            data = data.drop(data.columns[-1], axis=1)
        if sortcol is True:
            data.columns = [int(col) for col in data.columns]
            data = data.sort_index(axis=1)
        data.index = data.index.str.replace(",", ".").astype(float)
        for col in data.columns:
            data[col] = data[col].str.replace(",", ".").astype(float)
            data[col] = data[col] * 1000  # OD to mOD
        newfilepath = filepath.replace(".txt", ".csv")
        data.to_csv(newfilepath, sep=",", index=True)


def wash_nsta_kinetics(filepaths):
    for filepath in filepaths:
        data = pd.read_csv(filepath, sep="\t", header=0, index_col=0, na_values=" ")
        display(data)
        data.dropna(axis=0, how="all", inplace=True)
        data.dropna(axis=1, how="all", inplace=True)
        display(data)
        data.index = data.index.str.replace(",", ".").astype(float)
        # data.index = data.index.map('{:.2f}'.format)
        for col in data.columns:
            data[col] = data[col].str.replace(",", ".").astype(float)
            data[col] = data[col] * 1000  # mOD to OD
        display(data)
        try:
            data.columns = [int(col) for col in data.columns]
            data = data.sort_index(axis=1)
        except ValueError:
            data = data.sort_index(axis=1)
            pass
        display(data)
        newfilepath = filepath.replace(".txt", ".csv")
        data.to_csv(newfilepath, sep=",", index=True)


# def wash_nsta_v2(filepaths):
#     for filepath in filepaths:
#         data = pd.read_csv(filepath, sep="\t", header=0, index_col=0, na_values=" ")
#         display(data)
#         data.dropna(axis=0, how="all", inplace=True)
#         data.dropna(axis=1, how="all", inplace=True)
#         display(data)
#         data.index = data.index.str.replace(",", ".").astype(float)
#         # data.index = data.index.map('{:.2f}'.format)
#         for col in data.columns:
#             data[col] = data[col].str.replace(",", ".").astype(float)
#             data[col] = data[col] * 1000  # mOD to OD
#         display(data)
#         try:
#             data.columns = [int(col) for col in data.columns]
#             data = data.sort_index(axis=1)
#         except ValueError:
#             data = data.sort_index(axis=1)
#             pass
#         display(data)
#         newfilepath = filepath.replace(".txt", ".csv")
#         data.to_csv(newfilepath, sep=",", index=True)


# def wash_nspl_kinetics(filepaths):
#     for filepath in filepaths:
#         data = pd.read_csv(filepath, sep="\t", header=0, index_col=0, na_values=" ")
#         display(data)
#         data.dropna(axis=0, how="all", inplace=True)
#         data.dropna(axis=1, how="all", inplace=True)
#         display(data)
#         data.index = data.index.str.replace(",", ".").astype(float)
#         # data.index = data.index.map('{:.2f}'.format)
#         for col in data.columns:
#             data[col] = data[col].str.replace(",", ".").astype(float)
#             data[col] = data[col] * 1000  # V to mV
#         display(data)
#         try:
#             data.columns = [int(col) for col in data.columns]
#             data = data.sort_index(axis=1)
#         except ValueError:
#             data = data.sort_index(axis=1)
#             pass
#         display(data)
#         newfilepath = filepath.replace(".txt", ".csv")
#         data.to_csv(newfilepath, sep=",", index=True)


def load_nsta_spectra(filepath):
    if filepath.endswith(".csv"):
        df_1col = pd.read_csv(filepath, header=0, index_col=0, sep=",")
    df_1col.columns = df_1col.columns.astype(float)
    # df_1col.columns = df_1col.columns.astype(int)
    df_1col.columns.name = None
    df_1col.index = df_1col.index.astype(float)
    return df_1col


def load_nsta(filepath):
    if filepath.endswith(".csv"):
        df_1col = pd.read_csv(filepath, header=0, index_col=0, sep=",")
    # df_1col.columns = df_1col.columns.astype(float)
    try:
        df_1col.columns = [int(col) for col in df_1col.columns]
        # df_1col = df_1col.sort_index(axis=1)
        # print(data)
    except ValueError:
        pass
    # df_1col.columns = df_1col.columns.astype(int)
    df_1col.columns.name = None
    df_1col.index = df_1col.index.astype(float)
    return df_1col


### fs-TA

#### Usual Data Processing


def writeString(ufsfile, string):
    ufsfile.write(struct.pack(">I", len(string)))
    ufsfile.write(struct.pack(">{}s".format(len(string)), string))


def writeUInt(ufsfile, value):
    ufsfile.write(struct.pack(">I", value))


def writeDouble(ufsfile, value):
    ufsfile.write(struct.pack(">d", value))


def writeDoubles(ufsfile, values):
    ufsfile.write(struct.pack(">{}d".format(len(values)), *values))


def readString(filedata, cursor):
    string_length = struct.unpack_from(">I", filedata, offset=cursor)
    cursor += 4
    string = struct.unpack_from(
        ">{}s".format(string_length[0]), filedata, offset=cursor
    )
    cursor += string_length[0]
    return (string[0].decode("utf8"), cursor)


def readUInt(filedata, cursor):
    number = struct.unpack_from(">I", filedata, offset=cursor)
    cursor += 4
    return (number[0], cursor)


def readDouble(filedata, cursor):
    number = struct.unpack_from(">d", filedata, offset=cursor)
    cursor += 8
    return (number[0], cursor)


def readDoubles(filedata, cursor, count):
    numbers = struct.unpack_from(">{}d".format(count), filedata, offset=cursor)
    cursor += 8 * count
    return (list(numbers), cursor)


def load_fsta(filepath):
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, header=0, index_col=0, na_values="NaN", sep=",")
    elif filepath.endswith(".tsv"):
        # input_file = "euc0a1_meoh_ar_fsta_240314.tsv"
        # output_file = "euc0a1_meoh_ar_fsta_240314.csv"
        # with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        #     for line in f_in:
        #         data = line.strip().split()
        #         data_float = [float(item) for item in data]
        #         f_out.write(",".join(map(str, data_float)) + "\n")
        df = pd.read_csv(filepath, header=0, index_col=0, na_values="NaN", sep="\t", dtype=float)
    elif filepath.endswith(".dat"):
        raw_df = pd.read_csv(
            filepath, sep=r"\s+", lineterminator="\n", engine="python", header=None
        )
        df = pd.DataFrame(
            data=raw_df.iloc[1:, 1:].values,
            columns=raw_df.iloc[0, 1:],
            index=raw_df.iloc[1:, 0],
        )
        df = df.Te
    elif filepath.endswith(".ufs"):
        file = open(filepath, mode="rb")
        filedata = file.read()
        cursor = 0x0
        (version, cursor) = readString(filedata, cursor)
        (wl_axis_label, cursor) = readString(filedata, cursor)
        (wl_axis_units, cursor) = readString(filedata, cursor)
        (wl_axis_count, cursor) = readUInt(filedata, cursor)
        (wl_axis_data, cursor) = readDoubles(filedata, cursor, wl_axis_count)
        (t_axis_label, cursor) = readString(filedata, cursor)
        (t_axis_units, cursor) = readString(filedata, cursor)
        (t_axis_count, cursor) = readUInt(filedata, cursor)
        (t_axis_data, cursor) = readDoubles(filedata, cursor, t_axis_count)
        (data_label, cursor) = readString(filedata, cursor)
        (data_size0, cursor) = readUInt(filedata, cursor)
        (data_size1, cursor) = readUInt(filedata, cursor)
        (data_size2, cursor) = readUInt(filedata, cursor)
        data_matrix = []
        for row in range(data_size1):
            (row_data, cursor) = readDoubles(filedata, cursor, data_size2)
            row_data.insert(0, round(wl_axis_data[row], 1))
            row_data = [round(x, 7) for x in row_data]
            data_matrix.append(row_data)
        # (metadata, cursor) = readString(filedata, cursor)
        t_axis_data.insert(0, 0)
        data_matrix.insert(0, t_axis_data)
        data_ndarray = np.array(data_matrix)
        df = pd.DataFrame(
            data=data_ndarray[1:, 1:],
            index=data_ndarray[1:, 0],
            columns=data_ndarray[0, 1:],
        )
    df.columns = [float(i) for i in df.columns]
    df.columns.name = None
    df.index = [float(i) for i in df.index]
    df.index.name = None
    return df




def tailor_fsta(df, wl_range=[320, 760], t_range=[-100, 8000]):
    # df_crop = df[wl_range[0]:wl_range[1]]
    # df_crop = df.loc[(df.index >= wl_range[0]) & (df.index <= wl_range[1])]
    # df_crop = df_crop[df_crop.columns[(df_crop.columns >= t_range[0]) & (df_crop.columns <= t_range[1])]]
    df_crop = df.loc[
        (df.index >= wl_range[0]) & (df.index <= wl_range[1]),
        (df.columns >= t_range[0]) & (df.columns <= t_range[1]),
    ]
    return df_crop


# def replace_excitation(df, range=[345, 365]):
#     new_df = deepcopy(df)
#     new_df.loc[df.index[(df.index > range[0])&(df.index < range[1])]] = np.nan
#     return new_df


def replace_excitation(df, range=[345, 365]):
    df.loc[df.index[(df.index > range[0]) & (df.index < range[1])]] = np.nan
    return df


def remove_excitation(df, range=[345, 365]):
    df = df.drop(df.index[(df.index > range[0]) & (df.index < range[1])])
    return df


def count_nan(df):
    nan_locations = np.where(df.isna())
    nan_amount = len(nan_locations[0])
    return nan_amount


def repair_nan(df, mode="row"):
    if df.isna().any().any():
        locations = np.where(df.isna())
        print(f"Misssed {len(locations[0])} points.")
        data = deepcopy(df)
        for i in range(len(locations[0])):
            row = locations[0][i]
            column = locations[1][i]
            if mode == "row":
                if column == 0:
                    data.iloc[row, column] = data.iloc[row, column + 1]
                elif column == len(data.columns) - 1:
                    data.iloc[row, column] = data.iloc[row, column - 1]
                else:
                    data.iloc[row, column] = (
                        data.iloc[row, column - 1] + data.iloc[row, column + 1]
                    ) / 2
            elif mode == "column":
                if row == 0:
                    data.iloc[row, column] = data.iloc[row + 1, column]
                elif row == len(data.index) - 1:
                    data.iloc[row, column] = data.iloc[row - 1, column]
                else:
                    data.iloc[row, column] = (
                        data.iloc[row - 1, column] + data.iloc[row + 1, column]
                    ) / 2
    else:
        print("Return raw dataframe because no NaN values detected.")
        data = deepcopy(df)
    return data


def repair_outlier_fsta(df):
    data = deepcopy(df)
    for i in range(len(data.index)):  # index each kinetics (wavelength)
        kinetic = data.iloc[i]
        for j in range(len(kinetic.index)):  # index each time point
            measured_value = kinetic[kinetic.index[j]]
            if j == 0:
                averaged_value = kinetic[kinetic.index[j + 1]]
            elif j == len(kinetic.index) - 1:
                averaged_value = kinetic[kinetic.index[j - 1]]
            else:
                averaged_value = (
                    kinetic[kinetic.index[j + 1]] + kinetic[kinetic.index[j - 1]]
                ) / 2
            diff_od = abs(measured_value - averaged_value)
            # if diff_od > abs(0.1 * averaged_value):
            if diff_od > 0.0001:
                data.iloc[i, j] = averaged_value
    return data


def compare_scan_fsta(scan0, scan1):
    # compare two scans
    scan_diff = pd.DataFrame(
        data=scan1.values - scan0.values,
        index=scan0.index,
        columns=scan0.columns,
    )
    return scan_diff


def reaverage_fsta(df_list):
    # re-average the fsta files in the df_list, time axis is averaged while wavelength axis is kept
    delta_a_all = []
    time_all = []
    for df in df_list:
        delta_a = df.values
        delta_a_all.append(delta_a)
        times = [float(time) for time in df.columns.values]
        time_all.append(times)
    delta_a_average = (np.sum(delta_a_all, axis=0)) / len(delta_a_all)
    time_average = (np.sum(time_all, axis=0)) / len(time_all)
    averaged_fsta = pd.DataFrame(
        data=delta_a_average, index=df_list[0].index, columns=time_average
    )
    return averaged_fsta


def subtract_background_fsta(df, background_range=[-99, -4]):
    # construct the baseline from the background region and subtract it from the fsta data
    # background = df[background_range[0]:background_range[1]]
    background = df.loc[:, background_range[0] : background_range[1]]
    baseline = np.mean(background, axis=1)
    df_subtracted = df.subtract(baseline, axis=0)
    return df_subtracted


# def smooth_fsta(mode="kinetics", ):
#     # smooth each kinetic or spectrum
# if mode == "kinetics":
#     abc
# elif mode == "spectra":
#     pass


#### CarpetView (Light Conversion)


def writeCVTsv(df, filepath):
    # write the fs-TA data to a file in tsv (tab separated values) format for CarperView
    df = df.T
    if df.isna().any().any():
        locations = np.where(df.isna())
        print(
            f"\033[1m{len(locations)} NaN points\033[0m detected, please repair them first."
        )
    else:
        df.to_csv(
            filepath,
            header=True,
            index=True,
            index_label=0,
            sep="\t",
            float_format="%.4f",
        )


def readCVDat(filepath):
    data = pd.read_table(
        filepath, header=None, names=None, sep=r"\s{3,4}", engine="python"
    )
    data = data.T
    new_data = pd.DataFrame(
        data=data.iloc[1:, 1:].values, columns=data.iloc[0, 1:], index=data.iloc[1:, 0]
    )
    return new_data


def CVDat2Csv(filepaths):
    for filepath in filepaths:
        new_filepath = filepath.replace(".dat", ".csv")
        data = readCVDat(filepath)
        data.to_csv(new_filepath, header=True, index=True, index_label=0, sep=",")


#### FemtoSuite


def writeFSCsv(df, filepath):
    df.to_csv(
        filepath, header=True, index=True, index_label=0, sep=",", float_format="%.4f"
    )


#### Surface Xplorer (Ultrafast System)


def writeSXUfs(
    df,
    filepath,
    # metadata
):
    # write the fs-TA data to a file in UFS format (binary file)
    # metadata contained the information you want to show in the bottom right of Surface Xplorer
    version = b"Version2"
    wl_axis_label = b"Wavelength"
    wl_axis_units = b"nm"
    wl_axis_data = df.index.values.tolist()
    wl_axis_count = len(wl_axis_data)
    t_axis_label = b"Time"
    t_axis_units = b"ps"
    t_axis_data = df.columns.values.tolist()
    t_axis_data = [float(i) for i in t_axis_data]
    t_axis_count = len(t_axis_data)
    deltaA_label = b"dA"
    deltaA_matrix = df.values.tolist()
    # metadata = metadata
    # metadata = bytearray(metadata, "ascii")
    ufsfile = open(filepath, "wb")
    writeString(ufsfile, version)
    writeString(ufsfile, wl_axis_label)
    writeString(ufsfile, wl_axis_units)
    writeUInt(ufsfile, wl_axis_count)
    writeDoubles(ufsfile, wl_axis_data)
    writeString(ufsfile, t_axis_label)
    writeString(ufsfile, t_axis_units)
    writeUInt(ufsfile, t_axis_count)
    writeDoubles(ufsfile, t_axis_data)
    writeString(ufsfile, deltaA_label)
    writeUInt(ufsfile, 0)
    writeUInt(ufsfile, wl_axis_count)
    writeUInt(ufsfile, t_axis_count)
    for row in range(wl_axis_count):
        writeDoubles(ufsfile, deltaA_matrix[row])
    # writeString(ufsfile, metadata)


#### Data Conversion


def csv_convert_ufs(filepaths):
    # convert the csv files to ufs files
    for filepath in filepaths:
        new_filepath = filepath.replace(".csv", ".ufs")
        data = load_fsta(filepath)
        writeSXUfs(data, new_filepath, "")


def tsv_convert_ufs(filepaths):
    for filepath in filepaths:
        new_filepath = filepath.replace(".tsv", ".ufs")
        data = load_fsta(filepath)
        writeSXUfs(data, new_filepath, "")


def ufs_convert_csv(filepaths):
    for filepath in filepaths:
        new_filepath = filepath.replace(".ufs", ".csv")
        data = load_fsta(filepath)
        data.to_csv(
            new_filepath, header=True, index=True, index_label=0, na_rep="NaN", sep=","
        )


def ufs_convert_tsv(filepaths):
    for filepath in filepaths:
        new_filepath = filepath.replace(".ufs", ".tsv")
        data = load_fsta(filepath)
        writeCVTsv(data, new_filepath)


def transpose_csv(filename_type):
    filename = filename_type.split(".")
    del filename[-1]
    filename = filename[0]
    with open(f"{filename}.csv", "r") as raw_data:
        data = np.loadtxt(raw_data, delimiter=",")
    data_transposed = np.transpose(data)
    np.set_printoptions(suppress=True)
    np.savetxt(
        f"{filename}_transpose.csv",
        data_transposed,
        fmt="%s",
        delimiter=",",
        newline="\n",
    )


#### Glotaran


def writeGlotaran(df, filepath):
    df.to_csv(filepath, header=True, index=True, index_label=0, sep=",")


## Transient Emission Spectra

### TCSPC

#### FS5 (edinburgh instruments)


def tcspc_merge(filepaths):
    merged_tcspc_data = pd.DataFrame()
    fileamounts = len(filepaths)
    for i in range(fileamounts):
        current_filepath = filepaths[i]
        filename = mf.extract_directory_filename(filepaths[i])[1]
        current_data = pd.read_csv(
            current_filepath,
            sep=",",
            skiprows=10,
            usecols=[0, 1],
            names=["tcspc_t_ns", filename],
        )
        merged_tcspc_data = pd.concat([merged_tcspc_data, current_data], axis=1)
    return merged_tcspc_data


def tcspc_merge_classify(filepaths):
    merged_classified_tcspc_data = pd.DataFrame()
    file_amounts = len(filepaths)
    for i in range(file_amounts):
        rawname = mf.extract_directory_filename(filepaths[i])[1]
        sample_solvent = mf.join_str(rawname.split("_")[0:2])
        simplename = mf.simplify_filename(rawname)
        header = pd.MultiIndex.from_arrays(
            [
                [f"{sample_solvent}_tcspc", f"{sample_solvent}_tcspc"],
                [f"tcspc_wl_nm_{2*i}", f"{rawname}_{2*i+1}"],
                [f"tcspc_wl_nm_{2*i}", f"{simplename}_{2*i+1}"],
            ],
            names=["sample", "condition", "name"],
        )
        current_data = np.genfromtxt(
            filepaths[i], skip_header=10, usecols=(0, 1), delimiter=","
        )
        current_data = pd.DataFrame(current_data, columns=header)
        merged_classified_tcspc_data = pd.concat(
            [merged_classified_tcspc_data, current_data], axis=1
        )
    return merged_classified_tcspc_data


### ns-PL

#### LP920 (Edinburgh Instruments)
