import os, glob, inspect, re
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, tkinter as tk, plotly as py, plotly.graph_objects as go, plotly.io as pio, scipy.integrate as integrate
from tkinter import filedialog
from plotly.subplots import make_subplots
from scipy.integrate import simps
import matplotlib.pyplot as plt


### String


def variable_name(variable, variable_space):
    return [
        variable_name
        for variable_name in variable_space
        if variable_space[variable_name] is variable
    ]


def find_str(filename_type, string):
    linenum = []
    with open(filename_type) as f:
        current_linenum = 0
        for line in f.readlines():
            current_linenum += 1
            if string in line:
                linenum.append(current_linenum)
    return linenum


def join_str(list, separator="_"):
    return separator.join(list)


def del_character_af_specific(
    string, specific_string
):  # Delete all characters after a specific character/string in a string
    if specific_string in string:
        index = string.index(specific_string)
        new_string = string[: index + len(specific_string)]
    else:
        print("No specific string found!")
    return new_string


def judge_probe_mode(filename):
    filename = filename.split("_")
    possible_modes = (
        "ab",
        "em",
        "ex",
        "tcspc",
        "qy",
        "nstavis",
        "nstanir",
        "nsplvis",
        "fstavis",
        "fstanir",
        "fstamir",
    )
    detected_probe_modes = []
    for mode in possible_modes:
        if mode in filename:
            detected_probe_modes.append(mode)
    return detected_probe_modes


def extract_directory_filename(filepath):
    (directory, filename) = os.path.split(filepath)
    filename = filename.split(".")[0]
    return directory, filename


def simplify_filename(filename):
    filename = filename.split("_")
    possible_modes = (
        "ab",
        "em",
        "pl",
        "ex",
        "tcspc",
        "qy",
        "nstavis",
        "nstanir",
        "nsplvis",
        "fsta",
        "fstavis",
        "fstanir",
        "fstamir",
        "nsta",
        "nspl",
    )
    detected_modes = []
    for mode in possible_modes:
        if mode in filename:
            detected_modes.append(mode)
    if len(detected_modes) == 1:
        mode_index = filename.index(detected_modes[0])
        separator = "_"
        simplified_filename = separator.join(filename[0 : mode_index + 1])
        return simplified_filename
    else:
        print(f"Detected {len(detected_modes)} modes, no output!")


def simplify_df_column_names(df):
    new_column_names = []
    for i in range(int(len(df.columns) / 2)):
        new_column_names.append(df.columns[2 * i])
        new_column_names.append(simplify_filename(df.columns[2 * i + 1]))
    new_df = deepcopy(df)
    new_df.columns = new_column_names
    return new_df


def output_folder_sample():
    answer = input("Whether store the processed data in the same folder? (y/n)\n")
    if answer == "y":
        output_folder = "input folder"
    else:
        output_folder = input("Please input the output folder:\n")
    sample = input("\nPlease input the sample name:\n")
    return output_folder, sample


### File


def list_dirpath(directory):
    # list the path of 1st order subdirectories
    subdir_paths = []
    for subdir_subfile_name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir_subfile_name)):
            subdir_paths.append(os.path.join(directory, subdir_subfile_name))
    return subdir_paths


def list_dirname(directory):
    # list the name of 1st order subdirectories
    dir_paths = list_dirpath(directory)
    dir_names = []
    for dir_path in dir_paths:
        dir_names.append(os.path.basename(dir_path))
    return dir_names


def list_filepath(directory, postfix=None):
    # list the path of 1st order subfiles
    all_type_filepaths = []
    for subdir_subfile_name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, subdir_subfile_name)):
            all_type_filepaths.append(os.path.join(directory, subdir_subfile_name))
    specific_type_filepaths = []
    if postfix != None:
        for all_type_filepath in all_type_filepaths:
            if all_type_filepath.endswith(postfix):
                specific_type_filepaths.append(all_type_filepath)
        return specific_type_filepaths
    else:
        return all_type_filepaths


def list_filename(directory, postfix=None):
    file_paths = list_filepath(directory, postfix)
    file_names = []
    for file_path in file_paths:
        file_names.append(os.path.basename(file_path))
    return file_names


def join_folder_filename(folder, filenames):
    filepaths = []
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        filepaths.append(filepath)
    return filepaths


# def list_filename(folder_path, filetype=None):
#     # list the filename of file in 1st order subfiles in a directory
#     file_names = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(filetype):
#             file_names.append(file_name)
#     return file_names


def list_specific_file(folder, keyword):
    # list the 1st order subfiles ended with keyword in a directory
    files = list_file(folder)
    chosen_files = []
    for file in files:
        if file.endswith(keyword):
            chosen_files.append(file)
    return chosen_files


def list_file_tree(folder):
    # retrieve all files in a directory and subdirectories
    all_filepaths = []
    for file_folder in os.listdir(folder):
        path = folder + "/" + file_folder
        if os.path.isfile(path):
            all_filepaths.append(path)
        elif os.path.isdir(path):
            filepaths = list_file_tree(path)
            for filepath in filepaths:
                all_filepaths.append(filepath)
    return all_filepaths


def list_specific_file_tree(folder, keyword):
    # retrieve all files ended with keyword in a directory and subdirectories
    all_type_files = list_file_tree(folder)
    chosen_files = []
    for file in all_type_files:
        if file.endswith(keyword):
            chosen_files.append(file)
    return chosen_files


def copy_paste_files(all_type_files, output_folder):
    # copy and paste all files in a list to a folder
    for file in all_type_files:
        os.system(f"cp {file} {output_folder}")


def del_specific_file(folder, keyword):
    chosen_files = list_specific_file(folder, keyword)
    for file in chosen_files:
        os.remove(file)


def remove_file(folder, keyword):
    chosen_files = list_specific_file(folder, keyword)
    for file in chosen_files:
        os.remove(file)


def del_specific_file_tree(folder, keyword):
    # delete all files ended with keyword in a directory and subdirectories
    all_type_files = list_file_tree(folder)
    for file in all_type_files:
        if file.endswith(keyword):
            os.remove(file)


def remove_file_tree(folder, keyword):
    all_type_files = list_file_tree(folder)
    for file in all_type_files:
        if file.endswith(keyword):
            os.remove(file)


def choose_folder():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    return directory


# def choose_filepath_automatic(filetype=None):
#     folder = choose_folder()
#     filepaths = list_file_path(folder, filetype)
#     return filepaths


# def choose_filename_automatic(filetype=None):
#     folder = choose_folder()
#     filenames = list_file_name(folder, filetype)
#     return filenames


# def choose_file_automatic(filetype="csv"):
#     root = tk.Tk()
#     root.withdraw()
#     directory = filedialog.askdirectory()
#     alltype_files = os.listdir(directory)
#     filetype_names = []
#     for file in alltype_files:
#         if file.endswith(filetype):
#             filetype_names.append(file)
#     filepaths = []
#     for filetype_name in filetype_names:
#         filepaths.append(f"{directory}/{filetype_name}")
#     return directory, filepaths


# def choose_filepath_manual():
#     acquired_filepaths = []
#     active = True
#     while active:
#         root = tk.Tk()
#         root.withdraw()
#         filepath = filedialog.askopenfilename(
#             title="Please choose a file",
#             filetypes=[("CSV Files", "*.csv"), ("All Files", "*")],
#         )
#         if filepath == "":
#             active = False
#         else:
#             acquired_filepaths.append(filepath)
#     print(f"\n{acquired_filepath}\n" for acquired_filepath in acquired_filepaths)
#     return acquired_filepaths


def choose_filepaths(dir=None, filetype=None):
    if dir is None:
        acquired_filepaths = []
        active = True
        while active:
            root = tk.Tk()
            root.withdraw()
            if filetype is None:
                filepath = filedialog.askopenfilename(
                    title="Choose File:",
                    filetypes=[("All Files", "*")],
                )
            else:
                filepath = filedialog.askopenfilename(
                    title="Choose File:",
                    filetypes=[(f"{filetype} Files", f"{filetype}")],
                )
            if filepath == "":
                active = False
            else:
                acquired_filepaths.append(filepath)
        return acquired_filepaths
    else:
        acquired_filepaths = []
        active = True
        while active:
            root = tk.Tk()
            root.withdraw()
            if filetype is None:
                filepath = filedialog.askopenfilename(
                    title="Choose File:",
                    initialdir=dir,
                    filetypes=[("All Files", "*")],
                )
            else:
                filepath = filedialog.askopenfilename(
                    title="Choose File:",
                    initialdir=dir,
                    filetypes=[(f"{filetype} Files", f"{filetype}")],
                )
            if filepath == "":
                active = False
            else:
                acquired_filepaths.append(filepath)
        return acquired_filepaths


def choose_filenames(dir=None, filetype=None):
    filepaths = choose_filepaths(dir=dir, filetype=filetype)
    filenames = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        filenames.append(filename)
    return filenames


#### Data Process


def checklist_ifequal(list):
    first_item = list[0]
    result = True
    for item in list:
        if item == first_item:
            result = True
        elif item != first_item:
            result = False
            break
    if result == True:
        print("All the elements in the list are same.\n")
    else:
        print("Not all the elements in the list are same.\n")
    return result


def change_firstline(filepath):
    with open(filepath) as file:
        lines = file.readlines()
    first_line = lines[0].strip("\n").split(",")
    new_first_line = []
    new_first_line.append(first_line[-1])
    new_first_line = new_first_line * len(first_line)
    new_first_line = join_str(new_first_line, ",")
    new_first_line = new_first_line + "\n"
    lines[0] = new_first_line
    with open(filepath, "w") as file:
        file.writelines(lines)


def choose_spectra_data(df_2cols, wl_limit=None):
    # wl_limit = [lower_limit, upper_limit]
    df_2cols = df_2cols.dropna(axis=0, how="any")
    if wl_limit is None:
        wl_lower_limit = df_2cols.iloc[:, 0].min()
        wl_upper_limit = df_2cols.iloc[:, 0].max()
    else:
        wl_lower_limit = wl_limit[0]
        wl_upper_limit = wl_limit[1]

    chosen_data = df_2cols.loc[
        (df_2cols.iloc[:, 0] >= wl_lower_limit)
        & (df_2cols.iloc[:, 0] <= wl_upper_limit)
    ]
    return chosen_data


def choose_2coldf(df_2cols, wl_limit=None):
    # wl_limit = [lower_limit, upper_limit]
    df_2cols = df_2cols.dropna(axis=0, how="any")
    # if wl_limit is None:
    #     wl_lower_limit = df_2cols.iloc[:, 0].min()
    #     wl_upper_limit = df_2cols.iloc[:, 0].max()
    # else:
    wl_lower_limit = wl_limit[0]
    wl_upper_limit = wl_limit[1]
    chosen_2coldf = df_2cols.loc[
        (df_2cols.iloc[:, 0] >= wl_lower_limit)
        & (df_2cols.iloc[:, 0] <= wl_upper_limit)
    ]
    return chosen_2coldf


def peak_wavelength_intensity(df, wl_lower_upper_limit):
    search_region = choose_spectra_data(df, wl_lower_upper_limit)
    peak_row = df.loc[df.iloc[:, 1].values == search_region.iloc[:, 1].max()]
    peak_wavelength = peak_row.iat[0, 0]
    peak_intensity = peak_row.iat[0, 1]
    return peak_wavelength, peak_intensity


def determine_baseline_value(df, wl_lower_upper_limit):
    chosen_data = choose_spectra_data(df, wl_lower_upper_limit)
    baseline_value = chosen_data.iloc[:, 1].mean()
    return baseline_value


def determine_baseline(df_2col, wl_limit):
    # df_2ol with only two columns
    # wl_limit = [lower_limit, upper_limit]
    chosen_data = choose_spectra_data(df_2col, wl_limit)
    baseline = chosen_data.iloc[:, 1].mean()
    return baseline


def normalize_globalmax(df_2col):
    data = deepcopy(df_2col)
    for i in range(len(data.columns)):
        if (i % 2) != 0:
            data.iloc[:, i] = data.iloc[:, i].div(data.iloc[:, i].max())
    return data


# def normalize_globalmax(df_2col):
#     data = df_2col.copy()
#     for i, col in enumerate(data.columns):
#         if i % 2 != 0 and data[col].max() != 0:
#             data[col] /= data[col].max()
#     return data


# def normalize_globalmax(df, df_type="df_2col"):
#     data = df.copy()
#     if df_type is "df_1col":

#     for i, col in enumerate(data.columns):
#         if i % 2 != 0 and data[col].max() != 0:
#             data[col] /= data[col].max()
#     return data


def normalize_wlmax(df, wl_limit):  # wl_limit = [lower_limit, upper_limit]
    data = deepcopy(df)
    for i in range(len(df.columns)):
        if (i % 2) == 0:
            data.iloc[:, i] = df.iloc[:, i]
        else:
            local_maximum = (
                (df.iloc[:, i])
                .loc[
                    (df.iloc[:, i - 1] > wl_limit[0])
                    & (df.iloc[:, i - 1] < wl_limit[1])
                ]
                .max()
            )
            data.iloc[:, i] = (df.iloc[:, i]).div(local_maximum)
    return data


# def normalize_localmax(df_2col, range): # single 2coldf
#     local_maximum = (df_2col.iloc[:,1]).loc[
#         (df_2col.iloc[:, 0] > range[0]) & (df_2col.iloc[:, 0] < range[1])
#     ].max()
#     df_2col.iloc[:, 1] = (df_2col.iloc[:, 1]).div(local_maximum)
#     return df_2col


def normalize_localmax(df_2col, limit):
    data = deepcopy(df_2col)
    for i in range(int(len(data.columns) / 2)):
        df = data.iloc[:, 2 * i : 2 * i + 2]
        local_maximum = (
            (df.iloc[:, 1])
            .loc[(df.iloc[:, 0] > limit[0]) & (df.iloc[:, 0] < limit[1])]
            .max()
        )
        data.iloc[:, 2 * i + 1] = (data.iloc[:, 2 * i + 1]).div(local_maximum)
    return data


def normalize(df, max_lower_limit, max_upper_limit, min_lower_limit, min_upper_limit):
    # normalize a dataframe with two columns, first column is wavelength, second column is intensity/absorbance
    data = deepcopy(df)
    for i in range(len(data.columns)):
        if (i % 2) == 0:
            data.iloc[:, i] = data.iloc[:, i]
        else:
            maximum = (
                (data.iloc[:, i])
                .loc[
                    (data.iloc[:, i - 1] > max_lower_limit)
                    & (data.iloc[:, i - 1] < max_upper_limit)
                ]
                .max()
            )
            minimum = (
                (data.iloc[:, i])
                .loc[
                    (data.iloc[:, i - 1] > min_lower_limit)
                    & (data.iloc[:, i - 1] < min_upper_limit)
                ]
                .mean()
            )
            data.iloc[:, i] = (data.iloc[:, i]).sub(minimum).div(maximum - minimum)
    return data


def normalize_kinetics_col_absmax(df):
    # df.index is the x axis, each column is a set of data
    data = deepcopy(df)
    for col in data.columns:
        col_max = data[col].max()
        abs_col_max = abs(col_max)
        col_min = data[col].min()
        abs_col_min = abs(col_min)
        if abs_col_max > abs_col_min:
            data[col] = data[col] / (abs_col_max)
        elif abs_col_max < abs_col_min:
            data[col] = data[col] / (abs_col_min)
    return data


def normalize_kinetics_base_fit(df):
    # a set of data consists of four columns, the first two columns are raw data, the other two columns are fitted data
    data = deepcopy(df)
    for i in range(int(len(data.columns) / 4)):
        max_fit = data.iloc[:, 4 * i + 3].max()
        abs_max_fit = abs(max_fit)
        min_fit = data.iloc[:, 4 * i + 3].min()
        abs_min_fit = abs(min_fit)
        if abs_max_fit > abs_min_fit:
            data.iloc[:, 4 * i + 1] = data.iloc[:, 4 * i + 1].div(abs_max_fit)
            data.iloc[:, 4 * i + 3] = data.iloc[:, 4 * i + 3].div(abs_max_fit)
        elif abs_max_fit < abs_min_fit:
            data.iloc[:, 4 * i + 1] = data.iloc[:, 4 * i + 1].div(abs_min_fit)
            data.iloc[:, 4 * i + 3] = data.iloc[:, 4 * i + 3].div(abs_min_fit)
    return data


def average_dataframe(df_list):
    # list_df is a list of dataframes
    df_all = pd.concat(df_list)
    df_mean = df_all.groupby(df_all.index).mean()
    return df_mean


# def df_convert_matrix(df):
#     array = df.values
#     matrix = np.matrix(array)
#     return matrix


def integrate_spectra(df: pd.core.frame.DataFrame):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    integrated_spectra_area = integrate.trapezoid(y, x, axis=0)
    return integrated_spectra_area


def integrate_spectrum(df_2col):
    num_sets = int(df_2col.shape[1] / 2)
    integrated_areas = []
    for i in range(num_sets):
        integrated_area = simps(
            df_2col.iloc[:, 2 * i + 1].values, df_2col.iloc[:, 2 * i].values
        )
        integrated_area = round(integrated_area)
        integrated_areas.append(integrated_area)
        fig, ax = plt.subplots()
        ax.plot(
            df_2col.iloc[:, 0],
            df_2col.iloc[:, 2 * i + 1],
            label=f"{df_2col.columns[2*i+1]}",
        )
        ax.fill_between(
            df_2col.iloc[:, 2 * i],
            df_2col.iloc[:, 2 * i + 1],
            color="skyblue",
            alpha=0.4,
            label="Integration",
        )
        ax.set_title(f"Integrated Area: {integrated_area:.0f}")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True)
        plt.show()
    return integrated_areas


def find_closest_value(lst, target):
    closest_value = None
    min_difference = float("inf")
    for num in lst:
        difference = abs(num - target)
        if difference < min_difference:
            min_difference = difference
            closest_value = num
    return closest_value


# def fill_nan(raw_df, columns_range):
#     new_df = deepcopy(raw_df)
#     new_df.iloc[:,columns_range[1]][(new_df.iloc[:,columns_range[0]] > 380) & (new_df.iloc[:,columns_range[0]] < 400)] = np.NaN
# nor_data.iloc[:,1][(nor_data.iloc[:,0] > 385) & (nor_data.iloc[:,0] < 393)] = np.NaN
# nor_data.iloc[:,5][(nor_data.iloc[:,4] > 385) & (nor_data.iloc[:,4] < 393)] = np.NaN
#     return new_df


def wavelength_to_wavenumber(wavelength):
    wavenumber = 10**7 / wavelength
    return wavenumber


def wavenumber_to_wavelength(wavenumber):
    wavelength = 10**7 / wavenumber
    return wavelength


def wl2wn(wavelength):
    wavenumber = 10**7 / wavelength
    return wavenumber


def wn2wl(wavenumber):
    wavelength = 10**7 / wavenumber
    return wavelength


def nm_to_cm1(nm):
    cm1 = 10**7 / nm
    return cm1


def nm_to_e3cm1(nm):
    e3cm1 = 10**4 / nm
    return e3cm1


def cm1_to_nm(cm1):
    nm = 10**7 / cm1
    return nm


def e3cm1_to_nm(e3cm1):
    nm = 10**4 / e3cm1
    return nm


def nm_to_eV(nm):
    eV = 1240 / nm
    return eV


def eV_to_nm(eV):
    nm = 1240 / eV
    return nm


def cm1_to_eV(cm1):
    eV = 8065.544 / cm1
    return eV


def eV_to_cm1(eV):
    cm1 = 8065.544 / eV
    return cm1


# def nm_to_eV(nm):
#     eV = []
#     for wl in nm:
#         eV.append(round(1240 / wl, 2))
#     return eV

# def wn21000wl(wavenumber):
#     wavelength = 10**4 / wavenumber
#     return wavelength


# def wavelength_to_wavenumber(wavelength):
#     if wavelength == 0:
#         return None
#     else:
#         wavenumber = 10**7 / wavelength
#         return wavenumber


# def nm_to_cm1(wavelengths):
#     cm1 = []
#     for wavelength in wavelengths:
#         cm1.append(round(1 / wavelength, 2))
#     return cm1


def centimeter_to_inch(centimeter):
    return centimeter / 2.54


def inch_to_centimeter(inch):
    return 2.54 * inch


###################################### DataFrame ######################################

# df_1col
# df_1cols
# ls_df_1col
# ls_df_1cols

# df_2col
# df_2cols
# ls_df_2col
# ls_df_2cols


# def df2col_to_df1col(df_2col):


def reshape_2colto1col(dcoldf):
    data = dcoldf.iloc[:, 1::2].values
    index = dcoldf.iloc[:, 0].values
    columns = dcoldf.columns[1::2].values
    scoldf = pd.DataFrame(data, index, columns)
    return scoldf


def reshape_1colto2col(scoldf, indexname="wl_nm"):
    dcoldf = pd.DataFrame()
    for col in scoldf.columns:
        data = pd.DataFrame({indexname: scoldf.index.values, col: scoldf[col].values})
        dcoldf = pd.concat([dcoldf, data], axis=1)
    return dcoldf


def reshape_1colto2col_v1(df_1col, axisname=None):
    df_2col = pd.DataFrame()
    if axisname is not None:
        axisname = axisname
    else:
        if df_1col.index.name is None:
            axisname = "wl_nm"
        else:
            axisname = df_1col.index.name
    for col in df_1col.columns:
        data = pd.DataFrame({axisname: df_1col.index.values, col: df_1col[col].values})
        df_2col = pd.concat([df_2col, data], axis=1)
    return df_2col


def reshape_1colto2col_v2(df_1col, axisname=None):
    df_2col = pd.DataFrame()
    if axisname is not None:
        axisname = axisname
    else:
        if df_1col.index.name is None:
            axisname = "time_ps"
        else:
            axisname = df_1col.index.name
    for col in df_1col.columns:
        data = pd.DataFrame({axisname: df_1col.index.values, col: df_1col[col].values})
        df_2col = pd.concat([df_2col, data], axis=1)
    # df_2col = df_2col.reset_index(drop=True)
    return df_2col


def df1col_to_df2col(df_1col, axisname=None):
    df_2col = pd.DataFrame()
    if axisname is None:
        if df_1col.index.name is None:
            print("No index name!")
            axisname = None
        else:
            axisname = df_1col.index.name
    else:
        axisname = axisname
    for col in df_1col.columns:
        data = pd.DataFrame({axisname: df_1col.index.values, col: df_1col[col].values})
        df_2col = pd.concat([df_2col, data], axis=1)
    return df_2col


def combine_dataframes(df1, df2):
    # new_df = df1.iloc[:,0:2] + df2.iloc[:,0:2] + df1.iloc[:,2:4] + df2.iloc[:, 2:4] + ...
    ncol_df1 = len(df1.columns)
    if ncol_df1 % 2 != 0:
        raise ValueError("df1 has odd columns")
    ncol_df2 = len(df2.columns)
    if ncol_df2 % 2 != 0:
        raise ValueError("2 has odd columns")
    if ncol_df1 != ncol_df2:
        raise ValueError("df1 and df2 must have the same number of columns")
    new_df = pd.DataFrame()
    for i in range(int(ncol_df1 / 2)):
        new_df = pd.concat([new_df, df1.iloc[:, 2 * i : 2 * i + 2]], axis=1)
        new_df = pd.concat([new_df, df2.iloc[:, 2 * i : 2 * i + 2]], axis=1)
    return new_df


# ratio = width/height
def calculate_width(height, ratio):
    return height * ratio


def calculate_height(width, ratio):
    return width / ratio


def percent_to_float(percent):
    return float(percent.strip("%")) / 100


def float_to_percent(float, decimals=2):
    return f"{round(float*100, decimals)}%"


def float_to_percentage(x):
    if isinstance(x, float):
        return "{:.2f}%".format(x * 100)
    return x


def generate_negative_sequence(lowerlimit, interval):
    sequence = []
    for i in range(1, 101):
        number = -i * interval
        if number >= lowerlimit:
            sequence.append(number)
        else:
            break
    return sequence


def generate_positive_sequence(upperlimit, interval):
    sequence = []
    for i in range(1, 101):
        number = i * interval
        if number <= upperlimit:
            sequence.append(number)
        else:
            break
    return sequence


def generate_ticks_linearfield(limit, interval):
    negative_ticks = generate_negative_sequence(limit[0], interval)
    positive_ticks = generate_positive_sequence(limit[1], interval)
    ticks = negative_ticks + [0] + positive_ticks
    ticks.sort()
    return ticks


def save_dict_as_json(data, file_name):
    # 处理不同类型的数据
    for key, value in data.items():
        # 如果是DataFrame，转换为列表形式的字典
        if isinstance(value, pd.DataFrame):
            data[key] = value.to_dict(orient="records")
        # 如果是numpy数组，转换为列表
        elif isinstance(value, np.ndarray):
            data[key] = value.tolist()
    # 序列化字典并写入文件
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)


def combine_data_fit_kinetics(df_raw_kinetics, df_fit_kinetics):
    num_cols_raw = df_raw_kinetics.shape[1]
    num_cols_fit = df_fit_kinetics.shape[1]
    if num_cols_raw == num_cols_fit:
        list_df_raw_fit_kinetics = []
        for i in range(num_cols_raw):
            df_raw_fit = pd.concat(
                [df_raw_kinetics.iloc[:, i], df_fit_kinetics.iloc[:, i]], axis=1
            )
            df_raw_fit.columns = [
                int(float(df_raw_kinetics.columns[i])),
                f"{int(float(df_raw_kinetics.columns[i]))}fit",
            ]
            list_df_raw_fit_kinetics.append(df_raw_fit)
        return list_df_raw_fit_kinetics
    else:
        print("Number of columns in raw kinetics and fitted kinetics are different.")
        print("Please check the data.")
        return None


def tailor_2col_spctra(df_2col, x_limit):
    df = df_2col.loc[
        (df_2col.iloc[:, 0] >= x_limit[0]) & (df_2col.iloc[:, 0] <= x_limit[1])
    ]
    return df


def split_dataframe(df_2col, num_cols=2):
    if df_2col.shape[1] % 2 != 0:
        raise ValueError("Dataframe without even columns!")

    list_sets = []
    num_sets = int(df_2col.shape[1] / num_cols)
    for i in range(0, num_sets):
        df_set = df_2col.iloc[:, num_cols * i : num_cols * i + num_cols]
        list_sets.append(df_set)
    return list_sets


def calculate_solvent_polarity(e, n):
    return ((e - 1) / (2 * e + 1)) - ((n**2 - 1) / (2 * n**2 + 1))


def wl2wn_1coldf(df_1col):
    new_1coldf = deepcopy(df_1col)
    new_1coldf.index = 10**7 / new_1coldf.index
    return new_1coldf


def wl2wn_2coldf(df_2col):
    num_sets = df_2col.shape[1] // 2
    new_2coldf = deepcopy(df_2col)
    for i in range(num_sets):
        new_2coldf.iloc[:, 2 * i] = 10**7 / new_2coldf.iloc[:, 2 * i]
    return new_2coldf


# def nm_to_cm1_2coldf(df_2col):
#     num_sets = df_2col.shape[1]//2
#     newdf_2col = deepcopy(df_2col)
#     for i in range(num_sets):
#         newdf_2col.iloc[:,2*i] = 10**7 / newdf_2col.iloc[:,2*i]
#     return newdf_2col


def formalize_nsta_delaytime(delaytime):
    if delaytime > 0 and delaytime < 1000:
        label = f"{delaytime:.0f} ns"
    elif delaytime >= 1000 and delaytime < 1000000:
        label = f"{delaytime/1000:.0f} us"
    elif delaytime >= 1000000 and delaytime < 1000000000:
        label = f"{delaytime / 1000000:.0f} ms"
    elif delaytime >= 1000000000 and delaytime < 1000000000000:
        label = f"{delaytime / 1000000000:.0f} s"
    return label


def formate_nsta_time(delaytime):
    if delaytime > 0 and delaytime < 1000:
        label = f"{delaytime:0f} ns"
    elif delaytime >= 1000 and delaytime < 1000000:
        label = f"{delaytime/1000:.0f} us"
    elif delaytime >= 1000000 and delaytime < 1000000000:
        label = f"{delaytime / 1000000:.0f} ms"
    elif delaytime >= 1000000000 and delaytime < 1000000000000:
        label = f"{delaytime / 1000000000:.0f} s"
    return label


def formalize_fsta_delaytime(delaytime):
    if delaytime > 0 and delaytime < 10:
        label = f"{round(delaytime, 1)} ps"
    elif delaytime >= 10 and delaytime < 1000:
        label = f"{round(delaytime)} ps"
    elif delaytime >= 1000 and delaytime <= 80000:
        label = f"{delaytime / 1000:.1f} ns"
    else:
        label = "> 80 ns"
    return label


# def formalize_fsta_delaytime(delaytime):
#     if delaytime < 1:
#         label = round(delaytime, 1)
#         label = f"= {label} ps"
#     elif delaytime > 1 and delaytime < 1000:
#         label = round(delaytime)
#         label = f"= {label} ps"

#     elif delaytime >= 1000 and delaytime <= 50000:
#         label = delaytime / 1000  # ps to ns
#         label = f"= {label:.1f} ns"
#     else:
#         label = ">> 10 ns"
#     return label


def assignsample_lncom(filename):
    # fullfilename = mf.extract_directory_filename(filepath)[1]
    sample_solvent = "_".join(filename.split("_")[:2])
    if sample_solvent == "a1_meoh":
        sample = "CS124"
    elif sample_solvent == "c0a1_meoh":
        # sample = "L1"
        sample = "C0A1"
    elif sample_solvent == "gdc0a1_meoh":
        # sample = "GdL1"
        sample = "GdC0A1"
    elif sample_solvent == "tbc0a1_meoh":
        # sample = "TbL1"
        sample = "TbC0A1"
    elif sample_solvent == "euc0a1_meoh":
        # sample = "EuL1"
        sample = "EuC0A1"
    elif sample_solvent == "ybc0a1_meoh":
        # sample = "EuL1"
        sample = "YbC0A1"
    elif sample_solvent == "euc0a1_etoh":
        # sample = "EuL1 in EtOH"
        sample = "EuC0A1 in EtOH"
    elif sample_solvent == "euc0a1_h2o":
        # sample = "EuL1 in H2O"
        sample = "EuC0A1 in H2O"
    elif sample_solvent == "euc0a1mom_meoh":
        # sample = "EuL2"
        sample = "EuC0A1MOM"
    elif sample_solvent == "euc0a1cf3_meoh":
        # sample = "EuL3"
        sample = "EuC0A1CF3"
    elif sample_solvent == "c3a1_meoh":
        # sample = "L1"
        sample = "C3A1"
    elif sample_solvent == "gdc3a1_meoh":
        # sample = "GdL1"
        sample = "GdC3A1"
    elif sample_solvent == "tbc3a1_meoh":
        # sample = "TbL1"
        sample = "TbC3A1"
    elif sample_solvent == "euc3a1_meoh":
        # sample = "EuL1"
        sample = "EuC3A1"
    elif (
        sample_solvent == "euc3a1salavat_meoh" or sample_solvent == "euc3a1salauat_meoh"
    ):
        # sample = "EuL1"
        sample = "EuC3A1"
    elif sample_solvent == "ybc3a1_meoh":
        # sample = "EuL1"
        sample = "YbC3A1"
    elif sample_solvent == "c124d_meoh":
        # sample = "L1"
        sample = "C124COCH2Cl"
    elif sample_solvent == "gdc0a2_meoh":
        sample = "GdC0A2"
    elif sample_solvent == "tbc0a2_meoh":
        sample = "TbC0A2"
    elif sample_solvent == "euc0a2_meoh":
        sample = "EuC0A2"
    else:
        sample = sample_solvent
    return sample


def format_STEng(filename):
    # fullfilename = mf.extract_directory_filename(filepath)[1]
    sample_solvent = "_".join(filename.split("_")[:2])
    if sample_solvent == "a1_meoh":
        sample = "CS124"
    elif sample_solvent == "c0a1_meoh":
        # sample = "L1"
        sample = "C0A1"
    elif sample_solvent == "gdc0a1_h2o":
        # sample = "GdL1"
        sample = "GdC0A1 in H2O"
    elif sample_solvent == "gdc0a1_meoh":
        # sample = "GdL1"
        sample = "GdC0A1"
    elif sample_solvent == "gdc0a1_etoh":
        # sample = "GdL1"
        sample = "GdC0A1 in EtOH"
    elif sample_solvent == "tbc0a1_h2o":
        # sample = "TbL1"
        sample = "TbC0A1 in H2O"
    elif sample_solvent == "tbc0a1_meoh":
        # sample = "TbL1"
        sample = "TbC0A1"
    elif sample_solvent == "tbc0a1_etoh":
        # sample = "TbL1"
        sample = "TbC0A1 in EtOH"
    elif sample_solvent == "euc0a1_h2o":
        # sample = "EuL1 in H2O"
        sample = "EuC0A1 in H2O"
    elif sample_solvent == "euc0a1_meoh":
        # sample = "EuL1"
        sample = "EuC0A1"
    elif sample_solvent == "euc0a1_etoh":
        # sample = "EuL1 in EtOH"
        sample = "EuC0A1 in EtOH"
    elif sample_solvent == "ybc0a1_meoh":
        # sample = "EuL1"
        sample = "YbC0A1"
    elif sample_solvent == "euc0a1mom_meoh":
        # sample = "EuL2"
        sample = "EuC0A1MOM"
    elif sample_solvent == "euc0a1cf3_meoh":
        # sample = "EuL3"
        sample = "EuC0A1CF3"
    elif sample_solvent == "c3a1_meoh":
        # sample = "L1"
        sample = "C3A1"
    elif sample_solvent == "gdc3a1_meoh":
        # sample = "GdL1"
        sample = "GdC3A1"
    elif sample_solvent == "tbc3a1_meoh":
        # sample = "TbL1"
        sample = "TbC3A1"
    elif sample_solvent == "euc3a1_meoh":
        # sample = "EuL1"
        sample = "EuC3A1"
    elif (
        sample_solvent == "euc3a1salavat_meoh" or sample_solvent == "euc3a1salauat_meoh"
    ):
        # sample = "EuL1"
        sample = "EuC3A1"
    elif sample_solvent == "ybc3a1_meoh":
        # sample = "EuL1"
        sample = "YbC3A1"
    elif sample_solvent == "c124d_meoh":
        # sample = "L1"
        sample = "C124COCH2Cl"
    elif sample_solvent == "gdc0a2_meoh":
        sample = "GdC0A2"
    elif sample_solvent == "tbc0a2_meoh":
        sample = "TbC0A2"
    elif sample_solvent == "euc0a2_meoh":
        sample = "EuC0A2"
    else:
        print("Did not find the sample name.")
        # sample = sample_solvent
    return sample


def determine_timedelay_label(xlimit):
    if xlimit[0][1] < 100:
        major_tick = list(
            set(generate_ticks_linearfield([xlimit[0][0], xlimit[0][1]], xlimit[1]))
            | set([i for i in [100, 500, 1000, 5000, 10000] if i > xlimit[0][1]])
        )
        minor_tick = list(
            set(generate_ticks_linearfield([xlimit[0][0], xlimit[0][1]], xlimit[2]))
            | set(
                [
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    200,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    7000,
                    8000,
                    9000,
                ]
            )
        )
    elif xlimit[0][1] >= 100 and xlimit[0][1] < 1000:
        major_tick = list(
            set(generate_ticks_linearfield([xlimit[0][0], xlimit[0][1]], xlimit[1]))
            | set([i for i in [100, 500, 1000, 5000, 10000] if i > xlimit[0][1]])
        )
        minor_tick = list(
            set(generate_ticks_linearfield([xlimit[0][0], xlimit[0][1]], xlimit[2]))
            | set(
                [
                    200,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    7000,
                    8000,
                    9000,
                ]
            )
        )
    elif xlimit[0][1] >= 1000 and xlimit[0][1] < 10000:
        major_tick = list(
            set(generate_ticks_linearfield([xlimit[0][0], xlimit[0][1]], xlimit[1]))
            | set([i for i in [100, 500, 1000, 5000, 10000] if i > xlimit[0][1]])
        )
        minor_tick = list(
            set(generate_ticks_linearfield([xlimit[0][0], xlimit[0][1]], xlimit[2]))
            | set([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
        )
    major_tick.sort()
    minor_tick.sort()
    return [major_tick, minor_tick]


def compare_column_counts(*dataframes):
    # Get the number of columns for the first DataFrame
    num_columns = dataframes[0].shape[1]

    # Compare the number of columns for each DataFrame
    for df in dataframes[1:]:
        if df.shape[1] != num_columns:
            return False

    return True

def calculate_driving_force_PET(E_ox, E_red, q_dc, q_aa, q_d, q_a, epsilon_r, a, E_00):
    w_dcaa = ((q_dc * q_aa ) / (epsilon_r * a * 10**-10 )) * (2.307 * 10**-28) * (6.242 * 10**+18)
    w_da = ((q_d * q_a ) / (epsilon_r * a * 10**-10 )) * (2.307 * 10**-28) * (6.242 * 10**+18)
    delta_G = E_ox - E_red + w_dcaa- - w_da - E_00
    return print(delta_G)


if __name__ == "__main__":
    # print(centimeter_to_inch(8.5))
    # print(inch_to_centimeter(3.33))
    # print(generate_negative_sequence(-100, 20))
    # print(generate_positive_sequence(100, 20))
    # print(generate_ticks_linearfield([-100, 500], 30))
    # print(calculate_solvent_polarity(24.55, 1.363))
    # xlimit=[[-5, 80, 8000], 20, 10]
    # print(determine_timedelay_label(xlimit))
    # xlimit=[[-5, 200, 8000], 100, 50]
    # print(determine_timedelay_label(xlimit))
    # e3cm = 25
    # print(e3cm1_to_nm(e3cm))
    # nm = 500
    # print(nm_to_e3cm1(nm))
    # print(nm_to_cm1(400))
    # print(cm1_to_nm(20000))
    # print(cm1_to_eV(30000) - cm1_to_eV(27000))
    calculate_driving_force_PET(1.65, -3.9, 1, 1, 0, 0, 30, 6.2, 3.54)
    calculate_driving_force_PET(1.65, -3.7, 1, 1, 0, 0, 30, 6.2, 3.54)
    calculate_driving_force_PET(1.65, -0.83, 1, 1, 0, 0, 30, 6.2, 3.54)