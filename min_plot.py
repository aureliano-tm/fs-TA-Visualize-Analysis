import os

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as plcolor

# from plotly.express.colors import qualitative

import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.use('Agg')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# mpl.colormaps["tab20c"].colors

mpl.rcParams["font.serif"] = "Arial"  # 衬线字体
mpl.rcParams["font.sans-serif"] = "Arial"  # 无衬线字体
mpl.rcParams["legend.fancybox"] = False
# mpl.rcParams["legend.loc"] = "upper right"
# mpl.rcParams["legend.title_fontsize"] = 6
# mpl.rcParams["legend.fontsize"] = 6
# mpl.rcParams['legend.numpoints'] = 2
# mpl.rcParams['legend.framealpha'] = None
# mpl.rcParams['legend.scatterpoints'] = 3
# mpl.rcParams['legend.edgecolor'] = 'inherit'

import min_function as mf
import min_instrument as mins
import min_math as mm

# Colorway

rainbow7 = [
    "#FF0000",
    "#FF6600",
    "#FFEE00",
    "#00FF00",
    "#007DFF",
    "#4400FF",
    "#9900FF",
    "#808080",
]

system_color = [
    "#F00082",
    "#FA3C3C",
    "#F08228",
    "#E6AF2D",
    "#E6DC32",
    "#A0E632",
    "#00DC00",
    "#00D28C",
    "#00C8C8",
    "#00A0FF",
    "#1E3CFF",
    "#6E00DC",
    "#A000C8",
]

fadered = [
    "#d83933",
    "#db534d",
    "#de6c67",
    "#e18681",
    "#e39f9b",
    "#e6b9b5",
    "#e9d2cf",
    "#ecece9",
]

fadeblue = [
    "#274575",
    "#435d86",
    "#5f7596",
    "#7b8da7",
    "#98a4b7",
    "#b4bcc8",
    "#d0d4d8",
    "#ecece9",
]

min_colorway = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FFA500",
    "#800080",
    "#008000",
    "#000080",
    "#800000",
    "#FFC0CB",
    "#ADD8E6",
    "#FF69B4",
    "#800000",
    "#008080",
    "#00FF7F",
    "#FFD700",
    "#FF6347",
    "#FF8C00",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
]

color_lnpet = [
    "#274575",
    "#c13430",
    "#397e47",
]


# [(255, 0, 0), (255, 102, 0), (255, 238, 0), (0, 255, 0), (0, 125, 255), (68, 0, 255), (153, 0, 255)]
# gradient
# colorway  = [
#     "#274575", "#40436c", "#5a4262", "#734059", "#8c3e4f", "#a53c46", "#bf3b3c", "#d83933"
# ]


def gradient_color(maincolor, step, reverse=False):
    gradin = 1 / step
    multiple = np.arange(step, 0, -1)
    multiple = gradin * multiple
    multiple = multiple.tolist()
    if reverse is True:
        multiple = multiple[::-1]
    gradient_color = []
    for i in multiple:
        color = maincolor + (i,)
        gradient_color.append(color)
    return tuple(gradient_color)


rb_kinetics = ListedColormap(
    (
        (0.15294117647058825, 0.27058823529411763, 0.4588235294117647, 1.0),
        (0.7568627450980392, 0.20392156862745098, 0.18823529411764706, 1.0),
        (0.2235294117647059, 0.49411764705882355, 0.2784313725490196, 1.0),
    )
)

default_mplcolor = ListedColormap(
    (
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        (1.0, 0.4980392156862745, 0.054901960784313725),
    )
)

default_mplcolor_r = ListedColormap(
    (
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    )
)

# 43, 99, 151
# 180, 65, 16
# Preview

## 1col df


def preview_1coldf(
    df_1col=None,  # filepath or dataframe
    mode="lines",
    xtype="linear",
    ytype="linear",
    width=None,
    height=750,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
):
    if type(df_1col) is not pd.DataFrame:  # filepath
        data = pd.read_csv(df_1col, index_col=0)
    else:  # dataframe
        data = df_1col

    fig = go.Figure()
    for i in range(int(len(data.columns))):
        fig.add_scatter(
            x=data.index,
            y=data.iloc[:, i],
            name=f"{data.columns[i]}",
            mode=mode,
        )
    fig.update_layout(
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        legend=dict(xanchor="left", x=1, yanchor="top", y=1),
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    fig.show()


def preview_1coldf_v2(
    df_1col=None,  # filepath or dataframe
    mode="lines",
    xtype="linear",
    ytype="linear",
    yscale=0,
    width=None,
    height=750,
    showninwn=False,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
):
    if type(df_1col) is not pd.DataFrame:  # filepath
        data = pd.read_csv(df_1col, index_col=0)
    else:  # dataframe
        data = df_1col

    if yscale != 0:
        data = data * 10**yscale

    fig = go.Figure()
    for i in range(int(len(data.columns))):
        if showninwn is False:
            fig.add_scatter(
                x=data.index,
                y=data.iloc[:, i],
                name=f"{data.columns[i]}",
                mode=mode,
            )
        else:
            fig.add_scatter(
                x=10**7 / data.index,
                y=data.iloc[:, i],
                name=f"{data.columns[i]}",
                mode=mode,
            )
            fig.update_xaxes(autorange="reversed")
    fig.update_layout(
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        legend=dict(xanchor="left", x=1, yanchor="top", y=1),
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    fig.show()
    return data


def preview_list_1coldf(
    list_df_1col=None,
    mode="lines",
    xtype="linear",
    ytype="linear",
    width=None,
    height=750,
):
    fig = go.Figure()
    for df_1col in list_df_1col:
        if type(df_1col) is not pd.DataFrame:
            data = pd.read_csv(df_1col, index_col=0)
        else:
            data = df_1col

        for i in range(int(len(data.columns))):
            fig.add_scatter(
                x=data.index,
                y=data.iloc[:, i],
                name=f"{data.columns[i]}",
                mode=mode,
            )

    fig.update_layout(
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        legend=dict(xanchor="left", x=1, yanchor="top", y=1),
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )

    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)

    fig.show()


def preview_ls_1coldf(
    ls_df_1col=None,
    mode="lines",
    xtype="linear",
    ytype="linear",
    width=None,
    height=750,
    showninwn=False,
):
    for df_1col in ls_df_1col:
        fig = go.Figure()

        if type(df_1col) is not pd.DataFrame:
            data = pd.read_csv(df_1col, index_col=0, header=0)
        else:
            data = df_1col
        if showninwn is True:
            data.index = 10**7 / data.index
            # data = data[::-1]
        # display(data.head())
        display(data)

        for i in range(int(len(data.columns))):
            fig.add_scatter(
                x=data.index,
                y=data.iloc[:, i],
                name=f"{data.columns[i]}",
                mode=mode,
            )

        fig.update_layout(
            xaxis_type=xtype,
            yaxis_type=ytype,
            showlegend=True,
            legend=dict(xanchor="left", x=1, yanchor="top", y=1),
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
        )

        if showninwn is True:
            fig.update_xaxes(autorange="reversed")
        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)

        fig.show()


def preview_ls_1coldf_v2(
    ls_df_1col=None,
    mode="lines",
    xtype="linear",
    ytype="linear",
    width=None,
    height=750,
    showninwn=False,
    overlap=True,
):
    if overlap is True:
        fig = go.Figure()
        for df_1col in ls_df_1col:
            if type(df_1col) is not pd.DataFrame:
                data = pd.read_csv(df_1col, index_col=0, header=0)
            else:
                data = df_1col
            if showninwn is True:
                data.index = 10**7 / data.index
                # data = data[::-1]
            # display(data.head())
            # display(data)

            for i in range(int(len(data.columns))):
                fig.add_scatter(
                    x=data.index,
                    y=data.iloc[:, i],
                    name=f"{data.columns[i]}",
                    mode=mode,
                )

        fig.update_layout(
            xaxis_type=xtype,
            yaxis_type=ytype,
            showlegend=True,
            legend=dict(xanchor="left", x=1, yanchor="top", y=1),
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
        )

        if showninwn is True:
            fig.update_xaxes(autorange="reversed")
        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)

        fig.show()
        return
    else:
        for df_1col in ls_df_1col:
            fig = go.Figure()

            if type(df_1col) is not pd.DataFrame:
                data = pd.read_csv(df_1col, index_col=0, header=0)
            else:
                data = df_1col
            if showninwn is True:
                data.index = 10**7 / data.index
                # data = data[::-1]
            # display(data.head())
            display(data)

            for i in range(int(len(data.columns))):
                fig.add_scatter(
                    x=data.index,
                    y=data.iloc[:, i],
                    name=f"{data.columns[i]}",
                    mode=mode,
                )

            fig.update_layout(
                xaxis_type=xtype,
                yaxis_type=ytype,
                showlegend=True,
                legend=dict(xanchor="left", x=1, yanchor="top", y=1),
                hovermode="x unified",
                hoverlabel_bgcolor="rgba(0,0,0,0)",
                hoverlabel_bordercolor="rgba(0,0,0,0)",
            )

            if showninwn is True:
                fig.update_xaxes(autorange="reversed")
            if width is not None:
                fig.update_layout(width=width)
            if height is not None:
                fig.update_layout(height=height)

            fig.show()


def preview_ls_1coldf_v3(
    ls_df_1col=None,
    mode="lines",
    xtype="linear",
    ytype="linear",
    xscale=0,
    yscale=0,
    width=None,
    height=750,
    overlap=True,
    showninwn=False,
):
    if overlap is True:
        new_df_1col = pd.DataFrame()
        fig = go.Figure()
        for df_1col in ls_df_1col:
            if type(df_1col) is not pd.DataFrame:
                data = pd.read_csv(df_1col, index_col=0, header=0)
            else:
                data = df_1col

            df_copy = data.copy()

            new_df_1col = pd.concat([new_df_1col, df_copy], axis=1)

            if showninwn is True:
                df_copy.index = 10**7 / df_copy.index

            if yscale != 0:
                df_copy = df_copy * 10**yscale

            for i in range(int(len(df_copy.columns))):
                fig.add_scatter(
                    x=df_copy.index,
                    y=df_copy.iloc[:, i],
                    name=f"{df_copy.columns[i]}",
                    mode=mode,
                )

        fig.update_layout(
            xaxis_type=xtype,
            yaxis_type=ytype,
            showlegend=True,
            legend=dict(xanchor="left", x=1, yanchor="top", y=1),
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
        )

        if showninwn is True:
            fig.update_xaxes(autorange="reversed")

        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)

        fig.show()
        return new_df_1col
    else:
        new_ls_df_1col = []
        for df_1col in ls_df_1col:
            fig = go.Figure()

            if type(df_1col) is not pd.DataFrame:
                data = pd.read_csv(df_1col, index_col=0, header=0)
            else:
                data = df_1col

            df_copy = data.copy()

            new_ls_df_1col.append(df_copy)

            if showninwn is True:
                df_copy.index = 10**7 / df_copy.index
            
            if yscale != 0:
                df_copy = df_copy * 10**yscale

            for i in range(int(len(df_copy.columns))):
                fig.add_scatter(
                    x=df_copy.index,
                    y=df_copy.iloc[:, i],
                    name=f"{df_copy.columns[i]}",
                    mode=mode,
                )

            fig.update_layout(
                xaxis_type=xtype,
                yaxis_type=ytype,
                showlegend=True,
                legend=dict(xanchor="left", x=1, yanchor="top", y=1),
                hovermode="x unified",
                hoverlabel_bgcolor="rgba(0,0,0,0)",
                hoverlabel_bordercolor="rgba(0,0,0,0)",
            )

            if showninwn is True:
                fig.update_xaxes(autorange="reversed")
            if width is not None:
                fig.update_layout(width=width)
            if height is not None:
                fig.update_layout(height=height)

            fig.show()
        return new_ls_df_1col


def preview_ls_1coldf_v4(
    ls_df_1col=None,
    mode="lines",
    xtype="linear",
    ytype="linear",
    xscale=0,
    yscale=0,
    width=None,
    height=750,
    overlap=True,
    showninwn=False,
    **read_csv_kwargs,
):
    if overlap is True:
        new_df_1col = pd.DataFrame()
        fig = go.Figure()
        for df_1col in ls_df_1col:
            if type(df_1col) is not pd.DataFrame:
                data = pd.read_csv(
                    df_1col,
                    **read_csv_kwargs,
                )
            else:
                data = df_1col

            df_copy = data.copy()

            new_df_1col = pd.concat([new_df_1col, df_copy], axis=1)

            if showninwn is True:
                df_copy.index = 10**7 / df_copy.index

            if yscale != 0:
                df_copy = df_copy * 10**yscale

            for i in range(int(len(df_copy.columns))):
                fig.add_scatter(
                    x=df_copy.index,
                    y=df_copy.iloc[:, i],
                    name=f"{df_copy.columns[i]}",
                    mode=mode,
                )

        fig.update_layout(
            xaxis_type=xtype,
            yaxis_type=ytype,
            showlegend=True,
            legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
        )

        if showninwn is True:
            fig.update_xaxes(autorange="reversed")

        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)

        fig.show()
        return new_df_1col
    else:
        new_ls_df_1col = []
        for df_1col in ls_df_1col:
            fig = go.Figure()

            if type(df_1col) is not pd.DataFrame:
                data = pd.read_csv(df_1col, **read_csv_kwargs)
            else:
                data = df_1col

            df_copy = data.copy()

            new_ls_df_1col.append(df_copy)

            if showninwn is True:
                df_copy.index = 10**7 / df_copy.index

            if yscale != 0:
                df_copy = df_copy * 10**yscale

            for i in range(int(len(df_copy.columns))):
                fig.add_scatter(
                    x=df_copy.index,
                    y=df_copy.iloc[:, i],
                    name=f"{df_copy.columns[i]}",
                    mode=mode,
                )

            fig.update_layout(
                xaxis_type=xtype,
                yaxis_type=ytype,
                showlegend=True,
                legend=dict(xanchor="left", x=0, yanchor="top", y=1),
                hovermode="x unified",
                hoverlabel_bgcolor="rgba(0,0,0,0)",
                hoverlabel_bordercolor="rgba(0,0,0,0)",
            )

            if showninwn is True:
                fig.update_xaxes(autorange="reversed")
            if width is not None:
                fig.update_layout(width=width)
            if height is not None:
                fig.update_layout(height=height)

            fig.show()
        return new_ls_df_1col


def preview_ls_1coldf_v5(
    ls_df_1col=None,
    mode="lines",
    xtype="linear",
    ytype="linear",
    xscale=0,
    yscale=0,
    width=None,
    height=750,
    usefpname=True,
    overlap=True,
    showninwn=False,
    **read_csv_kwargs,
):
    if overlap is True:
        new_df_1col = pd.DataFrame()
        fig = go.Figure()
        for df_1col in ls_df_1col:
            if type(df_1col) is not pd.DataFrame:
                if usefpname is True:
                    data = pd.read_csv(
                        df_1col,
                        **read_csv_kwargs,
                    )
                    # data.columns = [os.path.basename(df_1col)]
                else:
                    data = pd.read_csv(
                        df_1col,
                        **read_csv_kwargs,
                    )
            else:
                data = df_1col

            df_copy = data.copy()

            new_df_1col = pd.concat([new_df_1col, df_copy], axis=1)

            if showninwn is True:
                df_copy.index = 10**7 / df_copy.index

            if yscale != 0:
                df_copy = df_copy * 10**yscale

            for i in range(int(len(df_copy.columns))):
                fig.add_scatter(
                    x=df_copy.index,
                    y=df_copy.iloc[:, i],
                    name=f"{df_copy.columns[i]}",
                    mode=mode,
                )

        fig.update_layout(
            xaxis_type=xtype,
            yaxis_type=ytype,
            showlegend=True,
            legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
        )

        if showninwn is True:
            fig.update_xaxes(autorange="reversed")

        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)

        fig.show()
        return new_df_1col
    else:
        new_ls_df_1col = []
        for df_1col in ls_df_1col:
            fig = go.Figure()

            if type(df_1col) is not pd.DataFrame:
                data = pd.read_csv(df_1col, **read_csv_kwargs)
            else:
                data = df_1col

            df_copy = data.copy()

            new_ls_df_1col.append(df_copy)

            if showninwn is True:
                df_copy.index = 10**7 / df_copy.index

            if yscale != 0:
                df_copy = df_copy * 10**yscale

            for i in range(int(len(df_copy.columns))):
                fig.add_scatter(
                    x=df_copy.index,
                    y=df_copy.iloc[:, i],
                    name=f"{df_copy.columns[i]}",
                    mode=mode,
                )

            fig.update_layout(
                xaxis_type=xtype,
                yaxis_type=ytype,
                showlegend=True,
                legend=dict(xanchor="left", x=0, yanchor="top", y=1),
                hovermode="x unified",
                hoverlabel_bgcolor="rgba(0,0,0,0)",
                hoverlabel_bordercolor="rgba(0,0,0,0)",
            )

            if showninwn is True:
                fig.update_xaxes(autorange="reversed")
            if width is not None:
                fig.update_layout(width=width)
            if height is not None:
                fig.update_layout(height=height)

            fig.show()
        return new_ls_df_1col
    

def preview_1coldf_wn(
    df_1col=None,
    mode="lines",
    # reversed=True,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    xtype="linear",
    ytype="linear",
    width=None,
    height=750,
):
    if type(df_1col) is not pd.DataFrame:
        data = pd.read_csv(df_1col, index_col=0)
    else:
        data = df_1col

    fig = go.Figure()
    for i in range(int(len(data.columns))):
        fig.add_scatter(
            x=10**7 / data.index,
            y=data.iloc[:, i],
            name=f"{data.columns[i]}",
            mode=mode,
        )
    fig.update_layout(
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        legend=dict(xanchor="left", x=1, yanchor="top", y=1),
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )

    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    # if reversed is True:
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    else:
        fig.update_xaxes(autorange="reversed")
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    fig.show()


# def preview_single_df(
#     df=None,
#     dftype="1col",
#     filepath=None,
#     mode="lines",
#     xtype="linear",
#     ytype="linear",
#     width=None,
#     height=None,
# ):
#     if filepath is None:
#         data = df

#     if dftype == "1col":
#         data = data
#     elif dftype == "2col":
#         data = mf.reshape_1colto2col(data)

#     fig = go.Figure()
#     for i in range(int(len(data.columns))):
#         fig.add_scatter(
#             x=data.index,
#             y=data.iloc[:, i],
#             name=f"{data.columns[i]}",
#             mode=mode,
#         )
#     fig.update_layout(
#         height=750,
#         xaxis_type=xtype,
#         yaxis_type=ytype,
#         showlegend=True,
#         legend=dict(xanchor="left", x=1, yanchor="top", y=1),
#         hovermode="x unified",
#         hoverlabel_bgcolor="rgba(0,0,0,0)",
#         hoverlabel_bordercolor="rgba(0,0,0,0)",
#     )
#     if width is not None:
#         fig.update_layout(width=width)
#     if height is not None:
#         fig.update_layout(height=height)
#     return fig.show()

# def preview_1col_df(
#     df=None,
#     filepath=None,
#     mode="lines",
#     xtype="linear",
#     ytype="linear",
#     width=None,
#     height=None,
# ):
#     if filepath is None:
#         data = df
#     else:
#         data = pd.read_csv(filepath, index_col=0)
#     fig = go.Figure()
#     for i in range(int(len(data.columns))):
#         fig.add_scatter(
#             x=data.index,
#             y=data.iloc[:, i],
#             name=f"{data.columns[i]}",
#             mode=mode,
#         )
#     fig.update_layout(
#         height=750,
#         xaxis_type=xtype,
#         yaxis_type=ytype,
#         showlegend=True,
#         legend=dict(xanchor="left", x=1, yanchor="top", y=1),
#         hovermode="x unified",
#         hoverlabel_bgcolor="rgba(0,0,0,0)",
#         hoverlabel_bordercolor="rgba(0,0,0,0)",
#     )
#     if width is not None:
#         fig.update_layout(width=width)
#     if height is not None:
#         fig.update_layout(height=height)
#     fig.show()

## 2col df


def preview_2coldf(
    df_2col=None,  # filepath or dataframe
    # filepath=None,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    width=None,
    height=750,
    mode="lines",
    xtype="linear",
    ytype="linear",
    legendposition="right",
    reversed=False,
):
    if type(df_2col) is not pd.DataFrame:  # filepath
        data = pd.read_csv(df_2col, index_col=0)
    else:  # dataframe
        data = df_2col

    fig = go.Figure()
    for i in range(int(len(data.columns) / 2)):
        if type(data.columns) == pd.core.indexes.base.Index:
            name = data.columns[2 * i + 1]
        elif type(data.columns) == pd.core.indexes.multi.MultiIndex:
            name = data.columns[2 * i + 1][-1]
        fig.add_scatter(
            x=data.iloc[:, 2 * i],
            y=data.iloc[:, 2 * i + 1],
            # name=f"{data.columns[2 * i + 1]}",
            name=name,
            mode=mode,
            # line_width=3.5,
        )
    fig.update_layout(
        font=dict(family="Arial", color="black", size=26),
        height=750,
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendposition == "right":
        fig.update_layout(
            legend=dict(
                xanchor="left",
                yanchor="top",
                x=1,
                y=1,
            )
        )
    elif legendposition == "top":
        fig.update_layout(
            legend=dict(
                # font,
                xanchor="left",
                yanchor="bottom",
                x=0,
                y=1,
            )
        )
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    if reversed is True:
        fig.update_xaxes(autorange="reversed")
    fig.show()


def preview_2coldf_v2(
    df_2col=None,  # filepath or dataframe
    # filepath=None,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    width=None,
    height=750,
    mode="lines",
    xtype="linear",
    ytype="linear",
    legendposition="right",
    reversed=False,
    showninwn=False,
):
    if type(df_2col) is not pd.DataFrame:  # filepath
        data = pd.read_csv(df_2col, index_col=0)
    else:  # dataframe
        data = df_2col

    fig = go.Figure()
    for i in range(int(len(data.columns) / 2)):
        if type(data.columns) == pd.core.indexes.base.Index:
            name = data.columns[2 * i + 1]
        elif type(data.columns) == pd.core.indexes.multi.MultiIndex:
            name = data.columns[2 * i + 1][-1]

        x = data.iloc[:, 2 * i]
        if showninwn is True:
            x = 10**7 / x
        y = data.iloc[:, 2 * i + 1]
        fig.add_scatter(
            x=x,
            y=y,
            name=name,
            mode=mode,
            # line_width=3.5,
        )
    fig.update_layout(
        font=dict(family="Arial", color="black", size=26),
        height=750,
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendposition == "right":
        fig.update_layout(
            legend=dict(
                xanchor="left",
                yanchor="top",
                x=1,
                y=1,
            )
        )
    elif legendposition == "top":
        fig.update_layout(
            legend=dict(
                # font,
                xanchor="left",
                yanchor="bottom",
                x=0,
                y=1,
            )
        )
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    if reversed is True:
        fig.update_xaxes(autorange="reversed")
    fig.show()
    return data

def preview_list_2coldf(
    list_df_2col=None,
    filepath=None,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    width=None,
    height=750,
    mode="lines",
    xtype="linear",
    ytype="linear",
    legendposition="right",
    reversed=False,
):
    fig = go.Figure()
    for df_2col in list_df_2col:
        if type(df_2col) is not pd.DataFrame:
            data = pd.read_csv(df_2col, index_col=0)
        else:
            data = df_2col

        for i in range(int(len(data.columns) / 2)):
            if type(data.columns) == pd.core.indexes.base.Index:
                name = data.columns[2 * i + 1]
            elif type(data.columns) == pd.core.indexes.multi.MultiIndex:
                name = data.columns[2 * i + 1][-1]
            fig.add_scatter(
                x=data.iloc[:, 2 * i],
                y=data.iloc[:, 2 * i + 1],
                # name=f"{data.columns[2 * i + 1]}",
                name=name,
                mode=mode,
                # line_width=3.5,
            )

    fig.update_layout(
        font=dict(family="Arial", color="black", size=26),
        height=750,
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )

    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendposition == "right":
        fig.update_layout(
            legend=dict(
                xanchor="left",
                yanchor="top",
                x=1,
                y=1,
            )
        )
    elif legendposition == "top":
        fig.update_layout(
            legend=dict(
                xanchor="left",
                yanchor="bottom",
                x=0,
                y=1,
            )
        )
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    if reversed is True:
        fig.update_xaxes(autorange="reversed")

    fig.show()


def preview_ls_2coldf(
    ls_df_2col=None,  # list of 2-column dataframes or list of filepaths
    overlap=True,
    mode="lines",
    xtype="linear",
    ytype="linear",
    xscale=0,
    yscale=0,
    showninwn=False,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    width=None,
    height=750,
    legendposition="right",
):
    if overlap is True:
        new_df_2col = pd.DataFrame()
        fig = go.Figure()
        for df_2col in ls_df_2col:
            if type(df_2col) is pd.DataFrame:
                data = df_2col
            else:
                data = pd.read_csv(df_2col, header=0, index_col=None)
            # display(data)

            if showninwn is True:
                data.iloc[:, 0::2] = 10**7 / data.iloc[:, 0::2]
                fig.update_xaxes(autorange="reversed")
            # display(data)

            if xscale != 0:
                data.iloc[:, 0::2] = data.iloc[:, 0::2] * 10**xscale
            if yscale != 0:
                data.iloc[:, 1::2] = data.iloc[:, 1::2] * 10**yscale

            new_df_2col = pd.concat([new_df_2col, data], axis=1)

            sets = int(data.shape[1] / 2)
            for i in range(sets):
                fig.add_scatter(
                    x=data.iloc[:, 2 * i],
                    y=data.iloc[:, 2 * i + 1],
                    name=f"{data.columns[2 * i + 1]}",
                    mode=mode,
                )

        fig.update_layout(
            xaxis_type=xtype,
            yaxis_type=ytype,
            showlegend=True,
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
        )

        if xlimit[0] is not None:
            fig.update_xaxes(range=xlimit[0])
        if xlimit[1] is not None:
            fig.update_xaxes(dtick=xlimit[1])
        if xlimit[2] is not None:
            fig.update_xaxes(minor_dtick=xlimit[2])

        if ylimit[0] is not None:
            fig.update_yaxes(range=ylimit[0])
        if ylimit[1] is not None:
            fig.update_yaxes(dtick=ylimit[1])
        if ylimit[2] is not None:
            fig.update_yaxes(minor_dtick=ylimit[2])

        if legendposition == "right":
            fig.update_layout(
                legend=dict(
                    xanchor="left",
                    yanchor="top",
                    x=1,
                    y=1,
                )
            )
        elif legendposition == "top":
            fig.update_layout(
                legend=dict(
                    xanchor="left",
                    yanchor="bottom",
                    x=0,
                    y=1,
                )
            )

        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)
        if reversed is True:
            fig.update_xaxes(autorange="reversed")

        fig.show()
        return new_df_2col
    else:
        new_df_2col = []
        for df_2col in ls_df_2col:
            df_copy = df_2col.copy()
            fig = go.Figure()

            if type(df_copy) is pd.DataFrame:
                data = df_copy
            else:
                data = pd.read_csv(df_copy, header=0, index_col=None)
            # display(data)

            if showninwn is True:
                data.iloc[:, 0::2] = 10**7 / data.iloc[:, 0::2]
            # display(data)

            if xscale != 0:
                data.iloc[:, 0::2] = data.iloc[:, 0::2] * 10**xscale
            if yscale != 0:
                data.iloc[:, 1::2] = data.iloc[:, 1::2] * 10**yscale

            new_df_2col.append(data)

            sets = int(data.shape[1] / 2)
            for i in range(sets):
                fig.add_scatter(
                    x=data.iloc[:, 2 * i],
                    y=data.iloc[:, 2 * i + 1],
                    name=f"{data.columns[2 * i + 1]}",
                    mode=mode,
                )

            fig.update_layout(
                xaxis_type=xtype,
                yaxis_type=ytype,
                showlegend=True,
                hovermode="x unified",
                hoverlabel_bgcolor="rgba(0,0,0,0)",
                hoverlabel_bordercolor="rgba(0,0,0,0)",
            )

            if xlimit[0] is not None:
                fig.update_xaxes(range=xlimit[0])
            if xlimit[1] is not None:
                fig.update_xaxes(dtick=xlimit[1])
            if xlimit[2] is not None:
                fig.update_xaxes(minor_dtick=xlimit[2])

            if ylimit[0] is not None:
                fig.update_yaxes(range=ylimit[0])
            if ylimit[1] is not None:
                fig.update_yaxes(dtick=ylimit[1])
            if ylimit[2] is not None:
                fig.update_yaxes(minor_dtick=ylimit[2])

            if legendposition == "right":
                fig.update_layout(
                    legend=dict(
                        xanchor="left",
                        yanchor="top",
                        x=1,
                        y=1,
                    )
                )
            elif legendposition == "top":
                fig.update_layout(
                    legend=dict(
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1,
                    )
                )

            if width is not None:
                fig.update_layout(width=width)
            if height is not None:
                fig.update_layout(height=height)
            if reversed is True:
                fig.update_xaxes(autorange="reversed")

            fig.show()
        return new_df_2col


def preview_ls_2coldf_v2(
    ls_df_2col=None,  # list of 2-column dataframes or list of filepaths
    overlap=True,
    mode="lines",
    xtype="linear",
    ytype="linear",
    xscale=0,
    yscale=0,
    showninwn=False,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    width=None,
    height=750,
    legendposition="right",
    **read_csv_kwargs,
):
    if overlap is True:
        new_df_2col = pd.DataFrame()
        fig = go.Figure()
        for df_2col in ls_df_2col:
            if type(df_2col) is pd.DataFrame:
                data = df_2col
            else:
                data = pd.read_csv(df_2col, **read_csv_kwargs)# header=0, index_col=None)
            # display(data)

            if showninwn is True:
                data.iloc[:, 0::2] = 10**7 / data.iloc[:, 0::2]
                fig.update_xaxes(autorange="reversed")
            # display(data)

            if xscale != 0:
                data.iloc[:, 0::2] = data.iloc[:, 0::2] * 10**xscale
            if yscale != 0:
                data.iloc[:, 1::2] = data.iloc[:, 1::2] * 10**yscale

            new_df_2col = pd.concat([new_df_2col, data], axis=1)

            sets = int(data.shape[1] / 2)
            for i in range(sets):
                fig.add_scatter(
                    x=data.iloc[:, 2 * i],
                    y=data.iloc[:, 2 * i + 1],
                    name=f"{data.columns[2 * i + 1]}",
                    mode=mode,
                )

        fig.update_layout(
            xaxis_type=xtype,
            yaxis_type=ytype,
            showlegend=True,
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
        )

        if xlimit[0] is not None:
            fig.update_xaxes(range=xlimit[0])
        if xlimit[1] is not None:
            fig.update_xaxes(dtick=xlimit[1])
        if xlimit[2] is not None:
            fig.update_xaxes(minor_dtick=xlimit[2])

        if ylimit[0] is not None:
            fig.update_yaxes(range=ylimit[0])
        if ylimit[1] is not None:
            fig.update_yaxes(dtick=ylimit[1])
        if ylimit[2] is not None:
            fig.update_yaxes(minor_dtick=ylimit[2])

        if legendposition == "right":
            fig.update_layout(
                legend=dict(
                    xanchor="left",
                    yanchor="top",
                    x=1,
                    y=1,
                )
            )
        elif legendposition == "top":
            fig.update_layout(
                legend=dict(
                    xanchor="left",
                    yanchor="bottom",
                    x=0,
                    y=1,
                )
            )

        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)
        if reversed is True:
            fig.update_xaxes(autorange="reversed")

        fig.show()
        return new_df_2col
    else:
        new_df_2col = []
        for df_2col in ls_df_2col:
            df_copy = df_2col.copy()
            fig = go.Figure()

            if type(df_copy) is pd.DataFrame:
                data = df_copy
            else:
                data = pd.read_csv(df_copy, **read_csv_kwargs)
            # display(data)

            if showninwn is True:
                data.iloc[:, 0::2] = 10**7 / data.iloc[:, 0::2]
            # display(data)

            if xscale != 0:
                data.iloc[:, 0::2] = data.iloc[:, 0::2] * 10**xscale
            if yscale != 0:
                data.iloc[:, 1::2] = data.iloc[:, 1::2] * 10**yscale

            new_df_2col.append(data)

            sets = int(data.shape[1] / 2)
            for i in range(sets):
                fig.add_scatter(
                    x=data.iloc[:, 2 * i],
                    y=data.iloc[:, 2 * i + 1],
                    name=f"{data.columns[2 * i + 1]}",
                    mode=mode,
                )

            fig.update_layout(
                xaxis_type=xtype,
                yaxis_type=ytype,
                showlegend=True,
                hovermode="x unified",
                hoverlabel_bgcolor="rgba(0,0,0,0)",
                hoverlabel_bordercolor="rgba(0,0,0,0)",
            )

            if xlimit[0] is not None:
                fig.update_xaxes(range=xlimit[0])
            if xlimit[1] is not None:
                fig.update_xaxes(dtick=xlimit[1])
            if xlimit[2] is not None:
                fig.update_xaxes(minor_dtick=xlimit[2])

            if ylimit[0] is not None:
                fig.update_yaxes(range=ylimit[0])
            if ylimit[1] is not None:
                fig.update_yaxes(dtick=ylimit[1])
            if ylimit[2] is not None:
                fig.update_yaxes(minor_dtick=ylimit[2])

            if legendposition == "right":
                fig.update_layout(
                    legend=dict(
                        xanchor="left",
                        yanchor="top",
                        x=1,
                        y=1,
                    )
                )
            elif legendposition == "top":
                fig.update_layout(
                    legend=dict(
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1,
                    )
                )

            if width is not None:
                fig.update_layout(width=width)
            if height is not None:
                fig.update_layout(height=height)
            if reversed is True:
                fig.update_xaxes(autorange="reversed")

            fig.show()
        return new_df_2col


def preview_2coldf_wn(
    df_2col=None,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    width=None,
    height=750,
    mode="lines",
    xtype="linear",
    ytype="linear",
    legendposition="right",
    reversed=True,
):
    if type(df_2col) is not pd.DataFrame:
        data = pd.read_csv(df_2col)
    else:
        data = df_2col

    fig = go.Figure()
    for i in range(int(len(data.columns) / 2)):
        fig.add_scatter(
            x=10**7 / data.iloc[:, 2 * i],
            y=data.iloc[:, 2 * i + 1],
            name=f"{data.columns[2 * i + 1]}",
            mode=mode,
        )
    fig.update_layout(
        font=dict(family="Arial", color="black", size=26),
        height=750,
        xaxis_type=xtype,
        yaxis_type=ytype,
        showlegend=True,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )

    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])

    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])

    if legendposition == "right":
        fig.update_layout(
            legend=dict(
                xanchor="left",
                yanchor="top",
                x=1,
                y=1,
            )
        )
    elif legendposition == "top":
        fig.update_layout(
            legend=dict(
                xanchor="left",
                yanchor="bottom",
                x=0,
                y=1,
            )
        )

    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)

    if reversed is True:
        fig.update_xaxes(autorange="reversed")

    fig.show()


def preview_2coldf_group(
    list_2coldf,
    colorway=plcolor.qualitative.Dark24,
    reversed=False,
):
    fig = go.Figure()

    num_columns = len(list_2coldf[0].columns)
    similarity = []
    for df_2col in list_2coldf[1:]:
        if len(df_2col.columns) != num_columns:
            similarity.append(0)
        else:
            similarity.append(1)

    if 0 in similarity:
        print("Not all the dataframe have the same columns.")
    else:
        num_plots = list_2coldf[0].shape[1] // 2
        for i in range(num_plots):
            sample = "_".join(list_2coldf[0].columns[2 * i + 1].split("_")[:2])
            color = colorway[i]
            for j, df in enumerate(list_2coldf):
                data = df.iloc[:, 2 * i : 2 * i + 2]
                if j == 0:
                    fig.add_scatter(
                        x=data.iloc[:, 0],
                        y=data.iloc[:, 1],
                        mode="lines",
                        name=data.columns[1],
                        legendgroup=sample,
                        line_color=color,
                        showlegend=True,
                    )
                else:
                    fig.add_scatter(
                        x=data.iloc[:, 0],
                        y=data.iloc[:, 1],
                        mode="lines",
                        name=data.columns[1],
                        legendgroup=sample,
                        line_color=color,
                        showlegend=False,
                    )

    fig.update_layout(
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
        height=750,
    )
    if reversed is True:
        fig.update_xaxes(autorange="reversed")
    return fig.show()


# def preview_2col_df(
#     df_2col=None,
#     xlimit=[None, None, None],
#     ylimit=[None, None, None],
#     width=None,
#     height=None,
#     mode="lines",
#     xtype="linear",
#     ytype="linear",
#     legendposition="right",
#     reversed=False,
# ):
#     if type(df_2col) is not pd.DataFrame:
#         data = pd.read_csv(df_2col, index_col=0)
#     else:
#         data = df_2col

#     fig = go.Figure()
#     for i in range(int(len(data.columns) / 2)):
#         if type(data.columns) == pd.core.indexes.base.Index:
#             name = data.columns[2 * i + 1]
#         elif type(data.columns) == pd.core.indexes.multi.MultiIndex:
#             name = data.columns[2 * i + 1][-1]
#         fig.add_scatter(
#             x=data.iloc[:, 2 * i],
#             y=data.iloc[:, 2 * i + 1],
#             # name=f"{data.columns[2 * i + 1]}",
#             name=name,
#             mode=mode,
#             # line_width=3.5,
#         )
#     fig.update_layout(
#         font=dict(family="Arial", color="black", size=26),
#         height=750,
#         xaxis_type=xtype,
#         yaxis_type=ytype,
#         showlegend=True,
#         hovermode="x unified",
#         hoverlabel_bgcolor="rgba(0,0,0,0)",
#         hoverlabel_bordercolor="rgba(0,0,0,0)",
#     )
#     if xlimit[0] is not None:
#         fig.update_xaxes(range=xlimit[0])
#     if xlimit[1] is not None:
#         fig.update_xaxes(dtick=xlimit[1])
#     if xlimit[2] is not None:
#         fig.update_xaxes(minor_dtick=xlimit[2])
#     if ylimit[0] is not None:
#         fig.update_yaxes(range=ylimit[0])
#     if ylimit[1] is not None:
#         fig.update_yaxes(dtick=ylimit[1])
#     if ylimit[2] is not None:
#         fig.update_yaxes(minor_dtick=ylimit[2])
#     if legendposition == "right":
#         fig.update_layout(
#             legend=dict(
#                 xanchor="left",
#                 yanchor="top",
#                 x=1,
#                 y=1,
#             )
#         )
#     elif legendposition == "top":
#         fig.update_layout(
#             legend=dict(
#                 xanchor="left",
#                 yanchor="bottom",
#                 x=0,
#                 y=1,
#             )
#         )
#     if width is not None:
#         fig.update_layout(width=width)
#     if height is not None:
#         fig.update_layout(height=height)
#     if reversed is True:
#         fig.update_xaxes(autorange="reversed")

#     return fig.show()

# def preview_mul_2coldf(
#     list_2coldf, colorway=plcolor.qualitative.Dark24, reversed=False
# ):
#     fig = go.Figure()
#     num_plots = list_2coldf[0].shape[1] // 2
#     for i in range(num_plots):
#         sample = "_".join(list_2coldf[0].columns[2 * i + 1].split("_")[:2])
#         color = colorway[i]
#         for j, df in enumerate(list_2coldf):
#             data = df.iloc[:, 2 * i : 2 * i + 2]
#             if j == 0:
#                 fig.add_scatter(
#                     x=data.iloc[:, 0],
#                     y=data.iloc[:, 1],
#                     mode="lines",
#                     name=data.columns[1],
#                     legendgroup=sample,
#                     line_color=color,
#                     showlegend=True,
#                 )
#             else:
#                 fig.add_scatter(
#                     x=data.iloc[:, 0],
#                     y=data.iloc[:, 1],
#                     mode="lines",
#                     name=data.columns[1],
#                     legendgroup=sample,
#                     line_color=color,
#                     showlegend=False,
#                 )
#     fig.update_layout(
#         hovermode="x unified",
#         hoverlabel_bgcolor="rgba(0,0,0,0)",
#         hoverlabel_bordercolor="rgba(0,0,0,0)",
#         height=1000,
#     )
#     if reversed is True:
#         fig.update_xaxes(autorange="reversed")
#     return fig.show()


# def preview_multiple_2coldf(
#     list_2coldf, colorway=plcolor.qualitative.Dark24, reversed=False
# ):
#     fig = go.Figure()
#     num_plots = list_2coldf[0].shape[1] // 2
#     for i in range(num_plots):
#         sample = "_".join(list_2coldf[0].columns[2 * i + 1].split("_")[:2])
#         color = colorway[i]
#         for j, df in enumerate(list_2coldf):
#             data = df.iloc[:, 2 * i : 2 * i + 2]
#             if j == 0:
#                 fig.add_scatter(
#                     x=data.iloc[:, 0],
#                     y=data.iloc[:, 1],
#                     mode="lines",
#                     name=data.columns[1],
#                     legendgroup=sample,
#                     line_color=color,
#                     showlegend=True,
#                 )
#             else:
#                 fig.add_scatter(
#                     x=data.iloc[:, 0],
#                     y=data.iloc[:, 1],
#                     mode="lines",
#                     name=data.columns[1],
#                     legendgroup=sample,
#                     line_color=color,
#                     showlegend=False,
#                 )
#     fig.update_layout(
#         hovermode="x unified",
#         hoverlabel_bgcolor="rgba(0,0,0,0)",
#         hoverlabel_bordercolor="rgba(0,0,0,0)",
#         height=1000,
#     )
#     if reversed is True:
#         fig.update_xaxes(autorange="reversed")
#     return fig.show()

## fit


def preview_data_fit(data, fit):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], mode="markers", name="Data")
    )
    fig.add_trace(
        go.Scatter(x=fit.iloc[:, 0], y=fit.iloc[:, 1], mode="lines", name="Fit")
    )
    fig.update_layout(
        # width=1000,
        height=750,
        legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
    )
    fig.show()


def preview_2coldf_guess(data, func_name, p0, mode="markers"):
    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    func = getattr(mm, func_name)
    y_guess = func(x_data, *p0)

    fig = go.Figure()
    fig.add_scatter(
        x=x_data,
        y=y_data,
        name=data.columns[1],
        mode=mode,
    )
    fig.add_scatter(
        x=x_data,
        y=y_guess,
        name="guess",
        mode="lines",
    )
    fig.update_layout(
        # width=1000,
        height=750,
        showlegend=True,
        legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
    )
    return fig.show()


# Display

## Generate Fig


def display_1col_df(df, mode="lines"):
    # df.index used as xaxis, each column in df is a set of data
    fig = go.Figure()
    for col in df.columns:
        name = df.columns[col]
        fig.add_scatter(
            x=df.index, y=df[col], name=name, mode=mode, line=dict(width=2.5)
        )
    return fig


def display_2col_df(df, mode="lines"):
    fig = go.Figure()
    for i in range(int(len(df.columns) / 2)):
        if type(df.columns) == pd.core.indexes.base.Index:
            name = df.columns[2 * i + 1]
        elif type(df.columns) == pd.core.indexes.multi.MultiIndex:
            name = df.columns[2 * i + 1][-1]
        # name = df.columns[2 * i + 1]
        fig.add_scatter(
            x=df.iloc[:, 2 * i],
            y=df.iloc[:, 2 * i + 1],
            name=name,
            showlegend=True,
            mode=mode,
            line_width=2.5,
        )
    return fig


# def display_fitted_kinetics(df, mode="circle", circle_size=0.5, line_width=0.5):
#     # a set of data consists four columns, the first two columns are raw data, the other two columns are the fitted data
#     fig = go.Figure()
#     for i in range(int(len(df.columns) / 4)):
#         data = df.iloc[:, i * 4 : i * 4 + 4]
#         name1 = data.columns[1]
#         name2 = data.columns[3]
#         if mode == "circle":
#             fig.add_scatter(
#                 x=data.iloc[:, 0],
#                 y=data.iloc[:, 1],  # raw data
#                 name=name1,
#                 mode="markers",
#                 marker=dict(
#                     symbol="circle-open",
#                     size=circle_size,
#                     line=dict(width=1),
#                     color=color,
#                 ),
#                 opacity=1,
#             )
#         elif mode == "line":
#             fig.add_scatter(
#                 x=data.iloc[:, 0],
#                 y=data.iloc[:, 1],  # raw data
#                 name=name1,
#                 mode="lines",
#                 marker=dict(line=dict(width=line_width), color=color),
#                 opacity=0.6,
#             )
#         fig.add_scatter(
#             x=data.iloc[:, 2],
#             y=data.iloc[:, 3],  # fitted data
#             name=name2,
#             mode="lines",
#             marker=dict(line=dict(width=3), color=color),
#         )
#     return fig

## Steady-State Spectroscopy

## Quantum Yield


def display_quantumyied(df_1col, legend_x=0.75, width=None, height=1000, yrange=None):
    num_cols = df_1col.shape[0]
    if width is None:
        width = num_cols * 150
    else:
        width = width
    height = height
    fig = go.Figure()
    for col in df_1col.columns:
        fig.add_bar(
            x=df_1col.index.values.tolist(),
            y=df_1col[col],
            name=col,
            text=df_1col[col].map(mf.float_to_percentage),
            textposition="auto",
        )
    fig.update_layout(
        width=width,
        height=height,
        barmode="stack",
        legend=dict(x=0.5, y=1, orientation="h", xanchor="center", yanchor="bottom"),
        yaxis_title_text="Relative Quantum Yield",
        # yaxis_range=[0, 1],
        # uniformtext_minsize=20,
        # uniformtext_mode="show",
        font=dict(family="Arial", size=20, color="black"),
    )
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    return fig


## ab


def display_ab(
    df_abs,
    xlimit=[None, 20, 10],
    # ylimit=[None, 0.2, 0.1],
    ylimit=[None, None, None],
    mode="lines",
    norm=True,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    colorway=plcolor.qualitative.Set1,
):
    fig = go.Figure()
    for i in range(0, int(len(df_abs.columns) / 2)):
        if legendtext is not None:
            name = legendtext[i]
        else:
            name = df_abs.columns[2 * i + 1]
        fig.add_trace(
            go.Scatter(
                x=df_abs.iloc[:, 2 * i],
                y=df_abs.iloc[:, 2 * i + 1],
                name=name,
                showlegend=True,
                mode=mode,
                line_width=2.5,
                line_shape="spline",
                hovertemplate="%{y:.2f}",
            )
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="Absorbance",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        width=1140,
        height=880,
        margin=dict(autoexpand=False, l=100, r=40, t=40, b=90),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            yanchor="top",
            x=0.95,
            y=0.95,
        ),
        colorway=colorway,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if norm is True:
        fig.update_layout(
            yaxis_title_text="Norm. Absorbance", yaxis_dtick=0.2, yaxis_minor_dtick=0.1
        )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    return fig


def display_ab_wl(
    df_ab_wl,
    xlimit=[None, 20, 10],
    # ylimit=[None, 0.2, 0.1],
    ylimit=[None, None, None],
    mode="lines",
    norm=True,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    colorway=plcolor.qualitative.Set1,
):
    fig = go.Figure()
    for i in range(0, int(len(df_ab_wl.columns) / 2)):
        if legendtext is not None:
            name = legendtext[i]
        else:
            name = df_ab_wl.columns[2 * i + 1]
        fig.add_trace(
            go.Scatter(
                x=df_ab_wl.iloc[:, 2 * i],
                y=df_ab_wl.iloc[:, 2 * i + 1],
                name=name,
                showlegend=True,
                mode=mode,
                line_width=2.5,
                line_shape="spline",
                hovertemplate="%{y:.2f}",
            )
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="Absorbance",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        width=1140,
        height=880,
        margin=dict(autoexpand=False, l=100, r=40, t=40, b=90),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            yanchor="top",
            x=0.95,
            y=0.95,
        ),
        colorway=colorway,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if norm is True:
        fig.update_layout(
            yaxis_title_text="Norm. Absorbance", yaxis_dtick=0.2, yaxis_minor_dtick=0.1
        )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    return fig


def display_ab_wn(
    df_ab_wl,
    xlimit=[None, 2, 1],
    # ylimit=[None, 0.2, 0.1],
    ylimit=[None, None, None],
    mode="lines",
    norm=True,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    reversed=True,
    colorway=plcolor.qualitative.Set1,
):
    fig = go.Figure()
    for i in range(0, int(len(df_ab_wl.columns) / 2)):
        if legendtext is None:
            name = df_ab_wl.columns[2 * i + 1]
        else:
            name = legendtext[i]
        fig.add_trace(
            go.Scatter(
                x=10**4 / df_ab_wl.iloc[:, 2 * i],  # convert to 10^-3 cm^-1
                y=df_ab_wl.iloc[:, 2 * i + 1],
                name=name,
                showlegend=True,
                mode=mode,
                line_width=2.5,
                line_shape="spline",
                hovertemplate="%{x:.3f},%{y:.2f}",
            )
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavenumber (\u00D710\u00b3 cm\u207b\u00b9)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="Absorbance",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        width=1140,
        height=880,
        margin=dict(autoexpand=False, l=100, r=40, t=40, b=90),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            yanchor="top",
            x=0.95,
            y=0.95,
        ),
        colorway=colorway,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if norm is True:
        fig.update_layout(
            yaxis_title_text="Norm. Absorbance", yaxis_dtick=0.2, yaxis_minor_dtick=0.1
        )
    if reversed is True:
        fig.update_xaxes(autorange="reversed")
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    return fig


def display_ab_wn_wl(
    df_ab_wl,
    xlimit1=[None, 2, 1],
    xlimit2=[None, 20, 10],
    ylimit=[None, None, None],
    mode="lines",
    norm=True,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    reversed=True,
    colorway=plcolor.qualitative.Set1,
):
    fig = go.Figure()
    for i in range(0, int(len(df_ab_wl.columns) / 2)):
        if legendtext is None:
            name = df_ab_wl.columns[2 * i + 1]
        else:
            name = legendtext[i]
        # generate the trace in wavenumber axis
        fig.add_trace(
            go.Scatter(
                x=10**4 / df_ab_wl.iloc[:, 2 * i],  # convert to 10^-3 cm^-1
                xaxis="x1",
                y=df_ab_wl.iloc[:, 2 * i + 1],
                yaxis="y1",
                name=name,
                showlegend=True,
                mode=mode,
                line_width=2.5,
                line_shape="spline",
                hovertemplate="%{y:.2f}",
            )
        )
        # generate the trace in wavelength axis
        fig.add_trace(
            go.Scatter(
                x=df_ab_wl.iloc[:, 2 * i],
                xaxis="x2",
                y=df_ab_wl.iloc[:, 2 * i + 1],
                yaxis="y1",
                name=name,
                showlegend=False,
                mode=mode,
                visible=True,
                line_width=2.5,
                line_shape="spline",
                hovertemplate="%{x2:.0f},%{y1:.2f}",
            )
        )
    fig.update_layout(
        xaxis1=dict(
            title=dict(
                text="Wavenumber (\u00D710\u00b3 cm\u207b\u00b9)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            # autorange="reversed",
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        xaxis2=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            # autorange="reversed",
            overlaying="x1",
            side="top",
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="Absorbance",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        width=1140,
        height=880,
        margin=dict(autoexpand=False, l=100, r=40, t=40, b=90),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            yanchor="top",
            x=0.95,
            y=0.95,
        ),
        colorway=colorway,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if norm is True:
        fig.update_layout(
            yaxis_title_text="Norm. Absorbance", yaxis_dtick=0.2, yaxis_minor_dtick=0.1
        )
    if xlimit1[0] is not None:
        fig.update_layout(xaxis1_range=xlimit1[0])
        fig.update_layout(
            xaxis2_range=[10**4 / xlimit1[0][0], 10**4 / xlimit1[0][1]]
        )
    if xlimit1[1] is not None:
        fig.update_layout(xaxis1_dtick=xlimit1[1])
    if xlimit1[2] is not None:
        fig.update_layout(xaxis1_minor_dtick=xlimit1[2])
    if xlimit2[1] is not None:
        fig.update_layout(xaxis2_dtick=xlimit2[1])
    if xlimit2[2] is not None:
        fig.update_layout(xaxis2_minor_dtick=xlimit2[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    return fig


def display_ab_bwl_twn(
    df_ab_wl,
    xlimit=[None, 20, 10],
    xlimit2=[None, 2000, 1000],
    ylimit=[None, 0.2, 0.1],
    legendtext=None,
):
    fig, ax = plt.subplots(
        # figsize=(3.33, mf.calculate_height(3.33, 3/2)),
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)),
        # figsize=(3.33, mf.calculate_height(3.33, 5/4)),
        # figsize=(3.33, mf.calculate_height(3.33, 16/9)),
        # figsize=(3.33, mf.calculate_height(3.33, 16/10)),
        dpi=600,
        layout="constrained",
    )

    for i in range(0, int(len(df_ab_wl.columns) / 2)):
        if legendtext is None:
            label = df_ab_wl.columns[2 * i + 1]
        else:
            label = legendtext[i]
        ax.plot(
            df_ab_wl.iloc[:, 2 * i],
            df_ab_wl.iloc[:, 2 * i + 1],
            label=label,
            linewidth=0.5,
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("Absorbance", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.1f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    secax = ax.secondary_xaxis(
        "top", functions=(mf.wavelength_to_wavenumber, mf.wavenumber_to_wavelength)
    )
    secax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)
    secax.spines["top"].set_linewidth(0.5)

    # ax.set_title('Absorption Spectra')
    ax.legend(
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc="upper right"
        # loc=5
    )

    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
    # plt.tight_layout()
    return fig


def display_ab_bwn_twl(
    df_ab_wl,
    xlimit=[None, 1, 0.5],
    xlimit2=[None, 100, 10],
    ylimit=[[-0.05, 1.05], 0.2, 0.1],
    legendtitle=None,
    legendtext=None,
    format_legendtext=None,  # mf.format_STEng
    legendposition="upper right",
    gridon=True,
    norm=True,
    colorscheme=plt.cm.tab10,
    figwidth=3.33,
    # figheight=3,
):
    fig, ax = plt.subplots(
        # figsize=(figwidth, figheight), dpi=600, layout="constrained"
        figsize=(figwidth, mf.calculate_height(3.33, 4 / 3)),
        dpi=600,
        layout="constrained",
    )

    for i in range(0, int(len(df_ab_wl.columns) / 2)):
        if legendtext is None:
            if format_legendtext is None:
                label = df_ab_wl.columns[2 * i + 1]
            else:
                label = format_legendtext(df_ab_wl.columns[2 * i + 1])
                # fomat = getattr(mf, format_legendtext)
                # label = fomat(df_ab_wl.columns[2 * i + 1])
        else:
            label = legendtext[i]
        nm = df_ab_wl.iloc[:, 2 * i]
        e3cm1 = 10**4 / nm
        ax.plot(
            # 10**7 / df_ab_wl.iloc[:, 2 * i],
            e3cm1,
            df_ab_wl.iloc[:, 2 * i + 1],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    # ax.set_xlabel("Wavenumber (\u00d710\u00b3 cm$^{-1}$)", fontsize=8, labelpad=2)
    ax.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)
    # ax.xaxis.set_major_formatter("{x:.0f}")
    # ax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    # ax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    if norm is True:
        ax.set_ylabel("Norm. Absorbance", fontsize=8, labelpad=2)
    else:
        ax.set_ylabel("Absorbance", fontsize=8, labelpad=2)

    # ax.yaxis.set_major_formatter("{x:.1f}")
    # ax.tick_params(axis="y", which="major", labelsize=6, width=0.5, length=2, pad=1)
    # ax.tick_params(axis="y", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    # ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3), useMathText=True)
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 2), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(6)

    # ax.set_title('Absorption Spectra')
    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    secax = ax.secondary_xaxis("top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)
    secax.spines["top"].set_linewidth(0.5)

    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    # plt.gca().invert_xaxis()
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
    # plt.tight_layout()
    return fig


def display_ab_line(df, xlimit=None, ylimit=None):
    fig = display_2col_df(df)
    fig.update_layout(
        template="ggplot2",
        # width=800,
        # height=600,
        width=1000,
        height=750,
        font=dict(family="Arial", color="black", size=22),
        # title=dict(
        #         text="Absorption Spectra",
        #         x=0.5,
        #         xanchor='center',
        #         y=0.8,
        #         yanchor='top',
        #         font=dict(size=24),
        # ),
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
            ),
            dtick=50,
            minor=dict(
                showgrid=True,
                tickcolor="black",
                ticklen=5,
                dtick=25,
                tickwidth=1,
                ticks="outside",
            ),
        ),
        yaxis=dict(
            title=dict(
                text="Absorbance (a.u.)",
            ),
            dtick=0.2,
            # tickformat = '.0f',
            minor=dict(
                showgrid=True,
                tickcolor="black",
                ticklen=5,
                dtick=0.1,
                tickwidth=1,
                ticks="outside",
            ),
        ),
        legend=dict(
            xanchor="right",
            x=0.965,
            yanchor="top",
            y=0.95,
            # title=dict(text="Sample name",), font=dict(size=22),
        ),
        # colorway=[
        #         '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
        #         '#00FFFF', '#FFA500', '#800080', '#008000', '#000080',
        #         '#800000', '#FFC0CB', '#ADD8E6', '#FF69B4', '#800000',
        #         '#008080', '#00FF7F', '#FFD700', '#FF6347', '#FF8C00',
        #         '#000000', '#000000', '#000000', '#000000', '#000000',
        #         '#000000', '#000000', '#000000', '#000000', '#000000',
        # ],
    )
    if xlimit is not None:
        fig.update_xaxes(range=[xlimit[0], xlimit[1]], dtick=xlimit[2])
    if ylimit is not None:
        fig.update_yaxes(range=[ylimit[0], ylimit[1]], dtick=ylimit[2])
    return fig


## pl


def display_pl_bwn_twl(
    df_2col,
    xlimit=[None, 1, 0.5],
    xlimit2=[None, 100, 10],
    ylimit=[None, None, None],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    format_legendtext=None,
    legendposition="upper right",
    gridon=False,
    norm=False,
    axvline_positions=None,  # list of wavenumbers
    colorscheme=plt.cm.tab10,
    yscale=0,
):
    fig, ax = plt.subplots(
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)),
        dpi=600,
        layout="tight",
        # layout="constrained"
    )

    for i in range(0, int(len(df_2col.columns) / 2)):
        if legendtext is None:
            if format_legendtext is None:
                label = df_2col.columns[2 * i + 1]
            else:
                label = format_legendtext(df_2col.columns[2 * i + 1])
            # label = mf.assignsample_lncom(df_2col.columns[2 * i + 1])
        else:
            label = legendtext[i]
        nm = df_2col.iloc[:, 2 * i]
        e3cm1 = 10**4 / nm
        ax.plot(
            # 10**7 / df_2col.iloc[:, 2 * i],
            e3cm1,
            df_2col.iloc[:, 2 * i + 1] / 10**yscale,
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    # ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    ax.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)

    if norm is True:
        ax.set_ylabel("Norm. PL Intensity", fontsize=8, labelpad=2)
    else:
        if yscale == 0:
            ax.set_ylabel("PL Intensity (Counts)", fontsize=8, labelpad=2)
        else:
            ax.set_ylabel(
                "PL Intensity (\u00d710$^{" + str(yscale) + "}$ Counts)",
                fontsize=8,
                labelpad=2,
            )

    # ax.ticklabel_format(
    #     axis="x",
    #     style="sci",
    #     scilimits=(3, 3),
    #     # scilimits=(0,0),
    #     useMathText=True,
    # )
    ax.xaxis.get_offset_text().set_fontsize(6)
    ax.xaxis.get_offset_text().set_position((1, -1))
    # ax.xaxis.set_major_formatter("{x:.0f}")

    ax.ticklabel_format(
        axis="y",
        style="plain",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.yaxis.get_offset_text().set_position((-0.06, 1))
    # ax.yaxis.set_major_formatter("{x:.0f}")

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )
    # legend.get_title().set_fontsize(6)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    secax = ax.secondary_xaxis("top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))

    if axvline_positions is not None:
        for position in axvline_positions:
            ax.axvline(
                x=position, color="black", linewidth=0.5, linestyle="dotted", alpha=0.5
            )

    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext)

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_pl(
    df_pl,
    xlimit=[None, 50, 25],
    ylimit=[None, None, None],
    mode="lines",
    norm=True,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    colorway=plcolor.qualitative.Set1,
):
    fig = go.Figure()
    for i in range(0, int(len(df_pl.columns) / 2)):
        if legendtext is not None:
            name = legendtext[i]
        else:
            name = df_pl.columns[2 * i + 1]
        fig.add_trace(
            go.Scatter(
                x=df_pl.iloc[:, 2 * i],
                y=df_pl.iloc[:, 2 * i + 1],
                name=name,
                showlegend=True,
                mode=mode,
                line_width=2.5,
                line_shape="spline",
                hovertemplate="%{y:.2f}",
            )
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="PL Intensity (Counts)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        width=1140,
        height=880,
        margin=dict(autoexpand=False, l=100, r=40, t=40, b=90),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            yanchor="top",
            x=0.95,
            y=0.95,
        ),
        colorway=colorway,
        hovermode="x unified",
        hoverlabel_bgcolor="rgba(0,0,0,0)",
        hoverlabel_bordercolor="rgba(0,0,0,0)",
    )
    if norm is True:
        fig.update_layout(
            yaxis_title_text="Norm. PL Intensity",
            yaxis_dtick=0.2,
            yaxis_minor_dtick=0.1,
        )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    return fig


def display_pl_line(df, xlimit=None, ylimit=None):
    fig = display_2col_df(df)
    fig.update_layout(
        template="ggplot2",
        width=1000,
        height=750,
        font=dict(family="Arial", color="black", size=22),
        title=dict(
            # text="Emission Spectra",
            x=0.5,
            xanchor="center",
            y=0.8,
            yanchor="top",
            font=dict(size=22),
        ),
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
            ),
            dtick=50,
            minor=dict(
                showgrid=True,
                tickcolor="black",
                ticklen=5,
                dtick=25,
                tickwidth=1,
                ticks="outside",
            ),
        ),
        yaxis=dict(
            title=dict(
                text="PL Intensity (Counts)",
            ),
            tickformat=".0f",
            # dtick=2,
            minor=dict(
                showgrid=True,
                tickcolor="black",
                ticklen=5,
                # dtick=0.5,
                tickwidth=1,
                ticks="outside",
            ),
        ),
        legend=dict(
            xanchor="right",
            x=0.965,
            yanchor="top",
            y=0.95,
            title=dict(
                text="Sample name",
            ),
            font=dict(size=22),
        ),
    )
    if xlimit is not None:
        fig.update_xaxes(range=[xlimit[0], xlimit[1]], dtick=xlimit[2])
    if ylimit is not None:
        fig.update_yaxes(range=[ylimit[0], ylimit[1]], dtick=ylimit[2])
    return fig


## ex


def display_ex_line(df):
    fig = display_2col_df(df)
    fig.update_layout(
        width=950,
        height=712.5,
        xaxis_title="Excitation Wavelength (nm)",
        xaxis_dtick=50,
        yaxis_title="PL Intensity (a.u.)",
        legend=dict(
            xanchor="left",
            x=0,
            yanchor="bottom",
            y=1,
        ),
        hovermode="closest",
    )
    return fig


## abpl


def display_abpl_1sample(
    df_abs,
    df_pl,
    xlimit=[None, 50, 25],
    ylimit=[None, None, None],
    # mode="lines",
    normab=True,
    normpl=True,
    titletext=None,
    titleposition=None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_abs.iloc[:, 0],
            xaxis="x",
            y=df_abs.iloc[:, 1],
            yaxis="y1",
            name=df_abs.columns[1],
            showlegend=True,
            mode="lines",
            line_width=3.5,
            line_shape="spline",
            line_color="#274575",
            hovertemplate="%{x},%{y:.2f}",
            # fill="tozeroy",
            # fillcolor="#889dbf",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_pl.iloc[:, 0],
            xaxis="x",
            y=df_pl.iloc[:, 1],
            yaxis="y2",
            name=df_pl.columns[1],
            showlegend=True,
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            line_shape="spline",
            hovertemplate="%{x},%{y:.2f}",
            # fill="tozeroy",
            # fillcolor="#df7c78",
        )
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # type="linear",
            # xaxis_tickvals=[0.1, 1, 10, 100, 1000, 10000],
            # xaxis_ticktext=['10^-1', '10^0', '10^1', '10^2', '10^3', '10^4'],
        ),
        yaxis1=dict(
            title=dict(
                text="Absorbance",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # color="red",
            # tickformat='.0f',
        ),
        yaxis2=dict(
            title=dict(
                text="PL Intensity",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            overlaying="y",
            side="right",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # color="blue",
            # tickfont=dict(color="blue")
        ),
        template="none",
        width=1200,
        height=890,
        margin=dict(l=100, r=100, t=40, b=100),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    if normab is True:
        fig.update_layout(
            yaxis1_title_text="Norm. Absorbance",
            yaxis1_dtick=0.2,
            yaxis1_minor_dtick=0.1,
        )
    if normpl is True:
        fig.update_layout(
            yaxis2_title_text="Norm. PL Intensity",
            yaxis2_dtick=0.2,
            yaxis2_minor_dtick=0.1,
        )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext, showlegend=False)
    if titleposition is not None:
        fig.update_layout(
            title_x=titleposition[0],
            title_y=titleposition[1],
        )
    return fig


def display_sabpl(
    df_abs,
    df_pl,
    xlimit=[None, 50, 25],
    ylimit=[None, None, None],
    # mode="lines",
    normab=True,
    normpl=True,
    titletext=None,
    titleposition=None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_abs.iloc[:, 0],
            xaxis="x",
            y=df_abs.iloc[:, 1],
            yaxis="y1",
            name=df_abs.columns[1],
            showlegend=True,
            mode="lines",
            line_width=3.5,
            line_shape="spline",
            line_color="#274575",
            hovertemplate="%{x},%{y:.2f}",
            # fill="tozeroy",
            # fillcolor="#889dbf",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_pl.iloc[:, 0],
            xaxis="x",
            y=df_pl.iloc[:, 1],
            yaxis="y2",
            name=df_pl.columns[1],
            showlegend=True,
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            line_shape="spline",
            hovertemplate="%{x},%{y:.2f}",
            # fill="tozeroy",
            # fillcolor="#df7c78",
        )
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # type="linear",
            # xaxis_tickvals=[0.1, 1, 10, 100, 1000, 10000],
            # xaxis_ticktext=['10^-1', '10^0', '10^1', '10^2', '10^3', '10^4'],
        ),
        yaxis1=dict(
            title=dict(
                text="Absorbance",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # color="red",
            # tickformat='.0f',
        ),
        yaxis2=dict(
            title=dict(
                text="PL Intensity",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            overlaying="y",
            side="right",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # color="blue",
            # tickfont=dict(color="blue")
        ),
        template="none",
        width=1200,
        height=890,
        margin=dict(l=100, r=100, t=40, b=100),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    if normab is True:
        fig.update_layout(
            yaxis1_title_text="Norm. Absorbance",
            yaxis1_dtick=0.2,
            yaxis1_minor_dtick=0.1,
        )
    if normpl is True:
        fig.update_layout(
            yaxis2_title_text="Norm. PL Intensity",
            yaxis2_dtick=0.2,
            yaxis2_minor_dtick=0.1,
        )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext, showlegend=False)
    if titleposition is not None:
        fig.update_layout(
            title_x=titleposition[0],
            title_y=titleposition[1],
        )
    return fig


def display_abpl_mulsample(
    df_ab_wl,
    df_pl_wl,
    xlimit=[None, 50, 25],
    y1limit=[None, None, None],
    y2limit=[None, None, None],
    # mode="lines",
    normab=True,
    normpl=True,
    titletext=None,
    titleposition=None,
    colorway=plcolor.qualitative.Dark24,
):
    fig = go.Figure()
    for i, color in zip(range(0, int(df_ab_wl.shape[1] / 2)), colorway):
        sample_solvent = "_".join(df_ab_wl.columns[2 * i + 1].split("_")[:2])
        ab = df_ab_wl.iloc[:, 2 * i : 2 * i + 2]
        pl = df_pl_wl.iloc[:, 2 * i : 2 * i + 2]
        fig.add_trace(
            go.Scatter(
                x=ab.iloc[:, 0],
                xaxis="x",
                y=ab.iloc[:, 1],
                yaxis="y1",
                name=ab.columns[1],
                legendgroup=sample_solvent,
                # legendgrouptitle_text=sample_solvent,
                showlegend=True,
                mode="lines",
                line_width=3.5,
                line_shape="spline",
                line_color=color,
                hovertemplate="%{x},%{y:.2f}",
                # fill="tozeroy",
                # fillcolor="#889dbf",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pl.iloc[:, 0],
                xaxis="x",
                y=pl.iloc[:, 1],
                yaxis="y2",
                name=pl.columns[1],
                legendgroup=sample_solvent,
                # legendgrouptitle_text=sample_solvent,
                showlegend=False,
                mode="lines",
                line_color=color,
                line_width=3.5,
                line_shape="spline",
                hovertemplate="%{x},%{y:.2f}",
                # fill="tozeroy",
                # fillcolor="#df7c78",
            )
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            color="black",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # type="linear",
            # xaxis_tickvals=[0.1, 1, 10, 100, 1000, 10000],
            # xaxis_ticktext=['10^-1', '10^0', '10^1', '10^2', '10^3', '10^4'],
        ),
        yaxis1=dict(
            title=dict(
                text="Absorbance",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # color="red",
            # tickformat='.0f',
        ),
        yaxis2=dict(
            title=dict(
                text="PL Intensity",
                standoff=0,
                font_size=32,
            ),
            showline=True,
            overlaying="y",
            side="right",
            mirror=False,
            zeroline=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
            # color="blue",
            # tickfont=dict(color="blue")
        ),
        template="none",
        width=1200,
        height=890,
        margin=dict(l=100, r=100, t=40, b=100),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="right",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    if normab is True:
        fig.update_layout(
            yaxis1_title_text="Norm. Absorbance",
            yaxis1_dtick=0.2,
            yaxis1_minor_dtick=0.1,
        )
    if normpl is True:
        fig.update_layout(
            yaxis2_title_text="Norm. PL Intensity",
            yaxis2_dtick=0.2,
            yaxis2_minor_dtick=0.1,
        )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if y1limit[0] is not None:
        fig.update_yaxes(range=y1limit[0])
    if y1limit[1] is not None:
        fig.update_yaxes(dtick=y1limit[1])
    if y1limit[2] is not None:
        fig.update_yaxes(minor_dtick=y1limit[2])
    if titletext is not None:
        fig.update_layout(title_text=titletext, showlegend=False)
    if titleposition is not None:
        fig.update_layout(
            title_x=titleposition[0],
            title_y=titleposition[1],
        )
    return fig


def display_abpl_bwl_twn(
    df_ab_wl,
    df_pl_wl,
    xlimit=[None, 100, 20],
    xlimit2=[None, 5000, 1000],
    ylimit=[[-0.05, 1.05], 0.2, 0.1],
    ylimit2=[[-0.05, 1.05], 0.2, 0.1],
    legendtext=None,
    norm=True,
    colorscheme=plt.cm.tab20,
):
    fig, ax1 = plt.subplots(
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
    )
    ax2 = ax1.twinx()

    for i in range(0, int(len(df_ab_wl.columns) / 2)):
        if legendtext is None:
            label = mf.assignsample_lncom(df_ab_wl.columns[2 * i + 1])
        else:
            label = legendtext[i]
        ax1.plot(
            df_ab_wl.iloc[:, 2 * i],
            df_ab_wl.iloc[:, 2 * i + 1],
            label=label,
            linewidth=0.7,
            linestyle="dashdot",
            color=colorscheme(i),
        )
        ax2.plot(
            df_pl_wl.iloc[:, 2 * i],
            df_pl_wl.iloc[:, 2 * i + 1],
            label=label,
            linewidth=0.7,
            linestyle="solid",
            color=colorscheme(i),
        )

    # ax1.set_title('Steady-State Spectra', fontsize=10, pad=2)

    ax1.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=2)
    if norm is False:
        ax1.set_ylabel("Absorbance", fontsize=8, labelpad=2)
        ax2.set_ylabel("Photoluminescence Intensity", fontsize=8, labelpad=2)
    else:
        ax1.set_ylabel("Norm. Absorbance", fontsize=8, labelpad=2)
        # ax2.set_ylabel("Norm. PL. Intensity", fontsize=8, labelpad=2)
        ax2.set_ylabel("Norm. Photoluminescence", fontsize=8, labelpad=2)

    ax1.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax1.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax1.tick_params(axis="y", right=False)
    ax1.xaxis.set_major_formatter("{x:.0f}")
    ax1.yaxis.set_major_formatter("{x:.1f}")
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)
    ax1.legend(
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc="upper right",
    )

    secax = ax1.secondary_xaxis(
        "top", functions=(mf.wavelength_to_wavenumber, mf.wavenumber_to_wavelength)
    )
    # secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax2.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax2.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax2.tick_params(axis="y", left=False)
    ax2.xaxis.set_major_formatter("{x:.0f}")
    ax2.yaxis.set_major_formatter("{x:.1f}")
    for axis in ["top", "bottom", "left", "right"]:
        ax2.spines[axis].set_linewidth(0.5)
    ax2.legend(
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc="upper right",
    )

    if xlimit[0] is not None:
        ax1.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))

    if ylimit[0] is not None:
        ax1.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if ylimit2[0] is not None:
        ax2.set_ylim(ylimit[0])
    if ylimit2[1] is not None:
        ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit2[2] is not None:
        ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    # ax1.set_ylim(-0.05, 1.05)
    # ax2.set_ylim(-0.05, 1.05)
    # plt.gca().invert_xaxis()
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
    # plt.tight_layout()
    return fig


def display_abpl_bwn_twl(
    df_ab_wl,
    df_pl_wl,
    xlimit=[None, 1, 0.5],
    xlimit2=[None, 100, 10],
    ylimit=[[-0.05, 1.05], 0.2, 0.1],
    ylimit2=[[-0.05, 1.05], 0.2, 0.1],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    format_legendtext=None,
    legendposition="upper right",
    normab=True,
    ab_scale=0,
    normpl=True,
    pl_scale=0,
    gridon=True,
    colorscheme=plt.cm.tab10,
    figwidth=3.33,
):
    fig, ax1 = plt.subplots(
        figsize=(figwidth, mf.calculate_height(3.33, 4 / 3)),
        dpi=600,
        layout="constrained",
    )
    ax2 = ax1.twinx()

    for i in range(0, int(len(df_ab_wl.columns) / 2)):
        # if legendtext is None:
        #     # label = mf.assignsample_lncom(df_ab_wl.columns[2 * i + 1])
        #     label = df_ab_wl.columns[2 * i + 1]
        # else:
        #     label = legendtext[i]
        if legendtext is None:
            if format_legendtext is None:
                label = df_ab_wl.columns[2 * i + 1]
            else:
                label = format_legendtext(df_ab_wl.columns[2 * i + 1])
                # fomat = getattr(mf, format_legendtext)
                # label = fomat(df_ab_wl.columns[2 * i + 1])
        else:
            label = legendtext[i]
        ax1.plot(
            10**4 / df_ab_wl.iloc[:, 2 * i],
            df_ab_wl.iloc[:, 2 * i + 1] / 10**ab_scale,
            label=label,
            linewidth=0.7,
            linestyle="dashdot",
            color=colorscheme(i),
        )
        ax2.plot(
            10**4 / df_pl_wl.iloc[:, 2 * i],
            df_pl_wl.iloc[:, 2 * i + 1] / 10**pl_scale,
            label=label,
            linewidth=0.7,
            linestyle="solid",
            color=colorscheme(i),
        )

    if titletext is not None:
        ax1.set_title(titletext, fontsize=8, pad=2)

    # ax1.set_xlabel(r"$ \mathrm{Wavenumber} \, (\times 10^3 \, \mathrm{cm}^{-1}) $", fontsize=8, labelpad=2) # \u00d710 \u00b3 cm$^{-1}
    # ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    ax1.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)

    if normab is False:
        if ab_scale == 0:
            ax1.set_ylabel("Absorbance", fontsize=8, labelpad=2)
        # else:
        #     ax1.set_ylabel("Absorbance (\u00d710$^{" + str(ab_scale) + "}$ Counts)", fontsize=8, labelpad=2)
    else:
        ax1.set_ylabel("Norm. Absorbance", fontsize=8, labelpad=2)

    if normpl is False:
        if pl_scale == 0:
            ax2.set_ylabel("PL Intensity (Counts)", fontsize=8, labelpad=2)
        else:
            ax2.set_ylabel(
                "PL Intensity (\u00d710$^{" + str(pl_scale) + "}$ Counts)",
                fontsize=8,
                labelpad=2,
            )
    else:
        ax2.set_ylabel("Norm. PL Intensity (Counts)", fontsize=8, labelpad=2)

    ax1.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax1.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax1.tick_params(axis="y", right=False)
    ax1.xaxis.set_major_formatter("{x:.0f}")
    ax1.yaxis.set_major_formatter("{x:.1f}")
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)
    # ax1.legend(
    #     title=legendtitle,
    #     title_fontsize=8,
    #     fontsize=6,
    #     frameon=False,
    #     facecolor="none",
    #     edgecolor="none",
    #     handlelength=1.5,
    #     loc=legendposition,
    # )

    if gridon is True:
        ax1.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax1.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    secax = ax1.secondary_xaxis("top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax2.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax2.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax2.tick_params(axis="y", left=False)
    ax2.xaxis.set_major_formatter("{x:.0f}")
    ax2.yaxis.set_major_formatter("{x:.1f}")
    for axis in ["top", "bottom", "left", "right"]:
        ax2.spines[axis].set_linewidth(0.5)
    ax2.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    if xlimit[0] is not None:
        ax1.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))

    if ylimit[0] is not None:
        ax1.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if ylimit2[0] is not None:
        ax2.set_ylim(ylimit[0])
    if ylimit2[1] is not None:
        ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit2[2] is not None:
        ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    # ax1.set_ylim(-0.05, 1.05)
    # ax2.set_ylim(-0.05, 1.05)
    # plt.gca().invert_xaxis()
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
    # plt.tight_layout()
    # plt.grid(True, which="both", color='gray', linestyle='solid', linewidth=0.1, alpha=0.2)

    # plt.xticks(plt.xticks()[0], [str(int(xtick/1000)) for xtick in plt.xticks()[0]]) # convert 50000 to 50

    return fig


# Time-Resolved Photoluminescence Spectra

## TCSPC


def display_tcspc_line(df):
    fig = display_2col_df(df)
    fig.update_layout(
        template="ggplot2",
        width=950,
        height=712.5,
        xaxis_title="Time (ns)",
        xaxis_range=(0, 50),
        xaxis_dtick=10,
        xaxis_minor=dict(
            showgrid=True,
            tickcolor="black",
            ticklen=5,
            dtick=5,
            tickwidth=1,
            ticks="outside",
        ),
        yaxis_title="Counts",
        legend=dict(
            xanchor="left",
            x=0,
            yanchor="bottom",
            y=1,
        ),
    )
    return fig


# def display_fitted_tcspc(df):
#     fig = display_fitted_kinetics(df)
#     fig.update_layout(
#                     template="ggplot2",
#                     width=950,
#                     height=712.5,
#                     font=dict(family="Arial", color="black", size=22),
#                     title=dict(
#                             text="fs-TA Kinetics",
#                             x=0.5,
#                             xanchor='center',
#                             y=0.8,
#                             yanchor='top',
#                             # font=dict(size=24),
#                     ),
#                     xaxis=dict(
#                             title=dict(
#                                     text="Time (ns)",
#                             ),
#                             type="linear",
#                             range=(0, 50),
#                             dtick=10,
#                             minor=dict(
#                                     showgrid=True,
#                                     tickcolor="black",
#                                     ticklen=5,
#                                     # tick0=0,
#                                     # dtick=100,
#                                     tickwidth=1,
#                                     ticks="outside",
#                             ),
#                     ),
#                     yaxis=dict(
#                             title=dict(
#                                     text="Counts",
#                             ),
#                             # tickformat = '.0f',
#                             # dtick=2,
#                             minor=dict(
#                                     showgrid=True,
#                                     tickcolor="black",
#                                     ticklen=5,
#                                     # dtick=0.5,
#                                     tickwidth=1,
#                                     ticks="outside",
#                             ),
#                     ),
#                     legend=dict(xanchor="right", x=0.965, yanchor="top", y=0.95, title=dict(text="Sample name",), font=dict(size=22),),
#     )
#     return fig


# def display_fitted_tcspc(df_fitted_tcspc, legendname):
#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             x=df_fitted_tcspc.iloc[:, 0],
#             xaxis="x",
#             y=df_fitted_tcspc.iloc[:, 1],
#             yaxis="y",
#             name=df_fitted_tcspc.columns[1],
#             showlegend=False,
#             mode="markers",
#             marker=dict(
#                 # symbol="circle-open",
#                 size=5,
#                 line=dict(width=1),
#                 color="whitesmoke",
#             ),
#             opacity=1,
#             hovertemplate="%{x:.2f},%{y:.2f}",
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=df_fitted_tcspc.iloc[:, 2],
#             xaxis="x",
#             y=df_fitted_tcspc.iloc[:, 3],
#             yaxis="y",
#             name=legendname,
#             mode="lines",
#             marker=dict(line=dict(width=5), color="#eb2027"),
#             hovertemplate="%{x:.2f},%{y:.2f}",
#         )
#     )
#     fig.update_layout(
#         template="none",
#         width=1000,
#         height=750,
#         font=dict(family="Arial", color="black", size=23),
#         title=dict(
#             # text=f"{dfabs.columns[1]}",
#             x=0.5,
#             xanchor="center",
#             y=0.8,
#             yanchor="top",
#         ),
#         xaxis=dict(
#             title=dict(text="Time (ns)"),
#             showline=True,
#             color="black",
#             zeroline=False,
#             linewidth=2,
#             linecolor="black",
#             range=[-5, 40],
#             dtick=10,
#             ticks="outside",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2,
#             showgrid=False,
#             minor=dict(
#                 dtick=5,
#                 ticks="outside",
#                 tickcolor="black",
#                 ticklen=5,
#                 tickwidth=2,
#                 showgrid=False,
#             ),
#         ),
#         yaxis=dict(
#             title=dict(text="Norm. Photoluminescence Intensity"),
#             showline=True,
#             zeroline=False,
#             linewidth=2,
#             range=[-0.05, 1.05],
#             dtick=0.2,
#             ticks="outside",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2,
#             showgrid=False,
#             minor=dict(
#                 dtick=0.1,
#                 ticks="outside",
#                 tickcolor="black",
#                 ticklen=5,
#                 tickwidth=2,
#                 showgrid=False,
#             ),
#         ),
#         showlegend=True,
#         legend=dict(
#             # title=dict(text="Sample name"),
#             # font=dict(family="Arial", size=28, color="black"),
#             xanchor="right",
#             x=0.95,
#             yanchor="top",
#             y=0.95,
#         ),
#     )
#     return fig


def display_fitted_tcspc(df_tcspc_fit, legend_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_tcspc_fit.iloc[:, 0],
            xaxis="x",
            y=df_tcspc_fit.iloc[:, 1],
            yaxis="y",
            name=df_tcspc_fit.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            hovertemplate="%{x:.2f},%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_tcspc_fit.iloc[:, 2],
            xaxis="x",
            y=df_tcspc_fit.iloc[:, 3],
            yaxis="y",
            name=legend_name,
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            hovertemplate="%{x:.2f},%{y:.2f}",
        )
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Time (ns)", standoff=10, font_size=32),
            showline=True,
            color="black",
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            range=[-5, 40],
            dtick=10,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                dtick=5,
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            # title=dict(text="Norm. Photoluminescence Intensity"),
            title=dict(text="Norm. PL Intensity", standoff=10, font_size=32),
            showline=True,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            range=[-0.05, 1.05],
            dtick=0.2,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                dtick=0.1,
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1120,
        height=850,
        margin=dict(l=100, r=20, t=10, b=90),
        # title=dict(
        #     # text=f"{dfabs.columns[1]}",
        #     # text=r"$\alpha_{1c} = 352 \pm 11 \text{ km s}^{-1}$",
        #     # text=r"$\large{\text{C124} \quad \tau \text{ = 4.62 ns}}$",
        #     # font_size=28,
        #     x=0.5,
        #     xanchor="center",
        #     y=0.8,
        #     yanchor="top",
        # ),
        showlegend=True,
        legend=dict(
            # title=dict(text="Sample name"),
            font_size=28,
            xanchor="right",
            x=0.9,
            yanchor="top",
            y=0.9,
        ),
    )
    return fig


def display_fitted_tcspc_irf(df_irf_fit, df_tcspc_fit, legend_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_irf_fit.iloc[:, 0],
            xaxis="x",
            y=df_irf_fit.iloc[:, 1],
            yaxis="y",
            name=df_irf_fit.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            hovertemplate="%{x:.2f},%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_irf_fit.iloc[:, 2],
            xaxis="x",
            y=df_irf_fit.iloc[:, 3],
            yaxis="y",
            name="IRF",
            mode="lines",
            line_color="#274575",
            line_width=3.5,
            hovertemplate="%{x:.2f},%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_tcspc_fit.iloc[:, 0],
            xaxis="x",
            y=df_tcspc_fit.iloc[:, 1],
            yaxis="y",
            name=df_tcspc_fit.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            hovertemplate="%{x:.2f},%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_tcspc_fit.iloc[:, 2],
            xaxis="x",
            y=df_tcspc_fit.iloc[:, 3],
            yaxis="y",
            name=legend_name,
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            hovertemplate="%{x:.2f},%{y:.2f}",
        )
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Time (ns)", standoff=10, font_size=32),
            showline=True,
            color="black",
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            range=[-5, 40],
            dtick=10,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                dtick=5,
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            # title=dict(text="Norm. Photoluminescence Intensity"),
            title=dict(text="Norm. PL Intensity (Counts)", standoff=10, font_size=32),
            showline=True,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            range=[-0.05, 1.05],
            dtick=0.2,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                dtick=0.1,
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1120,
        height=850,
        margin=dict(l=100, r=20, t=10, b=90),
        # title=dict(
        #     # text=f"{dfabs.columns[1]}",
        #     # text=r"$\alpha_{1c} = 352 \pm 11 \text{ km s}^{-1}$",
        #     # text=r"$\large{\text{C124} \quad \tau \text{ = 4.62 ns}}$",
        #     # font_size=28,
        #     x=0.5,
        #     xanchor="center",
        #     y=0.8,
        #     yanchor="top",
        # ),
        showlegend=True,
        legend=dict(
            # title=dict(text="Sample name"),
            font_size=28,
            xanchor="right",
            x=0.9,
            yanchor="top",
            y=0.9,
        ),
    )
    return fig


## Streak Camera


## nspl (Flash Photolysis)

### spectra


def display_nspl_spectra_bwn_twl(
    df_nspls_wl,  # df_1col
    xlimit=[None, 5, 1],
    xlimit2=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    ylimit=[None, None, None],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    zeroline=False,
    gridon=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
    # xscale=0,
    yscale=0,
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    nm = df_nspls_wl.index.values
    e3cm1 = 10**4 / nm
    for i in range(0, len(df_nspls_wl.columns)):
        if legendtext is None:
            label = df_nspls_wl.columns[i]
            try:
                label = mf.formalize_nsta_delaytime(float(label))
            except ValueError:
                label = label
            # label = mf.formalize_nsta_delaytime(df_nspls_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            e3cm1,
            df_nspls_wl.iloc[:, i] / 10**yscale,
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    # ax.ticklabel_format(
    #     axis="x",
    #     style="plain",
    #     scilimits=(3, 3),
    #     # scilimits=(0,0),
    #     useMathText=True,
    # )
    # ax.xaxis.get_offset_text().set_fontsize(6)
    # ax.xaxis.get_offset_text().set_position((1, -1))
    # ax.xaxis.set_major_formatter("{x:.0f}")

    # secax = ax.secondary_xaxis("top", functions=(mf.wn2wl, mf.wl2wn))
    secax = ax.secondary_xaxis("top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.wl2wn(xlimit2[0][0]),
            mf.wl2wn(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    # if norm is True:
    #     ax.set_ylabel("Norm. PL Intensity", fontsize=8, labelpad=2)
    # else:
    #     ax.set_ylabel("PL Intensity (Counts)", fontsize=8, labelpad=2)
    if yscale == 0:
        ax.set_ylabel("PL Intensity (Counts)", fontsize=8, labelpad=2)
    else:
        ax.set_ylabel(
            "PL Intensity (\u00d710$^{" + str(yscale) + "}$ Counts)",
            fontsize=8,
            labelpad=2,
        )
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.ticklabel_format(
        axis="both",
        style="plain",
        # scilimits=(0,0),
        # useMathText=True,
    )
    # ax.yaxis.get_offset_text().set_fontsize(6)
    # ax.yaxis.get_offset_text().set_position((-0.08, 0))
    # ax.yaxis.set_major_formatter("{x:.0f}")
    if zeroline is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_nspl_spectra(
    df_1col,
    xlimit=[None, 50, 25],
    ylimit=[None, None, None],
    legendtitle=None,
    legendposition=None,
    titletext=None,
    colorscheme=plcolor.qualitative.Dark24,
    ab=None,
    ab_scale=1,
    pl=None,
    pl_scale=1,
):
    fig = go.Figure()

    if ab is not None:
        fig.add_scatter(
            x=ab.iloc[:, 0],
            y=-ab_scale * ab.iloc[:, 1],
            name="Abs.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(110,190,235,0.2)"
            # line_color="rgba(0,0,0,0)",
            # fillcolor="#6ebeeb"
        )
    if pl is not None:
        fig.add_scatter(
            x=pl.iloc[:, 0],
            y=-pl_scale * pl.iloc[:, 1],
            name="PL.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(240,158,162,0.2)",
            # fillcolor="#f09ea2"
        )

    df_1col.columns = [float(col) for col in df_1col.columns]
    df_1col.index = [float(row) for row in df_1col.index]
    wavelength = df_1col.index
    for col, color in zip(df_1col.columns, colorscheme):
        if col < 1000:
            name = f"{col:.0f} ns"
        elif col >= 1000 and col < 1000000:
            name = col / 1000
            name = f"{name:.1f} \u00B5s"
        elif col >= 1000000 and col < 1000000000:
            name = col / 1000000
            name = f"{name:.1f} ms"

        fig.add_scatter(
            x=wavelength,
            y=df_1col[col],
            name=name,
            mode="lines",
            line_width=2.5,
            line_color=color,
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                font_size=32,
                standoff=0,
            ),
            showline=True,
            mirror=False,
            linewidth=2.5,
            color="black",
            showgrid=False,
            zeroline=False,
            ticks="outside",
            ticklen=10,
            tickwidth=2.5,
            tickcolor="black",
            minor=dict(
                ticks="outside",
                showgrid=False,
                ticklen=5,
                tickwidth=2.5,
                tickcolor="black",
            ),
        ),
        yaxis=dict(
            title=dict(
                text="PL Intensity (Counts)",
                font_size=32,
                standoff=0,
            ),
            showline=True,
            mirror=False,
            linewidth=2.5,
            color="black",
            showgrid=False,
            zeroline=False,
            zerolinecolor="gray",
            zerolinewidth=1.5,
            ticks="outside",
            ticklen=10,
            tickwidth=2.5,
            tickcolor="black",
            # tickformat = '.0f',
            minor=dict(
                ticks="outside",
                showgrid=False,
                ticklen=5,
                tickwidth=2.5,
                tickcolor="black",
            ),
        ),
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1130,
        height=870,
        margin=dict(autoexpand=False, l=100, r=30, t=30, b=90),
        title=dict(
            text=titletext,
            font_size=32,
            x=0.5,
            xanchor="right",
            yanchor="top",
            y=0.9,
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
        ),
        colorway=rainbow7,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    return fig


### kinetics


def display_nspl_1colkinetics_fitted_linear(
    ls_df_1cols,  # the first 2 columns of df_1cols should represent data and fit, other columns could be stored, but not be used, such as the residual
    xshift=0,
    yshift=0,
    xscale=0,
    yscale=0,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    ynorm=False,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, df_1cols in enumerate(ls_df_1cols):
        time = df_1cols.index
        if xshift >= 0:
            time = time - xshift
        else:
            time = time + xshift
        if xscale == 0:
            time = time
            ax.set_xlabel("Time (ns)", fontsize=8, labelpad=2)
        elif xscale == 3:
            time = time / 1000
            ax.set_xlabel("Time (us)", fontsize=8, labelpad=2)
        elif xscale == 6:
            time = time / 1000000
            ax.set_xlabel("Time (ms)", fontsize=8, labelpad=2)

        if yshift >= 0:
            data = df_1cols.iloc[:, 0] - yshift
            fit = df_1cols.iloc[:, 1] - yshift
        else:
            data = df_1cols.iloc[:, 0] + yshift
            fit = df_1cols.iloc[:, 1] + yshift
        residual = df_1cols.iloc[:, 2]
        if yscale == 0:
            data = data
            fit = fit
        else:
            data = data / 10**yscale
            fit = fit / 10**yscale

        if legendtext is None:
            label = df_1cols.columns[0]
            # label = f"{round(df_1col.columns[0])} nm"
        else:
            label = legendtext[i]
            # label = f"{round(mf.wavelength_to_wavenumber(df_1col.columns[0]))} cm\u207b\u00b9"

        ax.scatter(
            time,
            data,
            s=2,
            alpha=0.5,
            linewidth=0.1,
            edgecolor="black",
            # facecolors=None,
            color=colorscheme(i),
        )
        ax.plot(
            time,
            fit,
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    if ynorm is True:
        ax.set_ylabel("Norm. PL Intensity", fontsize=8, labelpad=2)
    else:
        if yscale == 0:
            ax.set_ylabel("PL Intensity (Counts)", fontsize=8, labelpad=2)
        else:
            ax.set_ylabel(
                "PL Intensity (\u00d710$^{" + str(yscale) + "}$ Counts)",
                fontsize=8,
                labelpad=2,
            )
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax.ticklabel_format(axis="both", style="plain", scilimits=(-1, 2), useMathText=True)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][1])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )


def display_nspl_2colkinetics_fitted_linear(
    ls_df_2cols,  # df_2col with 6 columns, which represents the raw data, fitted data, and residual
    xshift=0,
    yshift=0,
    xscale=0,
    yscale=0,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    ynorm=False,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="lines",
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, df_2cols in enumerate(ls_df_2cols):
        df_copy = df_2cols.copy()
        if xshift >= 0:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] - xshift
        else:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] + xshift

        if yshift >= 0:
            df_copy.iloc[:, [1, 3]] = df_copy.iloc[:, [1, 3]] - yshift
        else:
            df_copy.iloc[:, [1, 3]] = df_copy.iloc[:, [1, 3]] + yshift

        if xscale == 0:
            ax.set_xlabel("Time (ns)", fontsize=8, labelpad=2)
        elif xscale == 3:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] / 1000
            ax.set_xlabel("Time (us)", fontsize=8, labelpad=2)
        elif xscale == 6:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] / 1000000
            ax.set_xlabel("Time (ms)", fontsize=8, labelpad=2)
        elif xscale == 9:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] / 1000000000
            ax.set_xlabel("Time (s)", fontsize=8, labelpad=2)

        if ynorm is True:
            ax.set_ylabel("Norm. PL Intensity", fontsize=8, labelpad=2)
        else:
            if yscale == 0:
                ax.set_ylabel("PL Intensity (Counts)", fontsize=8, labelpad=2)
            else:
                df_copy.iloc[:, [1, 3]] = df_copy.iloc[:, [1, 3]] / 10**yscale
                ax.set_ylabel(
                    "PL Intensity (\u00d710$^{" + str(yscale) + "}$ Counts)",
                    fontsize=8,
                    labelpad=2,
                )

        if legendtext is None:
            label = df_copy.columns[0]
            # label = f"{round(df_1col.columns[0])} nm"
        else:
            label = legendtext[i]
            # label = f"{round(mf.wavelength_to_wavenumber(df_1col.columns[0]))} cm\u207b\u00b9"

        # if mode == "lines":
        ax.plot(
            df_copy.iloc[:, 0],
            df_copy.iloc[:, 1],
            # label=label,
            alpha=0.5,
            linewidth=0.3,
            color=colorscheme(i),
        )
        # elif mode == "scatter":
        ax.scatter(
            df_copy.iloc[:, 0],
            df_copy.iloc[:, 1],
            s=2,
            alpha=0.5,
            linewidth=0.3,
            edgecolor="black",
            # facecolors=None,
            color=colorscheme(i),
        )
        ax.plot(
            df_copy.iloc[:, 2],
            df_copy.iloc[:, 3],
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax.ticklabel_format(axis="both", style="plain", scilimits=(-1, 2), useMathText=True)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][1])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )
    return fig


# Transient Absorption Spectra

## ns-TA

### spectra


def display_nsta_spectra_bwn_twl(
    df_nstas_wl,  # df_1col
    xlimit=[None, 1, 0.5],
    xlimit2=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    ylimit=[None, None, None],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    format_legendtext=None,
    legendposition="upper right",
    gridon=True,
    zeroline=False,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    nm = df_nstas_wl.index.values
    e3cm1 = 10**4 / nm
    for i in range(0, len(df_nstas_wl.columns)):
        if legendtext is None:
            if format_legendtext is None:
                label = df_nstas_wl.columns[i]
            else:
                label = format_legendtext(df_nstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            e3cm1,
            df_nstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis("top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.nm_to_e3cm1(xlimit2[0][0]),
            mf.nm_to_e3cm1(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    # ax.yaxis.set_major_formatter("{x:.0f}")
    if zeroline is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_nsta_spectra(
    df_1col,
    xlimit=[None, 50, 25],
    ylimit=[None, None, None],
    legendtitle=None,
    legendposition=None,
    titletext=None,
    ab=None,
    ab_scale=1,
    pl=None,
    pl_scale=1,
):
    fig = go.Figure()
    if ab is not None:
        fig.add_scatter(
            x=ab.iloc[:, 0],
            y=-ab_scale * ab.iloc[:, 1],
            name="Abs.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(110,190,235,0.2)"
            # line_color="rgba(0,0,0,0)",
            # fillcolor="#6ebeeb"
        )
    if pl is not None:
        fig.add_scatter(
            x=pl.iloc[:, 0],
            y=-pl_scale * pl.iloc[:, 1],
            name="PL.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(240,158,162,0.2)",
            # fillcolor="#f09ea2"
        )
    df_1col.columns = [float(col) for col in df_1col.columns]
    df_1col.index = [float(row) for row in df_1col.index]
    wavelength = df_1col.index
    for col, color in zip(df_1col.columns, rainbow7):
        if col < 1000:
            name = f"{col:.0f} ns"
        elif col >= 1000 and col < 1000000:
            name = col / 1000
            name = f"{name:.1f} \u00B5s"
        elif col >= 1000000 and col < 1000000000:
            name = col / 1000000
            name = f"{name:.1f} ms"
        fig.add_scatter(
            x=wavelength,
            y=df_1col[col],
            name=name,
            mode="lines",
            line_width=2.5,
            line_color=color,
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                font_size=32,
                standoff=0,
            ),
            showline=True,
            mirror=False,
            linewidth=2.5,
            color="black",
            showgrid=False,
            zeroline=False,
            ticks="outside",
            ticklen=10,
            tickwidth=2.5,
            tickcolor="black",
            minor=dict(
                ticks="outside",
                showgrid=False,
                ticklen=5,
                tickwidth=2.5,
                tickcolor="black",
            ),
        ),
        yaxis=dict(
            title=dict(
                text="\u0394A (mOD)",
                font_size=32,
                standoff=0,
            ),
            showline=True,
            mirror=False,
            linewidth=2.5,
            color="black",
            showgrid=False,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1.5,
            ticks="outside",
            ticklen=10,
            tickwidth=2.5,
            tickcolor="black",
            # tickformat = '.0f',
            minor=dict(
                ticks="outside",
                showgrid=False,
                ticklen=5,
                tickwidth=2.5,
                tickcolor="black",
            ),
        ),
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1130,
        height=870,
        margin=dict(autoexpand=False, l=100, r=30, t=30, b=90),
        title=dict(
            text=titletext,
            font_size=32,
            x=0.5,
            xanchor="right",
            yanchor="top",
            y=0.9,
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
        ),
        colorway=rainbow7,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    return fig


# def display_nsta_spectra(
#     df_scol,
#     xlimit=[None, 50, 25],
#     ylimit=[None, None, None],
#     legendtitle=None,
#     legendposition=None,
#     titletext=None,
#     ab=None,
#     ab_scale=1,
#     pl=None,
#     pl_scale=1,
# ):
#     fig = go.Figure()
#     if ab is not None:
#         fig.add_scatter(
#             x=ab.iloc[:, 0],
#             y=-ab_scale * ab.iloc[:, 1],
#             name="Abs.",
#             mode="lines",
#             line=dict(width=0),
#             fill="tozeroy",
#             fillcolor="rgba(110,190,235,0.2)"
#             # line_color="rgba(0,0,0,0)",
#             # fillcolor="#6ebeeb"
#         )
#     if pl is not None:
#         fig.add_scatter(
#             x=pl.iloc[:, 0],
#             y=-pl_scale * pl.iloc[:, 1],
#             name="PL.",
#             mode="lines",
#             line=dict(width=0),
#             fill="tozeroy",
#             fillcolor="rgba(240,158,162,0.2)",
#             # fillcolor="#f09ea2"
#         )
#     df_scol.columns = [float(col) for col in df_scol.columns]
#     df_scol.index = [float(row) for row in df_scol.index]
#     wavelength = df_scol.index
#     for col, color in zip(df_scol.columns, rainbow7):
#         if col < 1000:
#             name = f"{col:.0f} ns"
#         elif col >= 1000 and col < 1000000:
#             name = col / 1000
#             name = f"{name:.1f} \u00B5s"
#         elif col >= 1000000 and col < 1000000000:
#             name = col / 1000000
#             name = f"{name:.1f} ms"
#         fig.add_scatter(
#             x=wavelength,
#             y=df_scol[col],
#             name=name,
#             mode="lines",
#             line_width=2.5,
#             line_color=color,
#         )
#     fig.update_layout(
#         xaxis=dict(
#             title=dict(
#                 text="Wavelength (nm)",
#                 font_size=32,
#                 standoff=0,
#             ),
#             showline=True,
#             mirror=False,
#             linewidth=2.5,
#             color="black",
#             showgrid=False,
#             zeroline=False,
#             ticks="outside",
#             ticklen=10,
#             tickwidth=2.5,
#             tickcolor="black",
#             minor=dict(
#                 ticks="outside",
#                 showgrid=False,
#                 ticklen=5,
#                 tickwidth=2.5,
#                 tickcolor="black",
#             ),
#         ),
#         yaxis=dict(
#             title=dict(
#                 text="\u0394A (mOD)",
#                 font_size=32,
#                 standoff=0,
#             ),
#             showline=True,
#             mirror=False,
#             linewidth=2.5,
#             color="black",
#             showgrid=False,
#             zeroline=True,
#             zerolinecolor="gray",
#             zerolinewidth=1.5,
#             ticks="outside",
#             ticklen=10,
#             tickwidth=2.5,
#             tickcolor="black",
#             # tickformat = '.0f',
#             minor=dict(
#                 ticks="outside",
#                 showgrid=False,
#                 ticklen=5,
#                 tickwidth=2.5,
#                 tickcolor="black",
#             ),
#         ),
#         template="none",
#         font=dict(family="Arial", color="black", size=26),
#         width=1130,
#         height=870,
#         margin=dict(autoexpand=False, l=100, r=30, t=30, b=90),
#         title=dict(
#             text=titletext,
#             font_size=32,
#             x=0.5,
#             xanchor="right",
#             y=0.9,
#             yanchor="top",
#         ),
#         showlegend=True,
#         legend=dict(
#             title_font_size=32,
#             font_size=28,
#             xanchor="right",
#             x=0.95,
#             yanchor="top",
#             y=0.95,
#         ),
#         colorway=rainbow7,
#         hovermode="x unified",
#         hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
#     )
#     if xlimit[0] is not None:
#         fig.update_xaxes(range=xlimit[0])
#     if xlimit[1] is not None:
#         fig.update_xaxes(dtick=xlimit[1])
#     if xlimit[2] is not None:
#         fig.update_xaxes(minor_dtick=xlimit[2])
#     if ylimit[0] is not None:
#         fig.update_yaxes(range=ylimit[0])
#     if ylimit[1] is not None:
#         fig.update_yaxes(dtick=ylimit[1])
#     if ylimit[2] is not None:
#         fig.update_yaxes(minor_dtick=ylimit[2])
#     if legendtitle is not None:
#         fig.update_layout(legend_title_text=legendtitle)
#     if legendposition is not None:
#         fig.update_layout(
#             legend_x=legendposition[0],
#             legend_y=legendposition[1],
#         )
#     return fig

# def display_nsta_spectra(
#     df_scol,
#     xlimit=[None, 50, 25],
#     ylimit=[None, None, None],
#     legendtitle=None,
#     legendposition=None,
#     titletext=None,
#     ab=None,
#     ab_scale=1,
#     pl=None,
#     pl_scale=1,
# ):
#     fig = go.Figure()
#     if ab is not None:
#         fig.add_scatter(
#             x=ab.iloc[:, 0],
#             y=-ab_scale * ab.iloc[:, 1],
#             name="Abs.",
#             mode="lines",
#             line=dict(width=0),
#             fill="tozeroy",
#             fillcolor="rgba(110,190,235,0.2)"
#             # line_color="rgba(0,0,0,0)",
#             # fillcolor="#6ebeeb"
#         )
#     if pl is not None:
#         fig.add_scatter(
#             x=pl.iloc[:, 0],
#             y=-pl_scale * pl.iloc[:, 1],
#             name="PL.",
#             mode="lines",
#             line=dict(width=0),
#             fill="tozeroy",
#             fillcolor="rgba(240,158,162,0.2)",
#             # fillcolor="#f09ea2"
#         )
#     df_scol.columns = [float(col) for col in df_scol.columns]
#     df_scol.index = [float(row) for row in df_scol.index]
#     wavelength = df_scol.index
#     for col, color in zip(df_scol.columns, rainbow7):
#         if col < 1000:
#             name = f"{col:.0f} ns"
#         elif col >= 1000 and col < 1000000:
#             name = col / 1000
#             name = f"{name:.1f} \u00B5s"
#         elif col >= 1000000 and col < 1000000000:
#             name = col / 1000000
#             name = f"{name:.1f} ms"
#         fig.add_scatter(
#             x=wavelength,
#             y=df_scol[col],
#             name=name,
#             mode="lines",
#             line_width=2.5,
#             line_color=color,
#         )
#     fig.update_layout(
#         xaxis=dict(
#             title=dict(
#                 text="Wavelength (nm)",
#                 font_size=32,
#                 standoff=0,
#             ),
#             showline=True,
#             mirror=False,
#             linewidth=2.5,
#             color="black",
#             showgrid=False,
#             zeroline=False,
#             ticks="outside",
#             ticklen=10,
#             tickwidth=2.5,
#             tickcolor="black",
#             minor=dict(
#                 ticks="outside",
#                 showgrid=False,
#                 ticklen=5,
#                 tickwidth=2.5,
#                 tickcolor="black",
#             ),
#         ),
#         yaxis=dict(
#             title=dict(
#                 text="\u0394A (mOD)",
#                 font_size=32,
#                 standoff=0,
#             ),
#             showline=True,
#             mirror=False,
#             linewidth=2.5,
#             color="black",
#             showgrid=False,
#             zeroline=True,
#             zerolinecolor="gray",
#             zerolinewidth=1.5,
#             ticks="outside",
#             ticklen=10,
#             tickwidth=2.5,
#             tickcolor="black",
#             # tickformat = '.0f',
#             minor=dict(
#                 ticks="outside",
#                 showgrid=False,
#                 ticklen=5,
#                 tickwidth=2.5,
#                 tickcolor="black",
#             ),
#         ),
#         template="none",
#         font=dict(family="Arial", color="black", size=26),
#         width=1130,
#         height=870,
#         margin=dict(autoexpand=False, l=100, r=30, t=30, b=90),
#         title=dict(
#             text=titletext,
#             font_size=32,
#             x=0.5,
#             xanchor="right",
#             y=0.9,
#             yanchor="top",
#         ),
#         showlegend=True,
#         legend=dict(
#             title_font_size=32,
#             font_size=28,
#             xanchor="right",
#             x=0.95,
#             yanchor="top",
#             y=0.95,
#         ),
#         colorway=rainbow7,
#         hovermode="x unified",
#         hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
#     )
#     if xlimit[0] is not None:
#         fig.update_xaxes(range=xlimit[0])
#     if xlimit[1] is not None:
#         fig.update_xaxes(dtick=xlimit[1])
#     if xlimit[2] is not None:
#         fig.update_xaxes(minor_dtick=xlimit[2])
#     if ylimit[0] is not None:
#         fig.update_yaxes(range=ylimit[0])
#     if ylimit[1] is not None:
#         fig.update_yaxes(dtick=ylimit[1])
#     if ylimit[2] is not None:
#         fig.update_yaxes(minor_dtick=ylimit[2])
#     if legendtitle is not None:
#         fig.update_layout(legend_title_text=legendtitle)
#     if legendposition is not None:
#         fig.update_layout(
#             legend_x=legendposition[0],
#             legend_y=legendposition[1],
#         )
#     return fig


# def wash_nspl_spectra(filepaths):
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
#             # data[col] = data[col] * 1000  # don't change counts
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


# def display_nspl_spectra(
#     df_1col,
#     xlimit=[None, 50, 25],
#     ylimit=[None, None, None],
#     titletext=None,
#     legendtitle=None,
#     legendtext=None,
#     legendposition=None,
#     # ab=None,
#     # ab_scale=1,
#     pl=None,
#     pl_scale=1,
# ):
#     fig = go.Figure()
#     wavelength = df_1col.index
#     for col in df_1col.columns:
#         if col < 1000:  # < 1 us
#             name = f"{col:.0f} ns"
#         elif col == 1000:  # 1 nus
#             name = "1 us"
#         elif col > 1000 and col < 1000000:  # > 1 us & < 1 ms
#             name = col / 1000
#             name = f"{name:.0f} \u00B5s"
#         elif col == 1000000:  # 1 ms:
#             name = "1 ms"
#         elif col > 1000000 and col < 1000000000:  # > 1 ms & < 1 s
#             name = col / 1000000
#             name = f"{name:.1f} ms"
#         fig.add_scatter(
#             x=wavelength,
#             y=df_1col[col],
#             name=name,
#             mode="lines",
#             line_width=2.5,
#             # line_color=color,
#         )
#     fig.update_layout(
#         xaxis=dict(
#             title=dict(
#                 text="Wavelength (nm)",
#                 font_size=32,
#                 standoff=0,
#             ),
#             showline=True,
#             mirror=False,
#             linewidth=2.5,
#             color="black",
#             showgrid=False,
#             zeroline=False,
#             ticks="outside",
#             ticklen=10,
#             tickwidth=2.5,
#             tickcolor="black",
#             minor=dict(
#                 ticks="outside",
#                 showgrid=False,
#                 ticklen=5,
#                 tickwidth=2.5,
#                 tickcolor="black",
#             ),
#         ),
#         yaxis=dict(
#             title=dict(
#                 text="PL Intensity (Counts)",
#                 font_size=32,
#                 standoff=0,
#             ),
#             showline=True,
#             mirror=False,
#             linewidth=2.5,
#             color="black",
#             showgrid=False,
#             zeroline=False,
#             zerolinecolor="gray",
#             zerolinewidth=1.5,
#             ticks="outside",
#             ticklen=10,
#             tickwidth=2.5,
#             tickcolor="black",
#             # tickformat = '.0f',
#             minor=dict(
#                 ticks="outside",
#                 showgrid=False,
#                 ticklen=5,
#                 tickwidth=2.5,
#                 tickcolor="black",
#             ),
#         ),
#         template="none",
#         font=dict(family="Arial", color="black", size=26),
#         width=1130,
#         height=870,
#         margin=dict(autoexpand=False, l=100, r=30, t=30, b=90),
#         title=dict(
#             text=titletext,
#             font_size=32,
#             x=0.5,
#             xanchor="right",
#             y=0.9,
#             yanchor="top",
#         ),
#         showlegend=True,
#         legend=dict(
#             title_font_size=32,
#             font_size=28,
#             xanchor="right",
#             x=0.95,
#             yanchor="top",
#             y=0.95,
#         ),
#         colorway=rainbow7,
#         hovermode="x unified",
#         hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
#     )
#     if xlimit[0] is not None:
#         fig.update_xaxes(range=xlimit[0])
#     if xlimit[1] is not None:
#         fig.update_xaxes(dtick=xlimit[1])
#     if xlimit[2] is not None:
#         fig.update_xaxes(minor_dtick=xlimit[2])
#     if ylimit[0] is not None:
#         fig.update_yaxes(range=ylimit[0])
#     if ylimit[1] is not None:
#         fig.update_yaxes(dtick=ylimit[1])
#     if ylimit[2] is not None:
#         fig.update_yaxes(minor_dtick=ylimit[2])
#     if legendtitle is not None:
#         fig.update_layout(legend_title_text=legendtitle)
#     if legendposition is not None:
#         fig.update_layout(
#             legend_x=legendposition[0],
#             legend_y=legendposition[1],
#         )
#     if pl is not None:
#         fig.add_scatter(
#             x=pl.iloc[:, 0],
#             y=-pl_scale * pl.iloc[:, 1],
#             name="PL.",
#             mode="lines",
#             line=dict(width=0),
#             fill="tozeroy",
#             fillcolor="rgba(240,158,162,0.2)",
#         )
#     return fig


### kinetics


def display_nsta_1colkinetics(
    df_1col,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    mode="lines",
    colorway=None,
):
    fig = go.Figure()
    time = df_1col.index
    if time[-1] < 1000:
        xaxis_title_text = "Time (ns)"
    elif time[-1] >= 1000 and time[-1] < 1000000:
        time = time / 1000
        xaxis_title_text = "Time (us)"
    elif time[-1] >= 1000000 and time[-1] < 1000000000:
        time = time / 1000000
        xaxis_title_text = "Time (ms)"
    for col in df_1col.columns:
        fig.add_scatter(
            x=time, y=df_1col[col], name=f"{col} nm", mode=mode, line_width=1.5
        )
    # for i in range(int(df_scol.shape[1] / 2)):
    #     timeaxis = df_scol.iloc[:, 2 * i].values
    #     timeaxis = timeaxis / 1000
    #     kinetic = df_scol.iloc[:, 2 * i + 1].values
    #     name = df_scol.columns[2 * i + 1]
    #     fig.add_scatter(x=timeaxis, y=kinetic, name=name, mode=mode)
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=xaxis_title_text,
                standoff=0,
                font_size=32,
            ),
            color="black",
            showline=True,
            # mirror=True,
            zeroline=False,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            tickformat="s",
            minor=dict(
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="\u0394A (mOD)",
                standoff=0,
                font_size=32,
            ),
            color="black",
            # tick0=0,
            showline=True,
            # mirror=True,
            zeroline=False,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            # tickformat="s",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1130,
        height=890,
        margin=dict(l=100, r=40, t=40, b=90),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            itemsizing="constant",
            # itemwidth=50,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
        ),
    )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if colorway is not None:
        fig.update_layout(colorway=colorway)
    return fig


# def display_nsta_kinetics_1col(
#     df_1col,
#     xlimit=[None, None, None],
#     ylimit=[None, None, None],
#     titletext=None,
#     legendtitle=None,
#     legendtext=None,
#     legendposition=None,
#     mode="lines",
#     colorway=None,
# ):
#     fig = go.Figure()
#     time = df_1col.index
#     if time[-1] < 1000:
#         xaxis_title_text = "Time (ns)"
#     elif time[-1] >= 1000 and time[-1] < 1000000:
#         time = time / 1000
#         xaxis_title_text = "Time (us)"
#     elif time[-1] >= 1000000 and time[-1] < 1000000000:
#         time = time / 1000000
#         xaxis_title_text = "Time (ms)"
#     for col in df_1col.columns:
#         fig.add_scatter(
#             x=time, y=df_1col[col], name=f"{col} nm", mode=mode, line_width=1.5
#         )
#     # for i in range(int(df_scol.shape[1] / 2)):
#     #     timeaxis = df_scol.iloc[:, 2 * i].values
#     #     timeaxis = timeaxis / 1000
#     #     kinetic = df_scol.iloc[:, 2 * i + 1].values
#     #     name = df_scol.columns[2 * i + 1]
#     #     fig.add_scatter(x=timeaxis, y=kinetic, name=name, mode=mode)
#     fig.update_layout(
#         xaxis=dict(
#             title=dict(
#                 text=xaxis_title_text,
#                 standoff=0,
#                 font_size=32,
#             ),
#             color="black",
#             showline=True,
#             # mirror=True,
#             zeroline=False,
#             showgrid=False,
#             linewidth=2.5,
#             ticks="outside",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2.5,
#             tickformat="s",
#             minor=dict(
#                 ticks="outside",
#                 ticklen=5,
#                 tickwidth=2.5,
#                 showgrid=False,
#             ),
#         ),
#         yaxis=dict(
#             title=dict(
#                 text="\u0394A (mOD)",
#                 standoff=0,
#                 font_size=32,
#             ),
#             color="black",
#             # tick0=0,
#             showline=True,
#             # mirror=True,
#             zeroline=False,
#             showgrid=False,
#             linewidth=2.5,
#             ticks="outside",
#             tickcolor="black",
#             # tickformat="s",
#             ticklen=10,
#             tickwidth=2.5,
#             minor=dict(
#                 ticks="outside",
#                 ticklen=5,
#                 tickwidth=2.5,
#                 showgrid=False,
#             ),
#         ),
#         template="none",
#         font=dict(family="Arial", color="black", size=26),
#         width=1130,
#         height=890,
#         margin=dict(l=100, r=40, t=40, b=90),
#         title=dict(
#             font_size=32,
#             x=0.5,
#             xanchor="center",
#             y=0.95,
#             yanchor="top",
#         ),
#         showlegend=True,
#         legend=dict(
#             title_font_size=32,
#             font_size=28,
#             itemsizing="constant",
#             # itemwidth=50,
#             xanchor="right",
#             x=0.95,
#             yanchor="top",
#             y=0.95,
#         ),
#     )
#     if xlimit[0] is not None:
#         fig.update_xaxes(range=xlimit[0])
#     if xlimit[1] is not None:
#         fig.update_xaxes(dtick=xlimit[1])
#     if xlimit[2] is not None:
#         fig.update_xaxes(minor_dtick=xlimit[2])
#     if ylimit[0] is not None:
#         fig.update_yaxes(range=ylimit[0])
#     if ylimit[1] is not None:
#         fig.update_yaxes(dtick=ylimit[1])
#     if ylimit[2] is not None:
#         fig.update_yaxes(minor_dtick=ylimit[2])
#     if legendtitle is not None:
#         fig.update_layout(legend_title_text=legendtitle)
#     if legendposition is not None:
#         fig.update_layout(
#             legend_x=legendposition[0],
#             legend_y=legendposition[1],
#         )
#     if titletext is not None:
#         fig.update_layout(title_text=titletext)
#     if colorway is not None:
#         fig.update_layout(colorway=colorway)
#     return fig


def display_nsta_1colkinetics_fitted_linear(
    ls_df_1cols,  # 3 columns, data, fit, residual
    xshift=0,
    xscale=0,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    yshift=0,
    # yscale=0,
    titletext=None,
    legendtitle=None,
    legendtext=None,  # list of strings
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    # showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, df_1cols in enumerate(ls_df_1cols):
        time = df_1cols.index
        if xshift >= 0:
            time = time - xshift
        else:
            time = time + xshift
        if xscale == 0:
            time = time
            ax.set_xlabel("Time (ns)", fontsize=8, labelpad=2)
        elif xscale == 3:
            time = time / 1000
            ax.set_xlabel("Time (us)", fontsize=8, labelpad=2)
        elif xscale == 6:
            time = time / 1000000
            ax.set_xlabel("Time (ms)", fontsize=8, labelpad=2)

        if yshift >= 0:
            data = df_1cols.iloc[:, 0] - yshift
            fit = df_1cols.iloc[:, 1] - yshift
        else:
            data = df_1cols.iloc[:, 0] + yshift
            fit = df_1cols.iloc[:, 1] + yshift
        residual = df_1cols.iloc[:, 2]

        if legendtext is None:
            label = df_1cols.columns[0]
            # label = f"{round(df_1col.columns[0])} nm"
        else:
            label = legendtext[i]
            # label = f"{round(mf.wavelength_to_wavenumber(df_1col.columns[0]))} cm\u207b\u00b9"

        if mode == "scatter":
            ax.scatter(
                time,
                data,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
                # facecolors=None,
                color=colorscheme(i),
            )
        elif mode == "line":
            ax.plot(
                time,
                data,
                linewidth=1,
                alpha=0.5,
                color=colorscheme(i),
            )
        ax.plot(
            time,
            fit,
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][1])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    return fig


def display_nsta_1colkinetics_fitted_symlog(
    ls_df_1col,  # 3 columns, data, fit, residual
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    # showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, df_1col in enumerate(ls_df_1col):
        time = df_1col.index  # .values
        data = df_1col.iloc[:, 0]
        fit = df_1col.iloc[:, 1]
        residual = df_1col.iloc[:, 2]
        if legendtext is None:
            label = df_1col.columns[0]
            # label = f"{round(df_1col.columns[0])} nm"
        else:
            label = legendtext[i]
            # label = f"{round(mf.wavelength_to_wavenumber(df_1col.columns[0]))} cm\u207b\u00b9"
        ax.scatter(
            time,
            data,
            # label=f"{round(df_1col.columns[0])} nm",
            s=2,
            alpha=0.5,
            linewidth=0.1,
            edgecolor="black",
            # facecolors=None,
            color=colorscheme(i),
        )
        ax.plot(
            time,
            fit,
            # label=f"{round(df_1col.columns[0])} nm",
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.set_xlabel("Time (ns)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    return fig


def display_nsta_2colkinetics_linear(
    df_2col,  # each df_2col has multiple sets of kinetics
    xshift=0,
    xscale=0,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    yshift=0,
    # yscale=0,
    titletext=None,
    legendtitle=None,
    legendtext=None,  # list of strings
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",  # "line",
    # showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    num_set = int(df_2col.shape[1] / 2)
    for i in range(num_set):
        time = df_2col.iloc[:, 2 * i]
        if xshift >= 0:
            time = time - xshift
        else:
            time = time + xshift
        if xscale == 0:
            time = time
            ax.set_xlabel("Time (ns)", fontsize=8, labelpad=2)
        elif xscale == 3:
            time = time / 1000
            ax.set_xlabel("Time (us)", fontsize=8, labelpad=2)
        elif xscale == 6:
            time = time / 1000000
            ax.set_xlabel("Time (ms)", fontsize=8, labelpad=2)

        kinetics = df_2col.iloc[:, 2 * i + 1]
        if yshift >= 0:
            kinetics = kinetics - yshift
        else:
            kinetics = kinetics + yshift

        if legendtext is None:
            label = df_2col.columns[1]
        else:
            label = legendtext[i]

        if mode == "scatter":
            ax.scatter(
                time,
                kinetics,
                s=2,
                alpha=0.5,
                label=label,
                linewidth=0.1,
                edgecolor="black",
                # facecolors=None,
                color=colorscheme(i),
            )
        elif mode == "line":
            ax.plot(
                time,
                kinetics,
                label=label,
                linewidth=1,
                color=colorscheme(i),
            )

    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][1])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)
    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)
    return fig


def display_nsta_2colkinetics_fitted_linear(
    ls_df_2cols,  # df_2col with 6 columns, which represents the raw data, fitted data, and residual
    xshift=0,
    yshift=0,
    xscale=0,
    yscale=0,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    ynorm=False,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="lines",
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, df_2cols in enumerate(ls_df_2cols):
        df_copy = df_2cols.copy()
        if xshift >= 0:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] - xshift
        else:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] + xshift

        if yshift >= 0:
            df_copy.iloc[:, [1, 3]] = df_copy.iloc[:, [1, 3]] - yshift
        else:
            df_copy.iloc[:, [1, 3]] = df_copy.iloc[:, [1, 3]] + yshift

        if xscale == 0:
            ax.set_xlabel("Time (ns)", fontsize=8, labelpad=2)
        elif xscale == 3:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] / 1000
            ax.set_xlabel("Time (us)", fontsize=8, labelpad=2)
        elif xscale == 6:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] / 1000000
            ax.set_xlabel("Time (ms)", fontsize=8, labelpad=2)
        elif xscale == 9:
            df_copy.iloc[:, [0, 2, 4]] = df_copy.iloc[:, [0, 2, 4]] / 1000000000
            ax.set_xlabel("Time (s)", fontsize=8, labelpad=2)

        if ynorm is True:
            ax.set_ylabel("Norm. \u0394A (mOD)", fontsize=8, labelpad=2)
        else:
            if yscale == 0:
                ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
            # else:
            #     df_copy.iloc[:, [1, 3]] = df_copy.iloc[:, [1, 3]] / 10**yscale
            #     ax.set_ylabel(
            #         "\u0394A (mOD) (\u00d710$^{" + str(yscale) + "}$ Counts)",
            #         fontsize=8,
            #         labelpad=2,
            #     )

        if legendtext is None:
            label = df_copy.columns[0]
            # label = f"{round(df_1col.columns[0])} nm"
        else:
            label = legendtext[i]
            # label = f"{round(mf.wavelength_to_wavenumber(df_1col.columns[0]))} cm\u207b\u00b9"

        # if mode == "lines":
        ax.plot(
            df_copy.iloc[:, 0],
            df_copy.iloc[:, 1],
            # label=label,
            alpha=0.5,
            linewidth=0.3,
            color=colorscheme(i),
        )
        # elif mode == "scatter":
        ax.scatter(
            df_copy.iloc[:, 0],
            df_copy.iloc[:, 1],
            s=2,
            alpha=0.5,
            linewidth=0.3,
            edgecolor="black",
            # facecolors=None,
            color=colorscheme(i),
        )
        ax.plot(
            df_copy.iloc[:, 2],
            df_copy.iloc[:, 3],
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    ax.ticklabel_format(axis="both", style="plain", scilimits=(-1, 2), useMathText=True)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][1])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )
    return fig


# def display_nsta_2colkinetics_fitted_linear(
#     df_2col,
#     xlimit=[None, None, None],
#     ylimit=[None, None, None],
#     titletext=None,
#     legendtitle=None,
#     legendtext=None,
#     legendposition=None,
#     mode="lines",
#     colorscheme=color_lnpet,
# ):
#     fig = go.Figure()
#     for i, df_set in enumerate(mf.split_dataframe(df_2col, num_cols=4)):
#         fig.add_scatter(
#             x=df_set.iloc[:, 0],
#             y=df_set.iloc[:, 1],
#             legendgroup=df_set.columns[1],
#             name=df_set.columns[1],
#             mode="lines",
#             line_width=1.5,
#             line_color=colorscheme[i],
#             showlegend=True,
#         )
#         fig.add_scatter(
#             x=df_set.iloc[:, 2],
#             y=df_set.iloc[:, 3],
#             legendgroup=df_set.columns[1],
#             # name=df_set.columns[1],
#             mode="lines",
#             line_width=2,
#             line_color="black",
#             showlegend=False,
#         )
#     # for col in df_2col.columns:
#     #     fig.add_scatter(
#     #         x=time, y=df_2col[col], name=f"{col} nm", mode=mode, line_width=1.5
#     #     )
#     # for i in range(int(df_scol.shape[1] / 2)):
#     #     timeaxis = df_scol.iloc[:, 2 * i].values
#     #     timeaxis = timeaxis / 1000
#     #     kinetic = df_scol.iloc[:, 2 * i + 1].values
#     #     name = df_scol.columns[2 * i + 1]
#     #     fig.add_scatter(x=timeaxis, y=kinetic, name=name, mode=mode)

#     # time_range = [df_2col.iloc[0, 0], df_2col.iloc[-1, 0]]
#     # if time_range[1] < 1000:
#     #     xaxis_titletext = "Time (ns)"
#     # elif time_range[1] >= 1000 and time_range[1] < 1000000:
#     #     time_range = time_range / 1000
#     #     xaxis_titletext = "Time (us)"
#     # elif time_range[1] >= 1000000 and time_range[1] < 1000000000:
#     #     time_range = time_range / 1000000
#     #     xaxis_titletext = "Time (ms)"
#     fig.update_layout(
#         xaxis=dict(
#             title=dict(
#                 text="Time (ns)",
#                 standoff=0,
#                 font_size=32,
#             ),
#             color="black",
#             showline=True,
#             # mirror=True,
#             zeroline=False,
#             showgrid=False,
#             linewidth=2.5,
#             ticks="outside",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2.5,
#             tickformat="s",
#             minor=dict(
#                 ticks="outside",
#                 ticklen=5,
#                 tickwidth=2.5,
#                 showgrid=False,
#             ),
#         ),
#         yaxis=dict(
#             title=dict(
#                 text="\u0394A (mOD)",
#                 standoff=0,
#                 font_size=32,
#             ),
#             color="black",
#             # tick0=0,
#             showline=True,
#             # mirror=True,
#             zeroline=False,
#             showgrid=False,
#             linewidth=2.5,
#             ticks="outside",
#             tickcolor="black",
#             # tickformat="s",
#             ticklen=10,
#             tickwidth=2.5,
#             minor=dict(
#                 ticks="outside",
#                 ticklen=5,
#                 tickwidth=2.5,
#                 showgrid=False,
#             ),
#         ),
#         template="none",
#         font=dict(family="Arial", color="black", size=26),
#         width=1130,
#         height=890,
#         margin=dict(l=100, r=40, t=40, b=90),
#         title=dict(
#             font_size=32,
#             x=0.5,
#             xanchor="center",
#             y=0.95,
#             yanchor="top",
#         ),
#         showlegend=True,
#         legend=dict(
#             title_font_size=32,
#             font_size=28,
#             itemsizing="constant",
#             # itemwidth=50,
#             xanchor="right",
#             x=0.95,
#             yanchor="top",
#             y=0.95,
#         ),
#     )

#     # if xlimit[0] is not None:
#     #     fig.update_xaxes(range=xlimit[0])
#     # if xlimit[1] is not None:
#     #     fig.update_xaxes(dtick=xlimit[1])
#     # if xlimit[2] is not None:
#     #     fig.update_xaxes(minor_dtick=xlimit[2])
#     # if ylimit[0] is not None:
#     #     fig.update_yaxes(range=ylimit[0])
#     # if ylimit[1] is not None:
#     #     fig.update_yaxes(dtick=ylimit[1])
#     # if ylimit[2] is not None:
#     #     fig.update_yaxes(minor_dtick=ylimit[2])
#     # if legendtitle is not None:
#     #     fig.update_layout(legend_title_text=legendtitle)
#     # if legendposition is not None:
#     #     fig.update_layout(
#     #         legend_x=legendposition[0],
#     #         legend_y=legendposition[1],
#     #     )
#     # if titletext is not None:
#     #     fig.update_layout(title_text=titletext)
#     # if colorway is not None:
#     #     fig.update_layout(colorway=colorway)
#     return fig

## fs-TA

### heatmap


def display_fsta_heatmap(
    df, xlimit=[300, 800, 50], ylimit=[-100, 8000, 1000], zlimit=[-5, 5, 1]
):
    delta_abs = df.T
    wavelength = delta_abs.columns
    time = delta_abs.index

    fig = go.Figure(
        go.Heatmap(
            x=wavelength,
            y=time,
            z=delta_abs,
            zmin=zlimit[0],
            zmax=zlimit[1],
            zsmooth="best",
            colorbar=dict(
                title=dict(
                    text="\u0394A (mOD)",
                    # text=r"$\Delta\text{Abs. (mOD)}$",
                    font=dict(color="black"),
                    side="right",
                ),
                orientation="v",
                # xanchor='left', x=1, yanchor='top', y=1,
                xpad=0,
                ypad=0,
                outlinecolor="black",
                outlinewidth=1,
                ticks="outside",
                ticklen=8,
            ),
            colorscale="RdBu_r",
        )
    )

    fig.update_layout(
        width=1000,
        height=750,
        # width=950,
        # height=712.5,
        # width=800,
        # height=600,
        font=dict(family="Arial", color="black", size=23),
    )

    fig.update_xaxes(
        title=dict(text="Wavelength (nm)"),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        tickcolor="black",
        ticks="outside",
        range=xlimit[:2],
        # tick0=350,
        dtick=xlimit[-1],
        tickformat="%0f",
        ticklen=8,
        minor=dict(
            # dtick=25,
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            tickcolor="black",
        ),
    )

    fig.update_yaxes(
        title=dict(text="Time (ps)"),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        tickcolor="black",
        ticks="outside",
        range=ylimit[:2],
        # tick0=0,
        dtick=ylimit[-1],
        tickformat="%0f",
        ticklen=8,
        minor=dict(
            # dtick=25,
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            tickcolor="black",
        ),
    )
    return fig


def display_fsta_heatmap_linear(
    df, xlimit=[300, 800, 50], ylimit=[-100, 8000, 1000], zlimit=[-5, 5, 1]
):
    delta_abs = df.T
    wavelength = delta_abs.columns
    time = delta_abs.index

    fig = go.Figure(
        go.Heatmap(
            x=wavelength,
            y=time,
            z=delta_abs,
            zmin=zlimit[0],
            zmax=zlimit[1],
            zsmooth=False,
            colorbar=dict(
                dtick=zlimit[2],
                title=dict(
                    text="\u0394A (mOD)",
                    font=dict(color="black"),
                    side="right",
                    # standoff=10
                ),
                orientation="v",
                xanchor="left",
                x=1.02,
                yanchor="top",
                y=1,
                xpad=0,
                ypad=0,
                xref="container",
                outlinecolor="black",
                outlinewidth=1,
                ticks="outside",
                ticklen=8,
            ),
            colorscale="RdBu_r",
        )
    )

    fig.update_layout(
        width=1000,
        height=750,
        margin=dict(l=80, r=90, t=10, b=80),
        font=dict(family="Arial", color="black", size=23),
    )

    fig.update_xaxes(
        title=dict(text="Wavelength (nm)", standoff=10),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        tickcolor="black",
        ticks="outside",
        range=xlimit[:2],
        # tick0=350,
        dtick=xlimit[-1],
        tickformat="%0f",
        ticklen=8,
        minor=dict(
            # dtick=25,
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            tickcolor="black",
        ),
    )

    fig.update_yaxes(
        title=dict(text="Time (ps)", standoff=10),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        tickcolor="black",
        ticks="outside",
        range=ylimit[:2],
        # tick0=0,
        dtick=ylimit[-1],
        # tickformat=',.2r',
        tickformat="~s",
        ticklen=8,
        minor=dict(
            # dtick=25,
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            tickcolor="black",
        ),
    )
    return fig


def display_fsta_heatmap_log(
    df,
    xlimit=[300, 800, 50],
    ylimit=[-0.7, 3.9, 1],  # 10^-0.7=0.2 ps
    zlimit=[-5, 5, 1],
    titletext=None,
    colorscale="jet",
    # ["turbo", "jet", "rainbow", "hsv", "edge", "spectral_r", "portland"]
):
    delta_abs = df.T
    wavelength = delta_abs.columns
    time = delta_abs.index

    fig = go.Figure(
        go.Heatmap(
            x=wavelength,
            y=time,
            z=delta_abs,
            zmin=zlimit[0],
            zmax=zlimit[1],
            # zsmooth="fast",
            colorbar=dict(
                title=dict(
                    text="\u0394A (mOD)",
                    font=dict(color="black", size=32),
                    side="right",
                    # standoff=10
                ),
                dtick=zlimit[2],
                orientation="v",
                # xanchor="left",
                # x=1.02,
                # yanchor="top",
                # y=1,
                xpad=0,
                ypad=0,
                # xref="container",
                len=1,
                # lenmode="fraction",
                outlinecolor="black",
                outlinewidth=2.5,
                # borderwidth=2.5,
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
            ),
            colorscale=colorscale,
        )
    )

    fig.update_xaxes(
        title=dict(text="Wavelength (nm)", standoff=0, font_size=32),
        showline=True,
        linecolor="black",
        linewidth=2.5,
        mirror=True,
        range=xlimit[:2],
        dtick=xlimit[-1],
        tick0=350,
        tickformat="%0f",
        ticklen=10,
        tickwidth=2.5,
        tickcolor="black",
        ticks="outside",
        minor=dict(
            dtick=xlimit[-1] / 5,
            ticklen=5,
            tickwidth=2.5,
            ticks="outside",
            tickcolor="black",
            showgrid=False,
        ),
    )
    fig.update_yaxes(
        title=dict(text="Time (ps)", standoff=0, font_size=32),
        showline=True,
        linecolor="black",
        linewidth=2.5,
        mirror=True,
        type="log",
        range=ylimit[:2],
        dtick=ylimit[-1],
        tick0=-1,
        ticklen=10,
        tickwidth=2.5,
        tickcolor="black",
        ticks="outside",
        # tickvals=[1, 10, 100, 1000],
        # ticktext=[f"10^{0}", f"10^{1}", f"10^{2}", f"10^{3}"],
        minor=dict(
            #     # tickmode="array",
            #     # tickvals=[
            #     #     0.2,
            #     #     0.3,
            #     #     0.4,
            #     #     0.5,
            #     #     0.6,
            #     #     0.7,
            #     #     0.8,
            #     #     0.9,
            #     #     2,
            #     #     3,
            #     #     4,
            #     #     5,
            #     #     6,
            #     #     7,
            #     #     8,
            #     #     9,
            #     #     20,
            #     #     30,
            #     #     40,
            #     #     50,
            #     #     60,
            #     #     70,
            #     #     80,
            #     #     90,
            #     #     200,
            #     #     300,
            #     #     400,
            #     #     500,
            #     #     600,
            #     #     700,
            #     #     800,
            #     #     900,
            #     #     2000,
            #     #     3000,
            #     #     4000,
            #     #     5000,
            #     #     6000,
            #     #     7000,
            #     #     8000,
            #     #     9000,
            #     # ],
            ticklen=5,
            tickwidth=2.5,
            ticks="outside",
            tickcolor="black",
        ),
    )
    fig.update_layout(
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1240,
        height=860,
        margin=dict(autoexpand=True, l=120, r=120, t=20, b=90),
        title=dict(
            text=titletext,
            font_size=32,
            xanchor="right",
            x=0.83,
            yanchor="top",
            y=0.9,
        ),
        showlegend=False,
    ),
    return fig


def display_fsta_heatmap_log_v2(
    df,
    xlimit=[None, 50, 25],
    ylimit=[None, 1, 0.1],
    zlimit=[None, 4, 2],
    titletext=None,
    colorscale="jet",
    # ["turbo", "jet", "rainbow", "hsv", "edge", "spectral_r", "portland"]
):
    delta_abs = df.T
    wavelength = delta_abs.columns
    time = delta_abs.index

    fig = go.Figure(
        go.Heatmap(
            x=wavelength,
            y=time,
            z=delta_abs,
            zmin=zlimit[0],
            zmax=zlimit[1],
            # zsmooth="fast",
            colorbar=dict(
                title=dict(
                    text="\u0394A (mOD)",
                    font=dict(color="black", size=32),
                    side="right",
                    # standoff=10
                ),
                dtick=zlimit[2],
                orientation="v",
                # xanchor="left",
                # x=1.02,
                # yanchor="top",
                # y=1,
                xpad=0,
                ypad=0,
                # xref="container",
                len=1,
                # lenmode="fraction",
                outlinecolor="black",
                outlinewidth=2.5,
                # borderwidth=2.5,
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
            ),
            colorscale=colorscale,
        )
    )

    fig.update_xaxes(
        title=dict(text="Wavelength (nm)", standoff=0, font_size=32),
        showline=True,
        linecolor="black",
        linewidth=2.5,
        mirror=True,
        tick0=350,
        tickformat="%0f",
        ticklen=10,
        tickwidth=2.5,
        tickcolor="black",
        ticks="outside",
        minor=dict(
            ticklen=5,
            tickwidth=2.5,
            ticks="outside",
            tickcolor="black",
            showgrid=False,
        ),
    )
    fig.update_yaxes(
        title=dict(text="Time (ps)", standoff=0, font_size=32),
        showline=True,
        linecolor="black",
        linewidth=2.5,
        mirror=True,
        type="log",
        tick0=-1,
        ticklen=10,
        tickwidth=2.5,
        tickcolor="black",
        ticks="outside",
        # tickvals=[1, 10, 100, 1000],
        # ticktext=[f"10^{0}", f"10^{1}", f"10^{2}", f"10^{3}"],
        minor=dict(
            #     # tickmode="array",
            #     # tickvals=[
            #     #     0.2,
            #     #     0.3,
            #     #     0.4,
            #     #     0.5,
            #     #     0.6,
            #     #     0.7,
            #     #     0.8,
            #     #     0.9,
            #     #     2,
            #     #     3,
            #     #     4,
            #     #     5,
            #     #     6,
            #     #     7,
            #     #     8,
            #     #     9,
            #     #     20,
            #     #     30,
            #     #     40,
            #     #     50,
            #     #     60,
            #     #     70,
            #     #     80,
            #     #     90,
            #     #     200,
            #     #     300,
            #     #     400,
            #     #     500,
            #     #     600,
            #     #     700,
            #     #     800,
            #     #     900,
            #     #     2000,
            #     #     3000,
            #     #     4000,
            #     #     5000,
            #     #     6000,
            #     #     7000,
            #     #     8000,
            #     #     9000,
            #     # ],
            ticklen=5,
            tickwidth=2.5,
            ticks="outside",
            tickcolor="black",
        ),
    )
    fig.update_layout(
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1240,
        height=860,
        margin=dict(autoexpand=True, l=120, r=120, t=20, b=90),
        title=dict(
            text=titletext,
            font_size=32,
            xanchor="right",
            x=0.83,
            yanchor="top",
            y=0.9,
        ),
        showlegend=False,
    ),
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    # if ylimit[0] is not None:
    #     fig.update_yaxes(range=ylimit[0])
    # if ylimit[1] is not None:
    #     fig.update_yaxes(dtick=ylimit[1])
    # if ylimit[2] is not None:
    #     fig.update_yaxes(minor_dtick=ylimit[2])
    # if titletext is not None:
    #     fig.update_layout(title_text=titletext)
    # if legendtitle is not None:
    #     fig.update_layout(legend_title_text=legendtitle)
    # if legendposition is not None:
    #     fig.update_layout(
    #         legend_x=legendposition[0],
    #         legend_y=legendposition[1],
    #     )
    return fig


def display_fsta_heatmap_linlog(
    df,
    xlimit=[300, 800, 50],
    ylimit=[-10, 100, 20, 8000, 0.5],
    zlimit=[-5, 5, 1],
    title="Sample",
    colorscale="jet",
):
    delta_abs = df.T
    wavelength = delta_abs.columns
    time = delta_abs.index

    fig = go.Figure()
    fig.add_heatmap(
        x=wavelength,
        xaxis="x1",
        y=time,
        yaxis="y1",
        z=delta_abs,
        zmin=zlimit[0],
        zmax=zlimit[1],
        zsmooth=False,
        # showscale=False,
        colorbar=dict(
            title=dict(
                text="\u0394A (mOD)",
                font=dict(color="black", size=32),
                side="right",
                # standoff=10
            ),
            dtick=zlimit[2],
            orientation="v",
            # xanchor="left",
            # x=1.02,
            # yanchor="top",
            # y=1,
            xpad=0,
            ypad=0,
            # xref="container",
            len=1,
            # lenmode="fraction",
            outlinecolor="black",
            outlinewidth=2.5,
            # borderwidth=2.5,
            ticks="outside",
            ticklen=5,
            tickwidth=2.5,
        ),
        colorscale=colorscale,
    )
    fig.add_heatmap(
        x=wavelength,
        xaxis="x2",
        y=time,
        yaxis="y2",
        z=delta_abs,
        zmin=zlimit[0],
        zmax=zlimit[1],
        zsmooth=False,
        showscale=False,
        colorscale=colorscale,
    )

    fig.update_layout(
        template="none",
        font=dict(family="Arial", color="black", size=26),
        xaxis1=dict(
            title=dict(
                text="Wavelength (nm)",
                #    standoff=5,
                font_size=32,
            ),
            domain=[0, 1],
            anchor="y1",
            showline=True,
            mirror=False,
            zeroline=False,
            range=xlimit[:2],
            dtick=xlimit[2],
            linecolor="black",
            linewidth=2.5,
            # tick0=350,
            ticklen=10,
            tickwidth=2.5,
            tickcolor="black",
            # tickformat="%0f",
            ticks="outside",
            minor=dict(
                # dtick=25,
                ticklen=5,
                tickwidth=2.5,
                ticks="outside",
                tickcolor="black",
                showgrid=False,
            ),
        ),
        yaxis1=dict(
            title=dict(
                text="Time (ps)",
                #    standoff=15,
                font_size=32,
            ),
            showline=True,
            linecolor="black",
            mirror=True,
            zeroline=False,
            zerolinecolor="black",
            zerolinewidth=2.5,
            tickcolor="black",
            ticks="outside",
            type="linear",
            domain=[0, ylimit[-1]],
            range=ylimit[:2],
            dtick=ylimit[2],
            tick0=0,
            # tickformat=".2s",
            linewidth=2.5,
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                # dtick=25,
                nticks=5,
                ticklen=5,
                tickwidth=2.5,
                ticks="outside",
                tickcolor="black",
                showgrid=False,
            ),
        ),
        xaxis2=dict(
            anchor="y2",
            # title=dict(text="Wavelength (nm)"),
            showline=True,
            side="top",
            overlaying="x2",
            linecolor="black",
            zeroline=False,
            showticklabels=False,
            ticks="",
            linewidth=2.5,
            # mirror=True,
            # tickcolor="black",
            # ticks="outside",
            showspikes=False,
            range=xlimit[:2],
            domain=[0, 1],
            # dtick=xlimit[2],
            # tick0=350,
            # tickformat="%0f",
            # ticklen=8,
            # minor=dict(
            #     # dtick=25,
            #     ticklen=4,
            #     tickwidth=1,
            #     ticks="outside",
            #     tickcolor="black",
            #     showgrid=False,
            # ),
        ),
        yaxis2=dict(
            showline=True,
            linecolor="black",
            mirror=True,
            zeroline=False,
            tickcolor="black",
            ticks="outside",
            type="log",
            domain=[ylimit[-1], 1],
            range=[np.log10(ylimit[1]), np.log10(ylimit[3])],
            # dtick=1,
            # tick0=0,
            # tickvals=[100, 200, 400, 800, 1600, 3200],
            tickmode="array",
            # tickvals=[100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000],
            # tickvals=[300, 500, 700, 900, 3000, 5000, 7000],
            tickvals=[100, 250, 500, 750, 1000, 2500, 5000, 7500],
            # tickformat="~s",
            ticklen=10,
            linewidth=2.5,
            tickwidth=2.5,
            # minor=dict(
            #     tickmode="array",
            #     tickvals=[
            #         0.1,
            #         0.2,
            #         0.3,
            #         0.4,
            #         0.5,
            #         0.6,
            #         0.7,
            #         0.8,
            #         0.9,
            #         1,
            #         2,
            #         3,
            #         4,
            #         5,
            #         6,
            #         7,
            #         8,
            #         9,
            #         10,
            #         20,
            #         30,
            #         40,
            #         50,
            #         60,
            #         70,
            #         80,
            #         90,
            #         100,
            #         200,
            #         300,
            #         400,
            #         500,
            #         600,
            #         700,
            #         800,
            #         900,
            #         1000,
            #         2000,
            #         3000,
            #         4000,
            #         5000,
            #         6000,
            #         7000,
            #         8000,
            #         9000,
            #         10000,
            #     ],
            #         ticklen=5,
            #         tickwidth=2.5,
            #         ticks="outside",
            #         tickcolor="black",
            #         showgrid=False,
            #     ),
        ),
        width=1240,
        height=860,
        margin=dict(autoexpand=True, l=120, r=120, t=20, b=90),
        # width=1245,
        # height=845,
        # margin=dict(autoexpand=False, pad=0, l=120, r=125, t=10, b=85),
        title=dict(
            text=title,
            font_size=32,
            xanchor="right",
            x=0.83,
            yanchor="top",
            y=0.9,
        ),
    )
    return fig


### contour


def display_fsta_contour_linear(
    df, xlimit=[300, 800, 50], ylimit=[-100, 8000, 1000], zlimit=[-5, 5, 1]
):
    delta_abs = df.T
    wavelength = delta_abs.columns
    time = delta_abs.index

    fig = go.Figure(
        go.Contour(
            x=wavelength,
            y=time,
            z=delta_abs,
            zmin=zlimit[0],
            zmax=zlimit[1],
            # connectgaps=True,
            line_smoothing=1,
            # contours=dict(
            #     # coloring="heatmap",
            #     showlabels=True,
            #     labelfont=dict(size=12, color="white"),
            # ),
            colorbar=dict(
                dtick=zlimit[2],
                title=dict(
                    text="\u0394A (mOD)",
                    font=dict(color="black"),
                    side="right",
                    # standoff=10
                ),
                orientation="v",
                xanchor="left",
                x=1.02,
                yanchor="top",
                y=1,
                xpad=0,
                ypad=0,
                xref="container",
                outlinecolor="black",
                outlinewidth=1,
                ticks="outside",
                ticklen=8,
            ),
            colorscale="RdBu_r",
        )
    )

    fig.update_layout(
        width=1000,
        height=750,
        margin=dict(l=80, r=90, t=10, b=80),
        font=dict(family="Arial", color="black", size=23),
    )

    fig.update_xaxes(
        title=dict(text="Wavelength (nm)", standoff=10),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        tickcolor="black",
        ticks="outside",
        range=xlimit[:2],
        # tick0=350,
        dtick=xlimit[-1],
        tickformat="%0f",
        ticklen=8,
        minor=dict(
            # dtick=25,
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            tickcolor="black",
        ),
    )

    fig.update_yaxes(
        title=dict(text="Time (ps)", standoff=10),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        tickcolor="black",
        ticks="outside",
        range=ylimit[:2],
        # tick0=0,
        dtick=ylimit[-1],
        # tickformat=',.2r',
        tickformat="~s",
        ticklen=8,
        minor=dict(
            # dtick=25,
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            tickcolor="black",
        ),
    )
    return fig


def display_fsta_contour_linlog(
    df, xlimit=[300, 800, 50], ylimit=[-10, 100, 20, 8000, 0.5], zlimit=[-5, 5, 1]
):
    delta_abs = df.T
    wavelength = delta_abs.columns
    time = delta_abs.index

    fig = go.Figure()
    fig.add_contour(
        x=wavelength,
        y=time,
        z=delta_abs,
        # zmin=zlimit[0],
        # zmax=zlimit[1],
        # zsmooth=False,
        line_smoothing=1,
        colorbar=dict(
            dtick=zlimit[2],
            title=dict(
                text="\u0394A (mOD)",
                font=dict(color="black"),
                side="right",
                # standoff=10
            ),
            orientation="v",
            xanchor="left",
            x=1.02,
            yanchor="top",
            y=1,
            xpad=0,
            ypad=0,
            xref="container",
            outlinecolor="black",
            outlinewidth=1,
            ticks="outside",
            ticklen=8,
        ),
        # colorscale="RdBu_r",
        # colorscale=[[0, 'rgb(0,0,255)'],[1, 'rgb(255,0,0)']],
    )
    fig.add_contour(
        x=wavelength,
        xaxis="x",
        y=time,
        yaxis="y2",
        z=delta_abs,
        zmin=zlimit[0],
        zmax=zlimit[1],
        # zsmooth=False,
        line_smoothing=1,
        colorbar=dict(
            dtick=zlimit[2],
            title=dict(
                text="\u0394A (mOD)",
                font=dict(color="black"),
                side="right",
                # standoff=10
            ),
            orientation="v",
            xanchor="left",
            x=1.02,
            yanchor="top",
            y=1,
            xpad=0,
            ypad=0,
            xref="container",
            outlinecolor="black",
            outlinewidth=1,
            ticks="outside",
            ticklen=8,
        ),
        colorscale="RdBu_r",
    )

    fig.update_layout(
        width=1000,
        height=750,
        margin=dict(l=90, r=90, t=10, b=80),
        font=dict(family="Arial", color="black", size=23),
        xaxis=dict(
            title=dict(text="Wavelength (nm)", standoff=0),
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            tickcolor="black",
            ticks="outside",
            range=xlimit[:2],
            dtick=xlimit[2],
            # tick0=350,
            tickformat="%0f",
            ticklen=8,
            minor=dict(
                # dtick=25,
                ticklen=4,
                tickwidth=1,
                ticks="outside",
                tickcolor="black",
            ),
        ),
        yaxis1=dict(
            title=dict(text="Time (ps)", standoff=0),
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            tickcolor="black",
            ticks="outside",
            type="linear",
            domain=[0, ylimit[-1]],
            range=ylimit[:2],
            dtick=ylimit[2],
            tick0=0,
            # tickformat="%0f",
            ticklen=8,
            minor=dict(
                # dtick=25,
                nticks=5,
                ticklen=4,
                tickwidth=1,
                ticks="outside",
                tickcolor="black",
            ),
        ),
        yaxis2=dict(
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            tickcolor="black",
            ticks="outside",
            type="log",
            domain=[ylimit[-1], 1],
            range=[np.log10(ylimit[1]), np.log10(ylimit[3])],
            # dtick=1,
            # tick0=0,
            # tickvals=[100, 200, 400, 800, 1600, 3200],
            tickmode="array",
            tickvals=[100, 500, 1000, 5000],
            # tickformat="~s",
            ticklen=8,
            minor=dict(
                tickmode="array",
                tickvals=[
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100,
                    200,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    1000,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    7000,
                    8000,
                    9000,
                    10000,
                ],
                ticklen=4,
                tickwidth=1,
                ticks="outside",
                tickcolor="black",
            ),
        ),
    )
    return fig


def display_fsta_contourf_log(
    df_fsta,
    xlimit=[[350, 700], 50, 25],
    ylimit=[[0.1, 10000], None, None],
    zlimit=[[-10, 10], 2, 1],
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme="RdBu_r",
    figsize=(4, 3),
):
    w = (df_fsta.index.values,)
    t = (df_fsta.columns.values,)
    od = df_fsta.T.values
    W, T = np.meshgrid(w, t)

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    contour = ax.contourf(
        W,
        T,
        od,
        cmap=colorscheme,
        levels=np.linspace(zlimit[0][0], zlimit[0][1], 25),
    )

    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

    ax.set_ylabel("Time (ps)", fontsize=8, labelpad=2)
    ax.set_yscale("log", base=10)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(tick_formatter_1))

    cbar = fig.colorbar(contour, ax=ax, pad=0.015, format="%0.1f")
    cbar.set_label("\u0394A (mOD)", fontsize=8, labelpad=2)

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    cbar.ax.tick_params(
        axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1
    )

    cbar.outline.set_linewidth(0.5)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_title(titletext, fontsize=6, pad=2)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    return fig


def display_fsta_contourf_symlog(
    df_fsta,
    xlimit=[[350, 700], 50, 25],
    ylimit=[[-5, 100, 8000], 50, 25],
    zlimit=[[-10, 10], 2, 1],
    ylinscale=1,
    titletext=None,
    legendtitle=None,
    # colorscheme="Spectral_r",
    colorscheme="RdBu_r",
    figsize=(4, 3),
):
    w = (df_fsta.index.values,)  # wavelength
    t = (df_fsta.columns.values,)  # delay time
    od = df_fsta.T.values
    W, T = np.meshgrid(w, t)  # meshgrid

    # fig, ax = plt.subplots(dpi=300, layout="constrained")
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    contour = ax.contourf(
        W,
        T,
        od,
        cmap=colorscheme,
        levels=np.linspace(zlimit[0][0], zlimit[0][1], 25),
    )

    ax.set_title(titletext, fontsize=6, pad=2)

    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=2)

    ax.set_yscale("symlog", base=10, linthresh=ylimit[0][1], linscale=ylinscale)
    ax.set_ylim(ylimit[0][0], ylimit[0][2])
    ax.axhline(y=ylimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    # ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(tick_formatter_1))
    yticks = mf.determine_timedelay_label(ylimit)
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks[0]))
    ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(yticks[1]))
    ax.set_ylabel("Time (ps)", fontsize=8, labelpad=2)
    ax.tick_params(axis="y", right=False)
    # ax.yaxis.set_major_formatter("{x:0.1f}")

    cbar = fig.colorbar(contour, ax=ax, pad=0.015, format="%0.1f")
    cbar.set_label("\u0394A (mOD)", fontsize=8, labelpad=2)
    # ax.clabel(contour, fmt='%2.1f', colors='black', fontsize=2)
    # cbar.locator = mpl.ticker.MultipleLocator(zlimit[1])

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    # ax.tick_params(axis="y", right=False)
    cbar.ax.tick_params(
        axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1
    )

    cbar.outline.set_linewidth(0.5)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    return fig


def display_fsta_contourf_symlog_bwn_twl(
    df_fsta,
    xlimit=[None, 4000, 1000],
    xlimit2=[None, 50, 10],
    ylimit=[[-5, 100, 8000], 50, 25],
    zlimit=[[-10, 10], 2, 1],
    ylinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    # norm=True,
    gridon=False,
    # colorscheme=plt.cm.tab10,
    # colorscheme="Spectral_r",
    colorscheme="RdBu_r",
    figsize=(4, 3),
):
    wl = (df_fsta.index.values)  # wavelength
    wn = (10**7 / df_fsta.index.values)  # wavenumber
    t = (df_fsta.columns.values,)  # delay time
    od = df_fsta.T.values
    WN, T = np.meshgrid(wn, t)  # meshgrid

    fig, ax1 = plt.subplots(
        # figsize=(figwidth, mf.calculate_height(figwidth, 4 / 3)),
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    contour = ax1.contourf(
        WN,
        T,
        od,
        cmap=colorscheme,
        levels=np.linspace(zlimit[0][0], zlimit[0][1], 25),
        # colors="k",
        # linewidth=1,
        # linestyles="solid",
    )

    if xlimit[0] is not None:
        ax1.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    ax1.xaxis.set_major_formatter("{x:.0f}")
    ax1.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax1.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)

    secax = ax1.secondary_xaxis(
        "top", functions=(mf.wavenumber_to_wavelength, mf.wavelength_to_wavenumber)
    )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax1.set_yscale("symlog", base=10, linthresh=ylimit[0][1], linscale=ylinscale)
    ax1.set_ylim(ylimit[0][0], ylimit[0][2])
    ax1.axhline(
        y=ylimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5
    )
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    yticks = mf.determine_timedelay_label(ylimit)
    ax1.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks[0]))
    ax1.yaxis.set_minor_locator(mpl.ticker.FixedLocator(yticks[1]))
    ax1.set_ylabel("Time (ps)", fontsize=8, labelpad=2)
    ax1.tick_params(axis="y", right=False)
    ax1.yaxis.set_major_formatter("{x:.0f}")

    cbar = fig.colorbar(contour, ax=ax1, pad=0.015, format="%0.1f")
    cbar.set_label("\u0394A (mOD)", fontsize=8, labelpad=2)
    cbar.outline.set_linewidth(0.5)
    # cbar.locator = mpl.ticker.MultipleLocator(zlimit[1])
    cbar.ax.tick_params(
        axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1
    )

    # ax1.clabel(contour, fmt='%.2f', colors='black', fontsize=2)

    if titletext is not None:
        ax1.set_title(titletext, fontsize=8, pad=2)

    ax1.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    if gridon is True:
        ax1.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax1.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_fsta_contourf_symlog_bcm1_tnm(
    df_fsta,
    xlimit=[None, 4000, 1000],
    xlimit2=[None, 50, 10],
    ylimit=[[-5, 100, 8000], 50, 25],
    zlimit=[[-10, 10], 2, 1],
    ylinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    # norm=True,
    gridon=False,
    # colorscheme=plt.cm.tab10,
    # colorscheme="Spectral_r",
    colorscheme="RdBu_r",
    figsize=(4, 3),
):
    wl = (df_fsta.index.values)  # wavelength
    wn = (10**7 / df_fsta.index.values)  # wavenumber
    t = (df_fsta.columns.values,)  # delay time
    od = df_fsta.T.values
    WN, T = np.meshgrid(wn, t)  # meshgrid

    fig, ax1 = plt.subplots(
        # figsize=(figwidth, mf.calculate_height(figwidth, 4 / 3)),
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    contour = ax1.contourf(
        WN,
        T,
        od,
        cmap=colorscheme,
        levels=np.linspace(zlimit[0][0], zlimit[0][1], 25),
        # colors="k",
        # linewidth=1,
        # linestyles="solid",
    )

    if xlimit[0] is not None:
        ax1.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    ax1.xaxis.set_major_formatter("{x:.0f}")
    ax1.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax1.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)

    secax = ax1.secondary_xaxis(
        "top", functions=(mf.wavenumber_to_wavelength, mf.wavelength_to_wavenumber)
    )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax1.set_yscale("symlog", base=10, linthresh=ylimit[0][1], linscale=ylinscale)
    ax1.set_ylim(ylimit[0][0], ylimit[0][2])
    ax1.axhline(
        y=ylimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5
    )
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    yticks = mf.determine_timedelay_label(ylimit)
    ax1.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks[0]))
    ax1.yaxis.set_minor_locator(mpl.ticker.FixedLocator(yticks[1]))
    ax1.set_ylabel("Time (ps)", fontsize=8, labelpad=2)
    ax1.tick_params(axis="y", right=False)
    ax1.yaxis.set_major_formatter("{x:.0f}")

    cbar = fig.colorbar(contour, ax=ax1, pad=0.015, format="%0.1f")
    cbar.set_label("\u0394A (mOD)", fontsize=8, labelpad=2)
    cbar.outline.set_linewidth(0.5)
    # cbar.locator = mpl.ticker.MultipleLocator(zlimit[1])
    cbar.ax.tick_params(
        axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1
    )

    # ax1.clabel(contour, fmt='%.2f', colors='black', fontsize=2)

    if titletext is not None:
        ax1.set_title(titletext, fontsize=8, pad=2)

    ax1.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    if gridon is True:
        ax1.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax1.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_fsta_contourf_symlog_be3cm1_tnm(
    df_fsta,
    xlimit=[None, 5, 1],
    xlimit2=[None, 50, 10],
    ylimit=[[-5, 100, 8000], 50, 25],
    zlimit=[[-10, 10], 2, 1],
    ylinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    # norm=True,
    gridon=False,
    # colorscheme=plt.cm.tab10,
    # colorscheme="Spectral_r",
    colorscheme="RdBu_r",
    figsize=(4, 3),
):
    nm = df_fsta.index.values  # wavelength
    e3cm1 = 10**4/nm # e3 wavenumber
    t = df_fsta.columns.values  # delay time
    OD = df_fsta.T.values
    E3CM1, T = np.meshgrid(e3cm1, t)  # meshgrid

    fig, ax1 = plt.subplots(
        # figsize=(figwidth, mf.calculate_height(figwidth, 4 / 3)),
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    contour = ax1.contourf(
        E3CM1,
        T,
        OD,
        cmap=colorscheme,
        levels=np.linspace(zlimit[0][0], zlimit[0][1], 25),
        # colors="k",
        # linewidth=1,
        # linestyles="solid",
    )

    if xlimit[0] is not None:
        ax1.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax1.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)
    ax1.xaxis.set_major_formatter("{x:.0f}")
    ax1.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax1.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)

    secax = ax1.secondary_xaxis(
        "top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1)
    )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax1.set_yscale("symlog", base=10, linthresh=ylimit[0][1], linscale=ylinscale)
    ax1.set_ylim(ylimit[0][0], ylimit[0][2])
    ax1.axhline(
        y=ylimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5
    )
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    yticks = mf.determine_timedelay_label(ylimit)
    ax1.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks[0]))
    ax1.yaxis.set_minor_locator(mpl.ticker.FixedLocator(yticks[1]))
    ax1.set_ylabel("Time (ps)", fontsize=8, labelpad=2)
    ax1.tick_params(axis="y", right=False)
    ax1.yaxis.set_major_formatter("{x:.0f}")

    cbar = fig.colorbar(contour, ax=ax1, pad=0.015, format="%0.1f")
    cbar.set_label("\u0394A (mOD)", fontsize=8, labelpad=2)
    cbar.outline.set_linewidth(0.5)
    # cbar.locator = mpl.ticker.MultipleLocator(zlimit[1])
    cbar.ax.tick_params(
        axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1
    )

    # ax1.clabel(contour, fmt='%.2f', colors='black', fontsize=2)

    if titletext is not None:
        ax1.set_title(titletext, fontsize=8, pad=2)

    ax1.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    if gridon is True:
        ax1.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax1.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_fsta_contourf_outline_symlog_bwn_twl(
    df_fsta,
    xlimit=[None, 4000, 1000],
    xlimit2=[None, 50, 10],
    ylimit=[[-5, 100, 8000], 50, 25],
    zlimit=[[-10, 10], 2, 1],
    ylinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    # norm=True,
    gridon=False,
    # colorscheme=plt.cm.tab10,
    # colorscheme="Spectral_r",
    colorscheme="RdBu_r",
    figsize=(4, 3),
):
    # wl = (df_fsta.index.values,)  # wavelength
    wn = (10**7 / df_fsta.index.values,)  # wavenumber
    t = (df_fsta.columns.values,)  # delay time
    od = df_fsta.T.values
    WN, T = np.meshgrid(wn, t)  # meshgrid

    fig, ax1 = plt.subplots(
        # figsize=(figwidth, mf.calculate_height(figwidth, 4 / 3)),
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    ax1.contour(
        WN,
        T,
        od,
        levels=np.linspace(zlimit[0][0], zlimit[0][1], 25),
        colors="k",
        linewidths=0.1,
        linestyles="solid",
        alpha=1,
    )
    contourf = ax1.contourf(
        WN, T, od, levels=np.linspace(zlimit[0][0], zlimit[0][1], 25), cmap=colorscheme
    )

    if xlimit[0] is not None:
        ax1.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    ax1.xaxis.set_major_formatter("{x:.0f}")
    ax1.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax1.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)

    secax = ax1.secondary_xaxis(
        "top", functions=(mf.wavenumber_to_wavelength, mf.wavelength_to_wavenumber)
    )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax1.set_yscale("symlog", base=10, linthresh=ylimit[0][1], linscale=ylinscale)
    ax1.set_ylim(ylimit[0][0], ylimit[0][2])
    ax1.axhline(
        y=ylimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5
    )
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    yticks = mf.determine_timedelay_label(ylimit)
    ax1.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks[0]))
    ax1.yaxis.set_minor_locator(mpl.ticker.FixedLocator(yticks[1]))
    ax1.set_ylabel("Time (ps)", fontsize=8, labelpad=2)
    ax1.tick_params(axis="y", right=False)
    ax1.yaxis.set_major_formatter("{x:.0f}")

    cbar = fig.colorbar(contourf, ax=ax1, pad=0.015)
    cbar.set_label("\u0394A (mOD)", fontsize=8, labelpad=2)
    cbar.outline.set_linewidth(0.5)
    # ax.clabel(contour, fmt='%2.1f', colors='black', fontsize=2)
    # cbar.locator = mpl.ticker.MultipleLocator(zlimit[1])
    cbar.ax.tick_params(
        axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1
    )

    if titletext is not None:
        ax1.set_title(titletext, fontsize=8, pad=2)

    ax1.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    if gridon is True:
        ax1.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax1.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_fsta_contourf_symlog_bwl_twn(
    df_fsta,
    xlimit=[None, 50, 10],
    xlimit2=[None, 5000, 1000],
    ylimit=[[-5, 100, 8000], 50, 25],
    zlimit=[[-10, 10], 2, 1],
    ylinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    # norm=True,
    gridon=False,
    # colorscheme=plt.cm.tab10,
    # colorscheme="Spectral_r",
    colorscheme="RdBu_r",
    figsize=(4, 3),
):
    wl = (df_fsta.index.values,)  # wavelength
    # wn = (10**7 / df_fsta.index.values,)  # wavenumber
    t = (df_fsta.columns.values,)  # delay time
    od = df_fsta.T.values
    WN, T = np.meshgrid(wl, t)  # meshgrid

    fig, ax1 = plt.subplots(
        # figsize=(figwidth, mf.calculate_height(figwidth, 4 / 3)),
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    contour = ax1.contourf(
        WN,
        T,
        od,
        cmap=colorscheme,
        levels=np.linspace(zlimit[0][0], zlimit[0][1], 25),
    )

    if xlimit[0] is not None:
        ax1.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax1.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    ax1.xaxis.set_major_formatter("{x:.0f}")
    ax1.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax1.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)

    secax = ax1.secondary_xaxis(
        "top", functions=(mf.wavelength_to_wavenumber, mf.wavenumber_to_wavelength)
    )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax1.set_yscale("symlog", base=10, linthresh=ylimit[0][1], linscale=ylinscale)
    ax1.set_ylim(ylimit[0][0], ylimit[0][2])
    ax1.axhline(
        y=ylimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5
    )
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax1.yaxis.set_major_locator(
        mpl.ticker.FixedLocator(
            mf.generate_ticks_linearfield([ylimit[0][0], ylimit[0][1]], ylimit[1])
            + [tick for tick in [100, 500, 1000, 5000, 10000] if tick > ylimit[0][1]]
        )
    )
    ax1.yaxis.set_minor_locator(
        mpl.ticker.FixedLocator(
            mf.generate_ticks_linearfield([ylimit[0][0], ylimit[0][1]], ylimit[2])
            + [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
                8000,
                9000,
                10000,
            ]
        )
    )
    ax1.set_ylabel("Time (ps)", fontsize=8, labelpad=2)
    ax1.tick_params(axis="y", right=False)
    ax1.yaxis.set_major_formatter("{x:.0f}")

    cbar = fig.colorbar(contour, ax=ax1, pad=0.015, format="%0.1f")
    cbar.set_label("\u0394A (mOD)", fontsize=8, labelpad=2)
    cbar.outline.set_linewidth(0.5)
    # ax.clabel(contour, fmt='%2.1f', colors='black', fontsize=2)
    # cbar.locator = mpl.ticker.MultipleLocator(zlimit[1])
    cbar.ax.tick_params(
        axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1
    )

    if titletext is not None:
        ax1.set_title(titletext, fontsize=8, pad=2)

    ax1.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    if gridon is True:
        ax1.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax1.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


### spectra


def display_fsta_spectra(
    data,
    xlimit=[300, 800, 50],
    zlimit=[-5, 5, 1],
    titletext=None,
    legendtext=None,
    ab=None,
    ab_scale=1,
    pl=None,
    pl_scale=1,
):
    fig = go.Figure()
    if ab is not None:
        fig.add_scatter(
            x=ab.iloc[:, 0],
            y=-ab_scale * ab.iloc[:, 1],
            name="Abs.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(110,190,235,0.2)"
            # line_color="rgba(0,0,0,0)",
            # fillcolor="#6ebeeb"
        )
    if pl is not None:
        fig.add_scatter(
            x=pl.iloc[:, 0],
            y=-pl_scale * pl.iloc[:, 1],
            name="PL.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(240,158,162,0.2)",
            # fillcolor="#f09ea2"
        )
    wavelength = data.index
    for col, color in zip(data.columns, rainbow7):
        if col < 1:
            name = round(col, 1)
            name = f"{name} ps"
        elif col > 1 and col < 1000:
            name = round(col)
            name = f"{name} ps"
        else:
            name = col / 1000  # ps to ns
            name = f"{name:.1f} ns"
        fig.add_scatter(
            x=wavelength,
            y=data[col],
            name=name,
            mode="lines",
            line_width=2.5,
            line_color=color,
        )
    fig.update_layout(
        template="none",
        font=dict(family="Arial", color="black", size=26),
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            color="black",
            range=[xlimit[0], xlimit[1]],
            tick0=300,
            dtick=xlimit[2],
            showline=True,
            # mirror=True,
            zeroline=False,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                dtick=25,
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="\u0394A (mOD)",
                standoff=0,
                font_size=32,
            ),
            color="black",
            range=[zlimit[0], zlimit[1]],
            # tick0=300,
            dtick=zlimit[2],
            showline=True,
            # mirror=True,
            zeroline=True,
            zerolinecolor="rgb(204,204,204)",
            zerolinewidth=1,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                dtick=zlimit[2] / 2,
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        width=1110,
        height=860,
        margin=dict(autoexpand=False, l=90, r=20, t=20, b=90),
        title=dict(
            text=titletext,
            font_size=28,
            x=0.88,
            xanchor="center",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_text=legendtext,
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    return fig



def display_fsta_spectra_v2(
    df_scol,
    xlimit=[None, 50, 25],
    ylimit=[None, 2, 1],
    legendtitle=None,
    legendposition=None,
    titletext=None,
    ab=None,
    ab_scale=1,
    pl=None,
    pl_scale=1,
    colorscheme=rainbow7,
):
    fig = go.Figure()
    wavelength = df_scol.index
    for col, color in zip(df_scol.columns, colorscheme):
        if col < 1:
            name = round(col, 1)
            name = f"{name} ps"
        elif col > 1 and col < 1000:
            name = round(col)
            name = f"{name} ps"
        else:
            name = col / 1000  # ps to ns
            name = f"{name:.1f} ns"
        fig.add_scatter(
            x=wavelength,
            y=df_scol[col],
            name=name,
            mode="lines",
            line_width=2.5,
            line_color=color,
        )
    fig.update_layout(
        template="none",
        font=dict(family="Arial", color="black", size=26),
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
                standoff=0,
                font_size=32,
            ),
            color="black",
            # tick0=300,
            showline=True,
            # mirror=True,
            zeroline=False,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="\u0394A (mOD)",
                standoff=0,
                font_size=32,
            ),
            color="black",
            # tick0=300,
            showline=True,
            # mirror=True,
            zeroline=True,
            zerolinecolor="rgb(204,204,204)",
            zerolinewidth=1,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        width=1130,
        height=880,
        margin=dict(autoexpand=False, l=90, r=40, t=40, b=90),
        title=dict(
            font_size=32,
            x=0.5,
            xanchor="center",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if ab is not None:
        fig.add_scatter(
            x=ab.iloc[:, 0],
            y=-ab_scale * ab.iloc[:, 1],
            name="Abs.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(110,190,235,0.2)",
        )
    if pl is not None:
        fig.add_scatter(
            x=pl.iloc[:, 0],
            y=-pl_scale * pl.iloc[:, 1],
            name="PL.",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(240,158,162,0.2)",
        )
    return fig



# spectra1 = [
#     0.3,
#     0.5,
#     1,
#     3,
#     5,
#     10,
#     48,
# ]
# spectra1 = mins.extract_spectra_trspectra(data, spectra1)
# spectra2 = [100, 1000, 3000, 5000, 8000]
# spectra2 = mins.extract_spectra_trspectra(data, spectra2)

# fig = make_subplots(
#     rows=2,
#     cols=1,
#     row_heights=[0.5, 0.5],
#     # shared_xaxes=True,
#     horizontal_spacing=0,
#     vertical_spacing=0,
#     specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
#     # column_titles=["nin"],
#     # row_titles=["fa", "faf"],
#     # y_title=dict(text="delta A (mOD)"),
#     # x_title="Energy (keV)",
#     print_grid=False,
# )
# # fig=go.Figure()
# for col in spectra1.columns:
#     fig.add_trace(
#         go.Scatter(
#             x=spectra1.index,
#             xaxis="x1",
#             y=spectra1[col],
#             yaxis="y1",
#             mode="lines",
#             name=col,
#             # showlegend=True,
#             # legend="legend",
#         ),
#         row=1,
#         col=1,
#     )
# for col in spectra2.columns:
#     fig.add_trace(
#         go.Scatter(
#             x=spectra1.index,
#             xaxis="x2",
#             y=spectra2[col],
#             yaxis="y2",
#             mode="lines",
#             name=col,
#             # showlegend=True,
#             legend="legend2",
#         ),
#         row=2,
#         col=1,
#     )
# fig.update_layout(
#     template="none",
#     height=750,
#     width=1000,
#     # xaxis=dict(
#     #     showline=True, linewidth=2.5, mirror=True, range=[345, 705], showgrid=False
#     # ),
#         xaxis=dict(
#             # title=dict(
#             #     text="Wavelength (nm)",
#             #     standoff=0,
#             #     font_size=32,
#             # ),
#             color="black",
#             showgrid=True,
#             # range=[xlimit[0], xlimit[1]],
#             tick0=300,
#             # dtick=xlimit[2],
#             showline=True,
#             # side="top",
#             mirror=True,
#             zeroline=False,
#             linewidth=2.5,
#             showticklabels=False,
#             ticks="",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2.5,
#             # minor=dict(
#             #     dtick=25,
#             #     ticks="outside",
#             #     ticklen=5,
#             #     tickwidth=2.5,
#             #     showgrid=False,
#             # ),
#         ),
#         xaxis2=dict(
#             title=dict(
#                 text="Wavelength (nm)",
#                 standoff=0,
#                 font_size=32,
#             ),
#             color="black",
#             # range=[xlimit[0], xlimit[1]],
#             tick0=300,
#             # dtick=xlimit[2],
#             showline=True,
#             # mirror=True,
#             zeroline=False,
#             showgrid=True,
#             linewidth=2.5,
#             showticklabels=True,
#             ticks="outside",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2.5,
#             minor=dict(
#                 dtick=25,
#                 ticks="outside",
#                 ticklen=5,
#                 tickwidth=2.5,
#                 showgrid=False,
#             ),
#         ),
#         yaxis=dict(
#             title=dict(
#                 text="\u0394A (mOD)",
#                 standoff=0,
#                 font_size=32,
#             ),
#             color="black",
#             range=[-3,7],
#             # tick0=300,
#             # domain=[0.5,1],
#             # dtick=zlimit[2],
#             showline=True,
#             mirror=True,
#             zeroline=True,
#             zerolinecolor="rgb(204,204,204)",
#             zerolinewidth=1,
#             showgrid=False,
#             linewidth=2.5,
#             ticks="outside",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2.5,
#             # minor=dict(
#             #     # dtick=0.5,
#             #     ticks="outside",
#             #     ticklen=5,
#             #     tickwidth=2.5,
#             #     showgrid=False,
#             # ),
#         ),
#         yaxis2=dict(
#             title=dict(
#                 text="\u0394A (mOD)",
#                 standoff=0,
#                 font_size=32,
#             ),
#             color="black",
#             range=[-3,7],
#             # range=[zlimit[0], zlimit[1]],
#             # tick0=300,
#                         # domain=[0,0.5],
#             # dtick=zlimit[2],
#             showline=True,
#             mirror=True,
#             zeroline=True,
#             zerolinecolor="rgb(204,204,204)",
#             zerolinewidth=1,
#             showgrid=False,
#             linewidth=2.5,
#             ticks="outside",
#             tickcolor="black",
#             ticklen=10,
#             tickwidth=2.5,
#             # minor=dict(
#             #     # dtick=0.5,
#             #     ticks="outside",
#             #     ticklen=5,
#             #     tickwidth=2.5,
#             #     showgrid=False,
#             # ),
#         ),
#     # xaxis2=dict(showline=True, linewidth=2.5, mirror=True, range=[345, 705]),
#     # showlegend=False,
#     # legend=dict(
#     #     title_text="abc",
#     #     xref="container",
#     #     yref="container",
#     #     bgcolor="black",
#         # title_font_size=10,
#         # font_size=28,
#         # xanchor="left",
#         # x=0,
#         # yanchor="top",
#         # y=0.95,
#         # yref="container",
#     # ),
#     legend2=dict(
#         title_text="xxx",
#         xref="container",
#         yref="container",
#         bgcolor="orange"
#         # title_font_size=32,
#         # font_size=28,
#         # xanchor="left",
#         # x=0.5,
#         # yanchor="top",
#         # y=0.95,
#         # yref="container",
#         # visible=True,
#     )
#     # yaxis2=dict(range=[400, 700])
# )
# # fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
# fig.show()



def display_fsta_spectra_bnm(
    df_fstas_wl,  # df_1col
    xlimit=[None, 50, 10],
    ylimit=[None, 2, 1],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    wl = df_fstas_wl.index.values
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            wl,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig



def display_fsta_spectra_bwl_twn(
    df_fstas_wl,  # df_1col
    xlimit=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    xlimit2=[None, 5000, 1000],
    ylimit=[None, 2, 1],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    wl = df_fstas_wl.index.values
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            wl,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis(
        "top", functions=(mf.wavelength_to_wavenumber, mf.wavenumber_to_wavelength)
    )
    secax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.wavelength_to_wavenumber(xlimit2[0][0]),
            mf.wavelength_to_wavenumber(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig



def display_fsta_spectra_bnm_tcm1(
    df_fstas_wl,  # df_1col
    xlimit=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    xlimit2=[None, 5000, 1000],
    ylimit=[None, 2, 1],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    wl = df_fstas_wl.index.values
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            wl,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis(
        "top", functions=(mf.wavelength_to_wavenumber, mf.wavenumber_to_wavelength)
    )
    secax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.wavelength_to_wavenumber(xlimit2[0][0]),
            mf.wavelength_to_wavenumber(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig



def display_fsta_spectra_bwn_twl(
    df_fstas_wl,  # df_1col
    xlimit=[None, 4000, 1000],
    xlimit2=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    ylimit=[None, 2, 1],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    wl = df_fstas_wl.index.values
    wn = 10**7 / wl
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            # label = f"{round(df_fstas_wl.columns[i],1)} ps"
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            wn,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis(
        "top", functions=(mf.wavenumber_to_wavelength, mf.wavelength_to_wavenumber)
    )
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.wavelength_to_wavenumber(xlimit2[0][0]),
            mf.wavelength_to_wavenumber(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig



def display_fsta_spectra_bcm1_tnm(
    df_fstas_wl,  # df_1col
    xlimit=[None, 4000, 1000],
    xlimit2=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    ylimit=[None, 2, 1],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    wl = df_fstas_wl.index.values
    wn = 10**7 / wl
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            # label = f"{round(df_fstas_wl.columns[i],1)} ps"
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            wn,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis(
        "top", functions=(mf.wavenumber_to_wavelength, mf.wavelength_to_wavenumber)
    )
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.wavelength_to_wavenumber(xlimit2[0][0]),
            mf.wavelength_to_wavenumber(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig



def display_fsta_spectra_be3cm1_tnm(
    df_fstas_wl,  # df_1col
    xlimit=[None, 5, 1],
    xlimit2=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    ylimit=[None, 2, 1],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    nm = df_fstas_wl.index.values
    # cm1 = 10**7 / nm
    e3cm1 = 10**4 / nm
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            # label = f"{round(df_fstas_wl.columns[i],1)} ps"
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            e3cm1,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis(
        "top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1)
    )
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.nm_to_e3cm1(xlimit2[0][0]),
            mf.nm_to_e3cm1(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig



### kineticss


def display_fsta_kinetics_linear(
    data,
    xlimit=[-100, 8000, 1000],
    zlimit=[-5, 5, 1],
    legendtext=None,
    titletext=None,
    mode="lines",
):
    fig = go.Figure()
    for col in data.columns:
        wavelegnth = round(col)
        fig.add_scatter(x=data.index, y=data[col], name=f"{wavelegnth} nm", mode=mode)
        fig.update_xaxes(
            title=dict(
                text="Time (ps)",
                standoff=10,
                font_size=32,
            ),
            color="black",
            range=xlimit[:2],
            tick0=0,
            dtick=xlimit[2],
            showline=True,
            # mirror=True,
            zeroline=False,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                dtick=xlimit[2] / 2,
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        )
        fig.update_yaxes(
            title=dict(
                text="\u0394A (mOD)",
                standoff=10,
                font_size=32,
            ),
            color="black",
            # tick0=0,
            range=zlimit[:2],
            dtick=zlimit[2],
            showline=True,
            # mirror=True,
            zeroline=True,
            zerolinecolor="rgb(204,204,204)",
            zerolinewidth=1,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                dtick=zlimit[2] / 2,
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        )
    fig.update_layout(
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1100,
        height=850,
        margin=dict(l=90, r=10, t=10, b=90),
        title=dict(
            text=titletext,
            font_size=28,
            x=0.88,
            xanchor="center",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_text=legendtext,
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.965,
            yanchor="top",
            y=0.95,
        ),
        colorway=rainbow7,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    return fig


def display_fsta_rawkinetics_linear(
    df_1col,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    colorway=plcolor.qualitative.Set1,
    mode="lines",
):
    fig = go.Figure()
    for col in df_1col.columns:
        wavelegnth = round(col)
        fig.add_scatter(
            x=df_1col.index, y=df_1col[col], name=f"{wavelegnth} nm", mode=mode
        )
        fig.update_xaxes(
            title=dict(
                text="Time (ps)",
                standoff=10,
                font_size=32,
            ),
            color="black",
            tick0=0,
            showline=True,
            # mirror=True,
            zeroline=False,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        )
        fig.update_yaxes(
            title=dict(
                text="\u0394A (mOD)",
                standoff=10,
                font_size=32,
            ),
            color="black",
            # tick0=0,
            showline=True,
            # mirror=True,
            zeroline=True,
            zerolinecolor="rgb(204,204,204)",
            zerolinewidth=1,
            showgrid=False,
            linewidth=2.5,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            minor=dict(
                ticks="outside",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        )
    fig.update_layout(
        template="none",
        font=dict(family="Arial", color="black", size=26),
        width=1100,
        height=850,
        margin=dict(l=90, r=10, t=10, b=90),
        title=dict(
            text=titletext,
            font_size=28,
            x=0.88,
            xanchor="center",
            y=0.9,
            yanchor="top",
        ),
        showlegend=True,
        legend=dict(
            title_text=legendtext,
            title_font_size=32,
            font_size=28,
            xanchor="right",
            x=0.965,
            yanchor="top",
            y=0.95,
        ),
        colorway=rainbow7,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    if xlimit[0] is not None:
        fig.update_xaxes(range=xlimit[0])
    if xlimit[1] is not None:
        fig.update_xaxes(dtick=xlimit[1])
    if xlimit[2] is not None:
        fig.update_xaxes(minor_dtick=xlimit[2])
    if ylimit[0] is not None:
        fig.update_yaxes(range=ylimit[0])
    if ylimit[1] is not None:
        fig.update_yaxes(dtick=ylimit[1])
    if ylimit[2] is not None:
        fig.update_yaxes(minor_dtick=ylimit[2])
    if legendtitle is not None:
        fig.update_layout(legend_title_text=legendtitle)
    if legendposition is not None:
        fig.update_layout(
            legend_x=legendposition[0],
            legend_y=legendposition[1],
        )
    if titletext is not None:
        fig.update_layout(title_text=titletext)
    if colorway is not None:
        fig.update_layout(colorway=colorway)
    return fig


def display_fsta_kinetics(
    df_1col,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    xlintresh=None,
    xlinscale=None,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition=None,
    colorscheme=plt.cm.tab20
    # mode="lines",
):
    fig, ax = plt.subplots(
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
    )

    for i, col in enumerate(df_1col.columns):
        wavelength = round(col)
        ax.plot(
            df_1col.index, df_1col[col], label=f"{wavelength} nm", color=colorscheme(i)
        )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc="upper right",
    )

    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][1])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    if xlintresh is not None:
        ax.set_xscale(
            "symlog",
            base=10,
            linthresh=xlintresh,
            subs=[2, 3, 4, 5, 6, 7, 8, 9],
            linscale=xlinscale,
        )
        # ax.xaxis.set_major_locator([1, 5, 10, 1000])
    # ax.xaxis.grid(True)
    ax.set_xticks([1, 5, 10], minor=True)
    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    # if legendtitle is not None:
    #     ax.legend(title=legendtitle)
    # ax.legend(fontsize=30)
    # if titletext is not None:
    #     ax.set_title(titletext, fontsize=30)

    # plt.show()
    return fig


def display_fsta_kinetics_1col_linear(
    df_1col,
    xlimit=[None, None, None],
    ylimit=[None, None, None],
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab20,
):
    fig, ax = plt.subplots(
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
    )

    for i, col in enumerate(df_1col.columns):
        wavelength = round(col)
        ax.plot(
            df_1col.index, df_1col[col], label=f"{wavelength} nm", color=colorscheme(i)
        )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        title=legendtitle,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    # ax.xaxis.grid(True)

    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][1])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    if titletext is not None:
        ax.set_title(titletext, fontsize=10)

    # plt.show()
    return fig



def display_fsta_kinetics_1col_symlog(
    df_1col,
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    showwn=False,
):
    fig, ax = plt.subplots(
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
    )

    for i, col in enumerate(df_1col.columns):
        if legendtext is None:
            if showwn is False:
                label = f"{round(float(col))} nm"
            else:
                label = (
                    f"{round(mf.wavelength_to_wavenumber(float(col)))} cm\u207b\u00b9"
                )
        else:
            label = legendtext[i]
        if mode == "line":
            ax.plot(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                linewidth=1,
            )
        elif mode == "scatter":
            ax.scatter(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                # facecolors=None,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
            )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )
    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    return fig



def display_fsta_kinetics_2col_symlog(
    df_2col,
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i in range(int(len(df_2col.columns) / 2)):
        wavelength = round(df_2col.columns[2 * i + 1])
        if mode == "line":
            ax.plot(
                df_2col.iloc[:, 2 * i],
                df_2col.iloc[:, 2 * i + 1],
                label=f"{wavelength} nm",
                color=colorscheme(i),
                linewidth=0.5,
            )
        elif mode == "scatter":
            ax.scatter(
                df_2col.iloc[:, 2 * i],
                df_2col.iloc[:, 2 * i + 1],
                label=f"{wavelength} nm",
                color=colorscheme(i),
                # facecolors=None,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
            )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    return fig



def display_fsta_1colkinetics_symlog(
    df_1col,
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    showwn=False,
):
    fig, ax = plt.subplots(
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
    )

    for i, col in enumerate(df_1col.columns):
        if legendtext is None:
            if showwn is False:
                label = f"{round(float(col))} nm"
            else:
                label = (
                    f"{round(mf.wavelength_to_wavenumber(float(col)))} cm\u207b\u00b9"
                )
        else:
            label = legendtext[i]
        if mode == "line":
            ax.plot(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                linewidth=1,
            )
        elif mode == "scatter":
            ax.scatter(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                # facecolors=None,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
            )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )
    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    return fig


# def display_fsta_1colkinetics_symlog(
#     df_1col,
#     xlimit=[[-5, 100, 8000],50,25],
#     ylimit=[None, None, None],
#     xlinscale=1,
#     titletext=None,
#     legendtitle=None,
#     legendposition="upper right",
#     colorscheme=plt.cm.tab10,
#     mode="scatter",
# ):
#     fig, ax = plt.subplots(
#         figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
#     )

#     for i in range(int(len(df_1col.columns)/2)):
#         wavelength = round(df_1col.columns[2 * i + 1])
#         if mode == "line":
#             ax.plot(
#                 df_1col.iloc[:, 2*i],
#                 df_1col.iloc[:, 2*i+1],
#                 label=f"{wavelength} nm",
#                 color=colorscheme(i),
#                 linewidth=1,
#             )
#         elif mode == "scatter":
#             ax.scatter(
#                 df_1col.iloc[:, 2*i],
#                 df_1col.iloc[:, 2*i+1],
#                 label=f"{wavelength} nm",
#                 color=colorscheme(i),
#                 # facecolors=None,
#                 s=2,
#                 alpha=0.5,
#                 linewidth=0.1,
#                 edgecolor="black",
#             )

#     ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
#     ax.xaxis.set_major_formatter("{x:.0f}")
#     ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
#     ax.yaxis.set_major_formatter("{x:.0f}")
#     ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
#     ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

#     for axis in ["top", "bottom", "left", "right"]:
#         ax.spines[axis].set_linewidth(0.5)

#     ax.legend(
#         title=legendtitle,
#         fontsize=6,
#         frameon=False,
#         facecolor="none",
#         edgecolor="none",
#         handlelength=1.5,
#         loc=legendposition,
#     )

#     ax.set_xscale(
#         "symlog",
#         base=10,
#         linthresh=xlimit[0][1],
#         # subs=[2, 3, 4, 5, 6, 7, 8, 9],
#         linscale=xlinscale,
#     )

#     # ax.axvline(
#     #     x=xlimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.1
#     # )
#     ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
#     # ax.set_xlim(left=-50, right=8000)
#     ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#     ax.xaxis.set_major_locator(
#         mpl.ticker.FixedLocator(
#             [-2*xlimit[1][0], -xlimit[1][0]] +
#             np.arange(0, xlimit[0][1] + xlimit[1][0], xlimit[1][0]).tolist() +
#             xlimit[1][1]
#         )
#     )
#     ax.xaxis.set_minor_locator(
#         mpl.ticker.FixedLocator(
#             np.arange(xlimit[0][0], xlimit[0][1] + xlimit[2][0], xlimit[2][0]).tolist()
#             + xlimit[2][1]
#         )
#     )
#     if ylimit[0] is not None:
#         ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
#     if ylimit[1] is not None:
#         ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
#     if ylimit[2] is not None:
#         ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
#     if titletext is not None:
#         ax.set_title(titletext, fontsize=10)

#     return fig


# def display_fsta_fitted2colkinetics_symlog(
#     df_2col,
#     # xlimit=[
#     #     [-25, 100, 8000],
#     #     [50, [500, 1000, 5000, 10000]],
#     #     [
#     #         25,
#     #         [
#     #             100,
#     #             200,
#     #             300,
#     #             400,
#     #             500,
#     #             600,
#     #             700,
#     #             800,
#     #             900,
#     #             1000,
#     #             2000,
#     #             3000,
#     #             4000,
#     #             5000,
#     #             6000,
#     #             7000,
#     #             8000,
#     #             9000,
#     #             10000,
#     #         ],
#     #     ],
#     # ],
#     # ylimit=[None, None, None],
#     xlimit=[[-5, 100, 8000],50,25],
#     ylimit=[None, None, None],
#     xlinscale=1,
#     titletext=None,
#     legendtitle=None,
#     legendposition="upper right",
#     colorscheme=plt.cm.tab10,
#     mode="scatter",
# ):
#     fig, ax = plt.subplots(
#         figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
#     )

#     for i in range(int(len(df_2col.columns)/2)):
#         wavelength = round(df_2col.columns[2 * i + 1])
#         if mode == "line":
#             ax.plot(
#                 df_2col.iloc[:, 2*i],
#                 df_2col.iloc[:, 2*i+1],
#                 label=f"{wavelength} nm",
#                 color=colorscheme(i),
#                 linewidth=1,
#             )
#         elif mode == "scatter":
#             ax.scatter(
#                 df_2col.iloc[:, 2*i],
#                 df_2col.iloc[:, 2*i+1],
#                 label=f"{wavelength} nm",
#                 color=colorscheme(i),
#                 # facecolors=None,
#                 s=2,
#                 alpha=0.5,
#                 linewidth=0.1,
#                 edgecolor="black",
#             )

#     ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
#     ax.xaxis.set_major_formatter("{x:.0f}")
#     ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
#     ax.yaxis.set_major_formatter("{x:.0f}")
#     ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
#     ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

#     for axis in ["top", "bottom", "left", "right"]:
#         ax.spines[axis].set_linewidth(0.5)

#     ax.legend(
#         title=legendtitle,
#         fontsize=6,
#         frameon=False,
#         facecolor="none",
#         edgecolor="none",
#         handlelength=1.5,
#         loc=legendposition,
#     )

#     ax.set_xscale(
#         "symlog",
#         base=10,
#         linthresh=xlimit[0][1],
#         # subs=[2, 3, 4, 5, 6, 7, 8, 9],
#         linscale=xlinscale,
#     )

#     # ax.axvline(
#     #     x=xlimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.1
#     # )
#     ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
#     # ax.set_xlim(left=-50, right=8000)
#     ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#     ax.xaxis.set_major_locator(
#         mpl.ticker.FixedLocator(
#             [-2*xlimit[1][0], -xlimit[1][0]] +
#             np.arange(0, xlimit[0][1] + xlimit[1][0], xlimit[1][0]).tolist() +
#             xlimit[1][1]
#         )
#     )
#     ax.xaxis.set_minor_locator(
#         mpl.ticker.FixedLocator(
#             np.arange(xlimit[0][0], xlimit[0][1] + xlimit[2][0], xlimit[2][0]).tolist()
#             + xlimit[2][1]
#         )
#     )
#     if ylimit[0] is not None:
#         ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
#     if ylimit[1] is not None:
#         ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
#     if ylimit[2] is not None:
#         ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
#     if titletext is not None:
#         ax.set_title(titletext, fontsize=10)

#     return fig




def display_fsta_2colkinetics_symlog(
    df_2col,
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i in range(int(len(df_2col.columns) / 2)):
        wavelength = round(df_2col.columns[2 * i + 1])
        if mode == "line":
            ax.plot(
                df_2col.iloc[:, 2 * i],
                df_2col.iloc[:, 2 * i + 1],
                label=f"{wavelength} nm",
                color=colorscheme(i),
                linewidth=0.5,
            )
        elif mode == "scatter":
            ax.scatter(
                df_2col.iloc[:, 2 * i],
                df_2col.iloc[:, 2 * i + 1],
                label=f"{wavelength} nm",
                color=colorscheme(i),
                # facecolors=None,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
            )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    return fig



def display_fsta_globalfit_spectra_bnm(
    df_fstas_wl,  # df_1col
    xlimit=[None, 50, 10],
    ylimit=[None, 2, 1],
    ylabel="DAS",
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    nm = df_fstas_wl.index.values
    # cm1 = 10**7 / nm
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            # label = f"{round(df_fstas_wl.columns[i],1)} ps"
            # label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
            label = f"species {i+1}"
        else:
            label = legendtext[i]
        ax.plot(
            nm,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig

def display_fsta_globalfit_spectra_bwn_twl(
    df_fstas_wl,  # df_1col
    xlimit=[None, 4000, 1000],
    xlimit2=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    ylimit=[None, 2, 1],
    ylabel="DAS",
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    wl = df_fstas_wl.index.values
    wn = 10**7 / wl
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            # label = f"{round(df_fstas_wl.columns[i],1)} ps"
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            wn,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis(
        "top", functions=(mf.wavenumber_to_wavelength, mf.wavelength_to_wavenumber)
    )
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.wavelength_to_wavenumber(xlimit2[0][0]),
            mf.wavelength_to_wavenumber(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    # ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_fsta_globalfit_spectra_be3cm1_tnm(
    df_fstas_wl,  # df_1col
    xlimit=[None, 5, 1],
    xlimit2=[None, 50, 10],  # only xlimit[0] or xlimit2[0] should be given
    ylimit=[None, 2, 1],
    ylabel="DAS", # "Evolution Associated Difference Spectra",
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=True,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    nm = df_fstas_wl.index.values
    e3cm1 = 10**4 / nm
    for i in range(0, len(df_fstas_wl.columns)):
        if legendtext is None:
            # label = f"{round(df_fstas_wl.columns[i],1)} ps"
            label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
        else:
            label = legendtext[i]
        ax.plot(
            e3cm1,
            df_fstas_wl.iloc[:, i],
            label=label,
            linewidth=0.5,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavenumber (\u00d710$^3$ cm$^{-1}$)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    ax.xaxis.set_major_formatter("{x:.0f}")

    secax = ax.secondary_xaxis(
        "top", functions=(mf.e3cm1_to_nm, mf.nm_to_e3cm1)
    )
    secax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=3)
    if xlimit2[0] is not None:
        ax.set_xlim(
            mf.nm_to_e3cm1(xlimit2[0][0]),
            mf.nm_to_e3cm1(xlimit2[0][1]),
        )
    if xlimit2[1] is not None:
        secax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit2[1]))
    if xlimit2[2] is not None:
        secax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit2[2]))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.tick_params(axis="x", which="major", labelsize=6, width=0.5, length=2, pad=1)
    secax.tick_params(axis="x", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    secax.spines["top"].set_linewidth(0.5)

    # ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_fsta_globalfit_concentration_symlog(
    df_1col,
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 0.2, 0.1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="line",  # "scatter"
    showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, col in enumerate(df_1col.columns):
        if legendtext is None:
            if showwn is False:
                label = f"{round(float(col))} nm"
            else:
                label = (
                    f"{round(mf.wavelength_to_wavenumber(float(col)))} cm\u207b\u00b9"
                )
        else:
            label = legendtext[i]
        if mode == "line":
            ax.plot(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                linewidth=0.5,
            )
        elif mode == "scatter":
            ax.scatter(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                # facecolors=None,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
            )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("Concentration", fontsize=8, labelpad=2)
    # ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )
    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    return fig

def display_fsta_globalfit_conc_symlog(
    df_1col,
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 0.2, 0.1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="line",  # "scatter"
    showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, col in enumerate(df_1col.columns):
        if legendtext is None:
            if showwn is False:
                label = f"species {i+1}"
            else:
                label = (
                    f"{round(mf.wavelength_to_wavenumber(float(col)))} cm\u207b\u00b9"
                )
        else:
            label = legendtext[i]

        if mode == "line":
            ax.plot(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                linewidth=0.5,
            )
        elif mode == "scatter":
            ax.scatter(
                df_1col.index,
                df_1col[col],
                label=label,
                color=colorscheme(i),
                # facecolors=None,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
            )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )
    if xlimit[0] is not None:
        ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))
    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    # ax.xaxis.set_major_formatter("{x:.0f}")

    ax.set_ylabel("Concentration", fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    # ax.yaxis.set_major_formatter("{x:.0f}")

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    return fig


def display_fsta_globalfit_das_bnm(
    df_das_wl,  # df_1col
    xlimit=[None, 50, 10],
    ylimit=[None, 2, 1],
    ylabel="DAS",
    titletext=None,
    legendtitle=None,
    legendtext=None,
    legendposition="upper right",
    gridon=False,
    show0mOD=True,
    colorscheme=plt.cm.tab10,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    nm = df_das_wl.index.values
    # cm1 = 10**7 / nm
    for i in range(0, len(df_das_wl.columns)):
        if legendtext is None:
            # label = f"{round(df_fstas_wl.columns[i],1)} ps"
            # label = mf.formalize_fsta_delaytime(df_fstas_wl.columns[i])
            label = f"species {i+1}"
        else:
            label = legendtext[i]
        ax.plot(
            nm,
            df_das_wl.iloc[:, i],
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=2)
    if xlimit[0] is not None:
        ax.set_xlim(xlimit[0])
    if xlimit[1] is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
    if xlimit[2] is not None:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))
    # ax.xaxis.set_major_formatter("{x:.0f}")

    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    if ylimit[0] is not None:
        ax.set_ylim(ylimit[0])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    # ax.yaxis.set_major_formatter("{x:.0f}")

    if show0mOD is True:
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="dotted", alpha=0.5)

    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    if titletext is not None:
        ax.set_title("Absorption Spectra")

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
        # loc=5
    )

    if gridon is True:
        ax.grid(
            which="major", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )
        ax.grid(
            which="minor", color="gray", linestyle="solid", linewidth=0.1, alpha=0.2
        )

    return fig


def display_fsta_fitted_spectra_bnm(
    list_df, # [df_spectra, df_fitted_spectra],
    xlimit=[[300, 800], 50, 10],
    ylimit=[[-10, 10], 2, 1],
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    figsize=(4, 3),
):
    if list_df[0].shape[1] != list_df[1].shape[1]:
        print("The number of spectra in the two dataframes are different.")
    else:
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=600,
            layout="constrained",
        )
        df_spectra = list_df[0]
        df_fitted_spectra = list_df[1]
        time = df_spectra.columns.values
        num_spectra = df_spectra.shape[1]
        for i in range(num_spectra):
            label = f"{round(float(time[i]),1)} ps"
            ax.scatter(
                x=df_spectra.index.values,
                y=df_spectra.iloc[:, i],
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
                color=colorscheme(i),
                # facecolors=None,
            )
            ax.plot(
                df_fitted_spectra.index.values,
                df_fitted_spectra.iloc[:, i],
                label=label,
                linewidth=1,
                color=colorscheme(i),
            )
        ax.set_xlabel("Wavelength (nm)", fontsize=8, labelpad=2)
        if xlimit[0] is not None:
            ax.set_xlim(xlimit[0])
        if xlimit[1] is not None:
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xlimit[1]))
        if xlimit[2] is not None:
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xlimit[2]))

        ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
        if ylimit[0] is not None:
            ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
        if ylimit[1] is not None:
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
        if ylimit[2] is not None:
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
        ax.axhline(y=0, color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5)
        # ax.yaxis.set_major_formatter("{x:.0f}")
        ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
        ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0.5)
        # ax.set_xscale(
        #     "symlog",
        #     base=10,
        #     linthresh=xlimit[0][1],
        #     # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        #     linscale=xlinscale,
        # )
        if titletext is not None:
            ax.set_title(titletext, fontsize=6)
        ax.legend(
            title=legendtitle,
            title_fontsize=8,
            fontsize=6,
            frameon=False,
            facecolor="none",
            edgecolor="none",
            handlelength=1.5,
            loc=legendposition,
        )
        return fig



def display_fsta_1colkinetics_fitted_linear(
    multi_df_1col,  # 3 columns, data, fit, residual
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, df_1col in enumerate(multi_df_1col):
        time = df_1col.index.values
        data = df_1col.iloc[:, 0]
        fit = df_1col.iloc[:, 1]
        # residual = df_1col.iloc[:, 2]
        if showwn is False:
            label = f"{round(df_1col.columns[0])} nm"
        else:
            label = f"{round(mf.wavelength_to_wavenumber(df_1col.columns[0]))} cm\u207b\u00b9"
        ax.scatter(
            time,
            data,
            # label=f"{round(df_1col.columns[0])} nm",
            s=2,
            alpha=0.5,
            linewidth=0.1,
            edgecolor="black",
            # facecolors=None,
            color=colorscheme(i),
        )
        ax.plot(
            time,
            fit,
            # label=f"{round(df_1col.columns[0])} nm",
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    # ax.set_xscale(
    #     "symlog",
    #     base=10,
    #     linthresh=xlimit[0][1],
    #     # subs=[2, 3, 4, 5, 6, 7, 8, 9],
    #     linscale=xlinscale,
    # )

    # ax.axvline(
    #     x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    # )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    # xticks = mf.determine_timedelay_label(xlimit)
    # ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    # ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    return fig



def display_fsta_fitted_kinetics_symlog(
    list_df, # [df_kinetics, df_fitted_kinetics],
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    df_kinetics = list_df[0]
    df_fitted_kinetics = list_df[1]
    time = df_kinetics.index.values
    wavelength = df_kinetics.columns.values
    num_kinetics = df_kinetics.shape[1]
    for i in range(num_kinetics):
        if showwn is False:
            label = f"{round(float(wavelength[i]))} nm"
        else:
            label = f"{round(mf.nm_to_cm1(float(wavelength[i])))} cm\u207b\u00b9"
        ax.scatter(
            time,
            df_kinetics.iloc[:, i],
            # label=f"{round(df_1col.columns[0])} nm",
            s=2,
            alpha=0.5,
            linewidth=0.1,
            edgecolor="black",
            # facecolors=None,
            color=colorscheme(i),
        )
        ax.plot(
            time,
            df_fitted_kinetics.iloc[:, i],
            # label=f"{round(df_1col.columns[0])} nm",
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    return fig


def display_fsta_1colkinetics_fitted_symlog(
    multi_df_1col,  # 3 columns, data, fit, residual
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    showwn=False,
    figsize=(4, 3),
):
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=600,
        layout="constrained",
    )

    for i, df_1col in enumerate(multi_df_1col):
        time = df_1col.index.values
        data = df_1col.iloc[:, 0]
        fit = df_1col.iloc[:, 1]
        # residual = df_1col.iloc[:, 2]
        if showwn is False:
            label = f"{round(df_1col.columns[0])} nm"
        else:
            label = f"{round(mf.wavelength_to_wavenumber(df_1col.columns[0]))} cm\u207b\u00b9"
        ax.scatter(
            time,
            data,
            # label=f"{round(df_1col.columns[0])} nm",
            s=2,
            alpha=0.5,
            linewidth=0.1,
            edgecolor="black",
            # facecolors=None,
            color=colorscheme(i),
        )
        ax.plot(
            time,
            fit,
            # label=f"{round(df_1col.columns[0])} nm",
            label=label,
            linewidth=1,
            color=colorscheme(i),
        )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    ax.axvline(
        x=xlimit[0][1], color="gray", linewidth=0.2, linestyle="dotted", alpha=0.5
    )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))

    if titletext is not None:
        ax.set_title(titletext, fontsize=6)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    return fig


def display_fsta_fitted2colkinetics_symlog(
    df_2col,
    xlimit=[[-5, 100, 8000], 50, 25],
    ylimit=[None, 2, 1],
    xlinscale=1,
    titletext=None,
    legendtitle=None,
    legendposition="upper right",
    colorscheme=plt.cm.tab10,
    mode="scatter",
    showwn=False,
):
    fig, ax = plt.subplots(
        figsize=(3.33, mf.calculate_height(3.33, 4 / 3)), dpi=600, layout="constrained"
    )

    for i in range(int(len(df_2col.columns) / 2)):
        wavelength = round(df_2col.columns[2 * i + 1])
        if mode == "line":
            ax.plot(
                df_2col.iloc[:, 2 * i],
                df_2col.iloc[:, 2 * i + 1],
                label=f"{wavelength} nm",
                color=colorscheme(i),
                linewidth=0.5,
            )
        elif mode == "scatter":
            ax.scatter(
                df_2col.iloc[:, 2 * i],
                df_2col.iloc[:, 2 * i + 1],
                label=f"{wavelength} nm",
                color=colorscheme(i),
                # facecolors=None,
                s=2,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
            )

    ax.set_xlabel("Time (ps)", fontsize=8, labelpad=2)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("\u0394A (mOD)", fontsize=8, labelpad=2)
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2, pad=1)
    ax.tick_params(axis="both", which="minor", labelsize=6, width=0.5, length=1, pad=1)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.legend(
        title=legendtitle,
        title_fontsize=8,
        fontsize=6,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handlelength=1.5,
        loc=legendposition,
    )

    ax.set_xscale(
        "symlog",
        base=10,
        linthresh=xlimit[0][1],
        # subs=[2, 3, 4, 5, 6, 7, 8, 9],
        linscale=xlinscale,
    )

    # ax.axvline(
    #     x=xlimit[0][1], color="gray", linewidth=0.5, linestyle="dotted", alpha=0.1
    # )
    ax.set_xlim(left=xlimit[0][0], right=xlimit[0][2])
    # ax.set_xlim(left=-50, right=8000)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    xticks = mf.determine_timedelay_label(xlimit)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks[0]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(xticks[1]))

    if ylimit[0] is not None:
        ax.set_ylim(bottom=ylimit[0][0], top=ylimit[0][1])
    if ylimit[1] is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ylimit[1]))
    if ylimit[2] is not None:
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ylimit[2]))
    if titletext is not None:
        ax.set_title(titletext, fontsize=10)

    return fig


# scale=8.5/38.81,


## kinetics fit


def display_fsta_1fitkinetics_linear(df_3cols):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_3cols.iloc[:, 0],
            xaxis="x",
            y=df_3cols.iloc[:, 1],
            yaxis="y",
            name=df_3cols.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            # hovertemplate="%{x:.0f},%{y:.1f}",
            hovertemplate="%{x},%{y:.1f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_3cols.iloc[:, 0],
            xaxis="x",
            y=df_3cols.iloc[:, 2],
            yaxis="y",
            name=df_3cols.columns[2],
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            hovertemplate="%{x:.0f},%{y:.1f}",
        )
    )
    fig.update_layout(
        template="none",
        # autosize=False,
        width=1000,
        height=750,
        # minreducedwidth=250,
        # minreducedheight=250,
        margin=dict(l=90, r=20, t=10, b=90),
        font=dict(family="Arial", color="black", size=26),
        # title=dict(
        #     # text=f"{dfabs.columns[1]}",
        #     # text=r"$\alpha_{1c} = 352 \pm 11 \text{ km s}^{-1}$",
        #     # text=r"$\large{\text{C124} \quad \tau \text{ = 4.62 ns}}$",
        #     # font_size=28,
        #     x=0.5,
        #     xanchor="center",
        #     y=0.8,
        #     yanchor="top",
        # ),
        xaxis=dict(
            title=dict(text="Time (ps)"),
            showline=True,
            # type="log",
            color="black",
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            # range=[-5, 40],
            # dtick=10,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            # minor=dict(
            #     dtick=5,
            #     ticks="outside",
            #     tickcolor="black",
            #     ticklen=5,
            #     tickwidth=2.5,
            #     showgrid=False,
            # ),
        ),
        yaxis=dict(
            # title=dict(text="Norm. Photoluminescence Intensity"),
            title=dict(text="\u0394A (mOD)"),
            showline=True,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            # range=[-0.05, 1.05],
            # dtick=0.2,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            # minor=dict(
            #     dtick=0.1,
            #     ticks="outside",
            #     tickcolor="black",
            #     ticklen=5,
            #     tickwidth=2.5,
            #     showgrid=False,
            # ),
        ),
        showlegend=True,
        legend=dict(
            # title=dict(text="Sample name"),
            font_size=28,
            xanchor="right",
            x=0.9,
            yanchor="top",
            y=0.9,
        ),
    )
    return fig


def display_fsta_1fitkinetics_linlog(df_3cols):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_3cols.iloc[:, 0],
            xaxis="x1",
            y=df_3cols.iloc[:, 1],
            yaxis="y",
            name=df_3cols.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            # hovertemplate="%{x:.0f},%{y:.1f}",
            hovertemplate="%{x:.0f},%{y:.1f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_3cols.iloc[:, 0],
            xaxis="x2",
            y=df_3cols.iloc[:, 1],
            yaxis="y",
            name=df_3cols.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            # hovertemplate="%{x:.0f},%{y:.1f}",
            hovertemplate="%{x:.0f},%{y:.1f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_3cols.iloc[:, 0],
            xaxis="x1",
            y=df_3cols.iloc[:, 2],
            yaxis="y",
            name=df_3cols.columns[2],
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            hovertemplate="%{x:.0f},%{y:.1f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_3cols.iloc[:, 0],
            xaxis="x2",
            y=df_3cols.iloc[:, 2],
            yaxis="y",
            name=df_3cols.columns[2],
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            hovertemplate="%{x:.0f},%{y:.1f}",
        )
    )
    fig.update_layout(
        template="none",
        # autosize=False,
        width=1000,
        height=750,
        # minreducedwidth=250,
        # minreducedheight=250,
        margin=dict(l=90, r=20, t=10, b=90),
        font=dict(family="Arial", color="black", size=26),
        # title=dict(
        #     # text=f"{dfabs.columns[1]}",
        #     # text=r"$\alpha_{1c} = 352 \pm 11 \text{ km s}^{-1}$",
        #     # text=r"$\large{\text{C124} \quad \tau \text{ = 4.62 ns}}$",
        #     # font_size=28,
        #     x=0.5,
        #     xanchor="center",
        #     y=0.8,
        #     yanchor="top",
        # ),
        xaxis=dict(
            title=dict(text="Time (ps)"),
            showline=True,
            type="linear",
            color="black",
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            domain=[0, 0.3],
            range=[-25, 100],
            # dtick=10,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            # minor=dict(
            #     dtick=5,
            #     ticks="outside",
            #     tickcolor="black",
            #     ticklen=5,
            #     tickwidth=2.5,
            #     showgrid=False,
            # ),
        ),
        xaxis2=dict(
            title=dict(text="Time (ps)"),
            showline=True,
            type="log",
            color="black",
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            domain=[0.3, 1],
            range=[2, 4],
            # dtick=10,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            # minor=dict(
            #     dtick=5,
            #     ticks="outside",
            #     tickcolor="black",
            #     ticklen=5,
            #     tickwidth=2.5,
            #     showgrid=False,
            # ),
        ),
        yaxis=dict(
            # title=dict(text="Norm. Photoluminescence Intensity"),
            title=dict(text="\u0394A (mOD)"),
            showline=True,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            # range=[-0.05, 1.05],
            # dtick=0.2,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            # minor=dict(
            #     dtick=0.1,
            #     ticks="outside",
            #     tickcolor="black",
            #     ticklen=5,
            #     tickwidth=2.5,
            #     showgrid=False,
            # ),
        ),
        showlegend=True,
        legend=dict(
            # title=dict(text="Sample name"),
            font_size=28,
            xanchor="right",
            x=0.9,
            yanchor="top",
            y=0.9,
        ),
    )
    return fig


def display_fsta_2fitkinetics_linear(df_3cols_1, df_3cols_2):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_3cols_1.iloc[:, 0],
            xaxis="x",
            y=df_3cols_1.iloc[:, 1],
            yaxis="y",
            name=df_3cols_1.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            # hovertemplate="%{x:.0f},%{y:.1f}",
            hovertemplate="%{x},%{y:.1f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_3cols_1.iloc[:, 0],
            xaxis="x",
            y=df_3cols_1.iloc[:, 2],
            yaxis="y",
            name=df_3cols_1.columns[2],
            mode="lines",
            line_color="#274575",
            line_width=3.5,
            hovertemplate="%{x:.0f},%{y:.1f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_3cols_2.iloc[:, 0],
            xaxis="x",
            y=df_3cols_2.iloc[:, 1],
            yaxis="y",
            name=df_3cols_2.columns[1],
            showlegend=False,
            mode="markers",
            marker=dict(
                # symbol="circle-open",
                size=6,
                line=dict(width=1),
                color="whitesmoke",
            ),
            opacity=1,
            # hovertemplate="%{x:.0f},%{y:.1f}",
            hovertemplate="%{x},%{y:.1f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_3cols_2.iloc[:, 0],
            xaxis="x",
            y=df_3cols_2.iloc[:, 2],
            yaxis="y",
            name=df_3cols_2.columns[2],
            mode="lines",
            line_color="#d83933",
            line_width=3.5,
            hovertemplate="%{x:.0f},%{y:.1f}",
        )
    )
    fig.update_layout(
        template="none",
        # autosize=False,
        width=1000,
        height=750,
        # minreducedwidth=250,
        # minreducedheight=250,
        margin=dict(l=80, r=50, t=10, b=85),
        font=dict(family="Arial", color="black", size=26),
        title=dict(
            # text=f"{dfabs.columns[1]}",
            # text=r"$\alpha_{1c} = 352 \pm 11 \text{ km s}^{-1}$",
            # text=r"$\large{\text{C124} \quad \tau \text{ = 4.62 ns}}$",
            font_size=28,
            x=0.5,
            xanchor="center",
            y=0.8,
            yanchor="top",
        ),
        xaxis=dict(
            title=dict(text="Time (ps)"),
            showline=True,
            # type="log",
            color="black",
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            range=[-100, 8000],
            dtick=1000,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                dtick=500,
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        yaxis=dict(
            # title=dict(text="Norm. Photoluminescence Intensity"),
            title=dict(text="\u0394A (mOD)"),
            showline=True,
            zeroline=False,
            linewidth=2.5,
            linecolor="black",
            # range=[-0.5, 10.5],
            dtick=2,
            ticks="outside",
            tickcolor="black",
            ticklen=10,
            tickwidth=2.5,
            showgrid=False,
            minor=dict(
                # dtick=1,
                ticks="outside",
                tickcolor="black",
                ticklen=5,
                tickwidth=2.5,
                showgrid=False,
            ),
        ),
        showlegend=True,
        legend=dict(
            # title=dict(text="Sample name"),
            font_size=28,
            xanchor="right",
            x=0.95,
            yanchor="top",
            y=0.95,
        ),
    )
    return fig



def scatters_from_dataframe_xy(df):
    fig = go.Figure()

    for i in range(int(len(df.columns) / 2)):
        if type(df.columns) == pd.core.indexes.base.Index:
            name = df.columns[2 * i + 1]
        elif type(df.columns) == pd.core.indexes.multi.MultiIndex:
            name = df.columns[2 * i + 1][-1]

        fig.add_trace(
            go.Scatter(
                x=df.iloc[:, 2 * i],
                y=df.iloc[:, 2 * i + 1],
                name=name,
                mode="lines",
                line_width=3,
                hovertemplate="%{y:.4f}",
            )
        )

    axis_template = dict(
        showgrid=True,
        zeroline=True,
        showticklabels=True,
        gridcolor="rgb(211, 211, 211)",
        gridwidth=2,
    )

    fig.update_layout(
        template=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=880,
        height=660,
        margin=dict(l=60, r=20, b=60, t=60, pad=0),
        font=dict(
            family="Arial",
            color="black",
        ),
        # legend=dict(
        #     orientation="v",
        #     xanchor="left",
        #     x=0.74,
        #     yanchor="middle",
        #     y=0.9,
        #     font=dict(size=16, color="Black"),
        # ),
        colorway=rainbow7,
    )

    fig.update_xaxes(
        title=dict(
            text="X-axis Title",
            font=dict(size=20),
            standoff=8,
        ),
        automargin=True,
        showline=True,
        linewidth=1,
        linecolor="black",
        side="bottom",
        type="linear",
        mirror=True,
        autorange=True,
        showticklabels=True,
        # dtick=50,
        tickangle=0,
        tickmode="linear",
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        ticklen=6,
        tickfont=dict(
            family="Arial",
            color="black",
            size=16,
        ),
        ticklabelposition="outside bottom",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",
        griddash="solid",
        minor=dict(
            showgrid=True,
            tickcolor="lightgrey",
            ticklen=5,
            #    dtick=25,
        ),
        minor_ticks="outside",
        zeroline=False,
    ),

    fig.update_yaxes(
        title=dict(
            text="Y-axis Title",
            font=dict(size=20),
            standoff=8,
        ),
        automargin=True,
        showline=True,
        linewidth=1,
        linecolor="black",
        side="bottom",
        type="linear",
        mirror=True,
        autorange=True,
        showticklabels=True,
        tickangle=0,
        tickmode="linear",
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        ticklen=6,
        tickfont=dict(size=16),
        ticklabelposition="outside",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",
        griddash="solid",
        minor=dict(
            showgrid=True,
            tickcolor="lightgrey",
            ticklen=5,
            #    dtick=25,
        ),
        minor_ticks="outside",
        zeroline=False,
    ),

    # fig.update_annotations(

    # ),

    return fig


def display_multi_absorption(dfem):
    fig = scatters_from_dataframe_xy(dfem)
    fig.update_layout(
        title=dict(
            text="Absorption Spectra",
            x=0.5,
            xanchor="center",
            y=0.96,
            yanchor="middle",
            font=dict(size=22),
        ),
        xaxis_title=dict(
            text="Wavelength (nm)",
            font=dict(size=20),
        ),
        yaxis_title=dict(
            text="Absorbance (a.u.)",
            font=dict(size=20),
        ),
        xaxis=dict(
            dtick=50,
            minor=dict(dtick=25),
        ),
        yaxis=dict(
            dtick=0.2,
            minor=dict(dtick=0.1),
        ),
    )
    return fig


def display_multi_emission(dfem):
    fig = scatters_from_dataframe_xy(dfem)
    fig.update_layout(
        title=dict(
            text="Emission Spectra",
            x=0.5,
            xanchor="center",
            y=0.96,
            yanchor="middle",
            font=dict(size=22),
        ),
        xaxis_title=dict(
            text="Wavelength (nm)",
            font=dict(size=20),
        ),
        yaxis_title=dict(
            text="Photoluminescence Intensity (a.u.)",
            font=dict(size=20),
        ),
        xaxis=dict(
            dtick=50,
            minor=dict(dtick=25),
        ),
        yaxis=dict(
            dtick=0.2,
            minor=dict(dtick=0.1),
        ),
    )
    return fig


def display_multi_excitation(dfex):
    fig = scatters_from_dataframe_xy(dfex)

    fig.update_layout(
        title="Excitation Spectra",
        xaxis_title="Wavelength (nm)",
        yaxis=dict(
            title="Photoluminescence Intensity (a.u.)",
        ),
    )

    return fig


def display_absorption_emission_xy1y2(dfabs, dfem):
    fig = go.Figure()

    for i in range(int(len(dfabs.columns) / 2)):
        if type(dfabs.columns) == pd.core.indexes.base.Index:
            absname = dfabs.columns[2 * i + 1]
        elif type(dfabs.columns) == pd.core.indexes.multi.MultiIndex:
            absname = dfabs.columns[2 * i + 1][-1]

        fig.add_trace(
            go.Scatter(
                x=dfabs.iloc[:, 2 * i],
                xaxis="x",
                y=dfabs.iloc[:, 2 * i + 1],
                yaxis="y1",
                name=absname,
                mode="lines",
                line_width=3,
                hovertemplate="%{x: 0f} %{y:.0f}",
            )
        )

    for i in range(int(len(dfem.columns) / 2)):
        if type(dfem.columns) == pd.core.indexes.base.Index:
            emname = dfem.columns[2 * i + 1]
        elif type(dfem.columns) == pd.core.indexes.multi.MultiIndex:
            emname = dfem.columns[2 * i + 1][-1]

        fig.add_trace(
            go.Scatter(
                x=dfem.iloc[:, 2 * i],
                xaxis="x",
                y=dfem.iloc[:, 2 * i + 1],
                yaxis="y2",
                name=emname,
                mode="lines",
                line_width=3,
                hovertemplate="%{y:.4f}",
            )
        )

    fig.update_layout(
        template="ggplot2",
        width=950,
        height=712.5,
        font=dict(family="Arial", color="black", size=22),
        title=dict(
            # text="Absorption and Emission Spectra",
            x=0.5,
            xanchor="center",
            y=0.8,
            yanchor="top",
            # font=dict(size=22),
        ),
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",
            ),
            range=[275, 850],
            # type="linear",
            # xaxis_tickvals=[0.1, 1, 10, 100, 1000, 10000],
            # xaxis_ticktext=['10^-1', '10^0', '10^1', '10^2', '10^3', '10^4'],
            dtick=50,
            minor=dict(
                showgrid=True,
                # tickcolor="black",
                # ticklen=5,
                # tick0=0,
                # dtick=25,
                # tickwidth=1,
                # ticks="outside",
            ),
        ),
        yaxis=dict(
            # title=dict(text="Absorbance (a.u.)",),
            title=dict(
                text="Absorbance",
            ),
            range=[-0.1, 1.1],
            # tickformat = '.0f',
            # dtick=2,
            minor=dict(
                showgrid=True,
                # tickcolor="black",
                # ticklen=5,
                # dtick=0.1,
                # tickwidth=1,
                # ticks="outside",
            ),
        ),
        yaxis2=dict(
            # title="Photoluminescence Intensity (a.u.)",
            title="PL. Intensity",
            range=[-0.1, 1.1],
            # color="Black",
            showline=True,
            minor=dict(
                showgrid=True,
                # tickcolor="black",
                # ticklen=5,
                # dtick=0.1,
                # tickwidth=1,
                # ticks="outside",
            ),
            linewidth=1,
            overlaying="y",
            side="right",
            # tickfont=dict(color="blue")
        ),
        showlegend=True,
        legend=dict(
            xanchor="right",
            x=1,
            yanchor="top",
            y=1,
            title=dict(
                text="Sample name",
            ),
            font=dict(size=22),
        ),
        colorway=[
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFFF00",
            "#FF00FF",
            "#00FFFF",
            "#FFA500",
            "#800080",
            "#008000",
            "#000080",
            "#800000",
            "#FFC0CB",
            "#ADD8E6",
            "#FF69B4",
            "#800000",
            "#008080",
            "#00FF7F",
            "#FFD700",
            "#FF6347",
            "#FF8C00",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
        ],
    )
    return fig


def generate_scatter_trace(df):
    trace = []
    for i in range(int(len(df.columns) / 2)):
        trace.append(
            go.Scatter(
                name=df.columns[2 * i + 1][-1],
                x=df.iloc[:, 2 * i],
                # xaxis="x1",
                y=df.iloc[:, 2 * i + 1],
                # yaxis="y1",
                mode="lines",  # "lines", "markers", "markers + lines", "none"
                line=dict(
                    # shape="spline",
                    dash="solid",  # dot, dash, dashdot
                    width=3,
                    # color="black",
                ),
                connectgaps=True,
                opacity=1,
                # visible="legendonly",
                # file="tozeroy",
                # fillcolor="grey",
                # legendrank=1,
                # hover_name=df.columns[2*i+1][-1],
                hovertemplate="%{y:.4f}",
                # hoverinfo="text",
            )
        )

    # fig.update_layout(width=800, height=600, colorway=[
    #     # high contrast
    #     '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
    #     '#00FFFF', '#FFA500', '#800080', '#008000', '#000080',
    #     '#800000', '#FFC0CB', '#ADD8E6', '#FF69B4', '#800000',
    #     '#008080', '#00FF7F', '#FFD700', '#FF6347', '#FF8C00',

    # Morandi colors
    # '#EFE1C5', '#C7BDB0', '#E2DFD6', '#B8B8A5', '#C4C7A6',
    # '#E8D0A9', '#C4B198', '#C6B5A5', '#D3C0B6', '#B0A99F',
    # '#C9C7B9', '#B3B3A3', '#C1B7B4', '#B4B1A9', '#C4C4B4',
    # '#B4B4A9', '#D3C9B8', '#B7B7A3', '#B7AFA6', '#B7A99F',
    # ],
    # hovermode="x unified",
    # )
    # colorway=[
    #         '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
    #         '#00FFFF', '#FFA500', '#800080', '#008000', '#000080',
    #         '#800000', '#FFC0CB', '#ADD8E6', '#FF69B4', '#800000',
    #         '#008080', '#00FF7F', '#FFD700', '#FF6347', '#FF8C00',
    # ],


# data
# text=df.iloc[:, 2*i+1],
# textposition='top center', # any combination of ['left', 'center', 'right', 'top', 'bottom']
# mode='lines', # any combination of ['lines', 'markers', 'text']
# # marker=dict(
# #             size=5,
# #             # color='#ffe375',
# #             opacity=1,
# #             # showscale=True,
# #             line=dict(width=1,color='MediumPurple'),
# #             ),
# line=dict(
#             dash='dot',
#             width=4,
#         ),
# connectgaps=False,
# visible=True,
# opacity=1,
# fill='tozeroy',
# fillcolor='rgba(255,255,255,0)',
# showlegend=True,
# legendwidth=1,


# template

pio.templates["em"] = go.layout.Template(
    layout=dict(
        # template="ggplot2", # ggplot2，seaborn，simple_white，plotly，plotly_white，plotly_dark，presentation，xgridoff，ygridoff，gridon，none
        autosize=False,
        height=600,
        width=800,
        # autosize=False,
        margin=dict(
            autoexpand=False,
            t=80,
            b=80,
            l=100,
            r=30,
        ),
        plot_bgcolor="white",
        # orientation="h",
        # font=dict(
        #         color="black",
        #         family="Arial",
        #         size=20,
        # )
        title=dict(
            text="Emission Spectra of Sample in Solvent.",
            font=dict(
                family="Arial",
                size=26,
                color="black",
            ),
            # x=0.1,
        ),
        xaxis=dict(
            title=dict(
                text="Wavelength (nm)",  # "<b>Wavelength (nm)<b>"
                font=dict(
                    family="Arial",
                    size=24,
                    color="black",
                ),
                standoff=10,
            ),
            showline=True,
            linewidth=1,
            linecolor="black",
            type="linear",
            side="bottom",
            mirror=False,
            showspikes=True,
            showticklabels=True,
            tick0=300,
            dtick=50,
            nticks=12,
            ticklabelstep=2,
            #     tickvals=[300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900],
            tickangle=0,
            tickfont=dict(
                family="Arial",
                color="black",
                size=20,
            ),
            #     tickprefix="",
            #     ticksuffix="nm",
            ticklabelposition="outside bottom",
            tickmode="linear",  # "auto", "linear", "array"
            ticks="outside",
            tickwidth=1,
            tickcolor="black",
            ticklen=8,
            # autorange=True, # True, False, "reversed"
            fixedrange=False,
            rangemode="nonnegative",  # "normal", "tozero", "nonnegative"
            range=(300, 900),
            domain=(0, 1),
            constrain="domain",
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGrey",
            griddash="solid",
            #     minor=dict(
            #                 showgrid=True,
            #                 tickcolor="black",
            #                 ticklen=5
            #               ),
            #     minor_ticks="outside",
            # minor_griddash='dot',
            zeroline=False,
            # zerolinewidth=1,
            # zerolinecolor='Grey',
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=dict(
                text="Absorbance (a.u.)",
                font=dict(
                    family="Arial",
                    size=24,
                    color="black",
                ),
                standoff=10,
            ),
            showline=True,
            linewidth=1,
            linecolor="black",
            type="linear",
            side="left",
            mirror=False,
            showspikes=True,
            showticklabels=True,
            tick0=0,
            dtick=0.1,
            nticks=12,
            ticklabelstep=2,
            #     tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            tickangle=0,
            tickfont=dict(
                family="Arial",
                color="black",
                size=20,
            ),
            #     tickprefix="",
            #     ticksuffix="nm",
            ticklabelposition="outside",
            #     range=(-0.05, 1.05),
            fixedrange=False,
            domain=(0, 1),
            constrain="domain",
            rangemode="normal",
            ticks="outside",
            tickwidth=1,
            tickcolor="black",
            ticklen=8,
            showgrid=True,
            gridwidth=0.25,
            gridcolor="LightGrey",
            griddash="solid",
            #     minor=dict(
            #                 showgrid=True,
            #                 tickcolor="black",
            #                 ticklen=5
            #               ),
            #     minor_ticks="outside",
            # minor_griddash='dot',
            zeroline=False,
            # zerolinewidth=1,
            # zerolinecolor='Grey',
            scaleanchor="y",
            scaleratio=1,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            entrywidth=70,
            xanchor="right",
            x=0.01,
            yanchor="top",
            y=0.5,
            # traceorder="reversed",
            title_font_family="Arial",
            font=dict(
                family="Arial",
                size=20,
                color="black",
            ),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=2,
            itemsizing="constant",
        ),
        # annotations=[
        #         dict(
        #                 visible=True,
        #                 name="draft watermark",
        #                 text="Love You!",
        #                 textangle=0,
        #                 font=dict(
        #                         family="Arial",
        #                         color="black",
        #                         size=100,
        #                         ),
        #                 opacity=0.1,
        #                 xref="paper",
        #                 yref="paper",
        #                 x=0.5,
        #                 y=0.5,
        #                 showarrow=True,
        #              ),
        #              ],
        colorscale=dict(),
        hovermode="closest",  # "x" | "y" | "closest" | False | "x unified" | "y unified"
        hoverlabel=dict(
            # bgcolor="white",
            font_size=16,
            font_family="Arial",
            align="right",  # 'left', 'right', 'auto'
        ),
        # hover_name="country",
        # hover_data=["continent", "pop"],
        # selector=dict(
        #                 type=,
        #                 mode=,
        #              ),
    ),
)


# fig.update_layout(title_text=f"Absorption spectra of {xxxx}",
#                   title_x=0.05,
#                   xaxis_title="Wavelength (nm)",
#                   yaxis_title="Absorbance (a.u.)",)


# fig.update_layout(title_text=f"Emission spectra of {xxxx}",
#                   title_x=0.05,
#                   xaxis_title="Wavelength (nm)",
#                   yaxis_title="Photoluminescence Intensity (a.u.)",)


# fig.update_layout(title_text=f"Excitation spectra of {xxxx}",
#                   title_x=0.05,
#                   xaxis_title="Excitation Wavelength (nm)",
#                   yaxis_title="Photoluminescence Intensity (a.u.)",)


# fig.update_layout(title_text=f"TCSPC of {xxxx}",
#                   title_x=0.05,
#                   xaxis_title="Time (ns)",
#                   yaxis_title="Counts",)


# output

# fig.show(renderer='svg', width=800, height=600)
# if not os.path.exists("images"):
#     os.mkdir("images")
# fig.write_image("images/fig1.svg")
# fig.to_image(format='svg',engine='orca')
# py.offline.plot(fig, filename='abcd.html', auto_open=True)


# def display_abfl(sample_name):
def display_abfl(df_ab, df_fl):
    # colors = ['#e41a1c', '#ff7f00', '#f6f926', '#4daf4a', '#0df9ff', '#377eb8', '#7e7dcd', '#000000', '#000000', '#000000', '#000000']

    trace = []

    for file in washed_files:
        sample_ab = f"{sample_name}_ab"
        sample_fl = f"{sample_name}-solvents"
        if sample_ab in file:
            ab_file = file
        elif sample_fl in file:
            fl_file = file

    ab_data = pd.read_csv(ab_file)
    ab_spectra_amounts = int(len(ab_data.columns) / 2)
    for i in range(ab_spectra_amounts):
        trace.append(
            go.Scatter(
                x=ab_data[ab_data.columns[2 * i]],
                y=ab_data[ab_data.columns[2 * i + 1]],
                name=ab_data.columns[2 * i + 1],
                mode="lines",
                line_shape="spline",
                line={
                    "width": 2,
                    "color": colors[i],
                    "dash": "dashdot",
                },
                opacity=1,
                connectgaps=True,
            )
        )

    fl_data = pd.read_csv(fl_file)
    fl_spectra_amounts = int(len(fl_data.columns) / 2)
    for i in range(fl_spectra_amounts):
        trace.append(
            go.Scatter(
                x=fl_data[fl_data.columns[2 * i]],
                y=fl_data[fl_data.columns[2 * i + 1]],
                name=fl_data.columns[2 * i + 1],
                xaxis="x",
                yaxis="y2",
                mode="lines",
                line_shape="spline",
                line={
                    "width": 2,
                    "color": colors[i],
                },
                opacity=1,
                connectgaps=True,
            )
        )

    fig = go.Figure(
        data=trace,
        layout={
            "width": 1000,
            "height": 750,
            "font": {
                "color": "black",
                "family": "Arial",
                "size": 20,
            },
            "title": {
                "text": f"absorption and fluorescence spectra of {sample_name}",
                "font": {"color": "black", "family": "Arial", "size": 30},
                "x": 0.165,
            },
            "xaxis": {
                "title": {
                    "text": "Wavelength (nm)",
                    "font": {"color": "black", "family": "Arial", "size": 30},
                },
                "side": "bottom",
                "range": [250, 750],
            },
            "yaxis": {
                "title": {
                    "text": "Absorbance (a.u.)",
                    "font": {"color": "black", "family": "Arial", "size": 30},
                },
                "side": "left",
                "range": [0, 1.05],
            },
            "yaxis2": {
                "title": {
                    "text": "Fluorescence Intensity (a.u.)",
                    "font": {"color": "black", "family": "Arial", "size": 30},
                },
                "side": "right",
                "range": [0, 1.05],
                "anchor": "y",
                "overlaying": "y",
            },
            "legend": {
                "orientation": "v",
                "x": 0.98,
                "xanchor": "right",  # "auto", "left", "center", "right"
                "y": 0.98,
                "yanchor": "top",  # "auto", "top", "middle", "bottom"
            },
        },
    )
    fig.show()
    # fig.write_image(f"{sample_name}_abfl.svg")
    return fig

    # annotations = [dict(
    #             x=5,
    #             y=1,
    #             xref='x',
    #             yref='y',
    #             text="2 ns",
    #             # showarrow=True,
    #             # arrowhead=7,
    #             # ax=0,
    #             # ay=-40,
    #             )
    # ]


# ticks

def tick_formatter_1(x, pos):
    if x >= 1:
        return f"{int(x):d}"
    elif x == 0:
        return "0"
    else:
        return f"{x:.1f}"

if __name__ == "__main__":
    # test ab
    df_ab_wl = pd.read_csv("Data/ab_wl.csv", header=0, index_col=None, sep=",")
    # print(df_ab_wl)
    # display(df_ab_wl)
    df_pl_wl = pd.read_csv("Data/pl_wl.csv", header=0, index_col=None, sep=",")
    # print(df_pl_wl)
    # display(df_pl_wl)
    # preview_2col_df(df_ab_wl)
    # display_ab_wl(df_ab_wl.iloc[:, 0:4]).show()
    # display_ab_wn(df_ab_wl.iloc[:,0:4]).show()
    # display_ab_wn_wl(df_ab_wl.iloc[:,0:2], xlimit1=[[50, 25], 2, 1]).show()
    # display(df_ab_wl)
    # mp.preview_2col_df(df_ab_wl)

    # fig = display_ab_bwl_twn(
    #     df_ab_wl,
    #     xlimit=[[275, 400], 20, 10],
    #     # xlimit2=[None, 2000, 1000],
    #     # ylimit=[[-1, 1.1], 0.4, 0.2],
    #     legendtext=np.arange(1, 17, 1),
    # )
    # plt.show()  # To display the plot
    # fig.savefig("test.svg")  # To save the plot

    # fig = display_ab_wn_wl(
    #     df_ab_wl,
    #     xlimit=[[mf.wavelength_to_wavenumber(300), mf.wavelength_to_wavenumber(850)], 5000, 1000],
    #     # legendtext=np.arange(1, 17, 1),
    # )
    # plt.show()  # To display the plot
    # fig.savefig("test.svg")  # To save the plot

    # fig = display_abpl_bwn_twl(df_ab_wl.iloc[:,4:10], df_pl_wl.iloc[:,4:10], xlimit=[[32500, mf.wavelength_to_wavenumber(850)], 5000, 1000], xlimit2=[None, 100, 25])
    # fig = display_abpl_bwn_twl(df_ab_wl, df_pl_wl, xlimit=[[32500, mf.wavelength_to_wavenumber(850)], 5000, 1000], xlimit2=[None, 100, 25])
    # plt.show()

    # fig = display_abpl_bwl_twn(
    #     df_ab_wl.iloc[:,:12],
    #     df_pl_wl.iloc[:,:12],
    #     xlimit=[[300, 850], 50, 25],
    #     # xlimit2=[None, 2000, 1000],
    #     ylimit=[[-0.02, 1.02], 0.2, 0.1],
    #     # legendtext=np.arange(1, 17, 1),
    # )
    # plt.show()

    # test pl

    # test fsta
    # df_fsta = mins.load_fsta("Data/fsta.csv")
    # print(df_fsta)
    # display(df_fsta)
    #     # display_fsta_heatmap(df_fsta).show()
    #     # display_fsta_heatmap_log(df_fsta).show()
    # kinetics = mins.extract_1colkinetics_trspectra(df_fsta, [360, 470])
    # print(kinetics)
    #     # preview_1col_df(kinetics)
    #     # display_fsta_rawkinetics_linear(
    #     #     kinetics,
    #     #     xlimit=[[-100, 1000], 100, 50]
    #     #     #  ylimit=[[-1, 7], 2, 1]
    #     # ).show()
    # display_fsta_kinetics_linear(
    #     kinetics,
    #     xlimit=[[-100, 8000], 1000, 500],
    #     # xlimit=[[-100, 8000], 1000, 500],
    #     # xlimit=[[-50, 10000], 10, None],
    #     ylimit=[[-1, 8], 2, 1],
    #     # titletext="abcd",
    #     # legendtitle="sample"
    #     # xlintresh=100,
    #     # xlinscale=0.5,
    #     # legendtitle="Wavelength",
    # )
    # display_fsta_kinetics_symlog(
    #     kinetics,
    #     # xlimit=[
    #     #     [-10, 100, 10, 1],
    #     #     [
    #     #         100,
    #     #         8000,
    #     #     ],
    #     # ],
    #     # xlimit=[[None, None, None, None],[None, None, None, None]],
    #     # xlimit=[[-10, 100, 10, 2],[100, 10000, 1, 0.1]],
    #     ylimit=[[-0.5, 7], 2, 1],
    #     # titletext="abcd",
    #     # legendtitle="sample",
    #     # xlintresh=100,
    #     # xlinscale=0.5,
    #     mode="scatter",
    # )
    # plt.show()
    # display_fsta_contourf_symlog(df_fsta)
    # plt.show()

    # display_fsta_heatmap_log_v2(df_fsta).show()

    px.colors.qualitative.swatches().show()

    px.colors.sequential.swatches().show()
    px.colors.sequential.swatches_continuous().show()

    px.colors.diverging.swatches_continuous().show()

    px.colors.cyclical.swatches_continuous().show()
