import os
import numpy as np
import re
from glob import glob1
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import timeit


def FileDataCreator():
    os.chdir('./')
    wd = os.getcwd()
    results = os.path.join(wd, 'logs')
    # Pattern for getting data from outputs
    pattern_nodes = re.compile(r'(p\d{4})\n', re.IGNORECASE)
    pattern_model = re.compile(r'model\: (\w+)', re.I)
    pattern_size = re.compile(r'Global lattice size: (\d+)x(\d+)x(\d+)', re.I)
    pattern_local_size = re.compile(r'Local lattice size: (\d+)x(\d+)x(\d+)', re.I)
    # pattern_devices = re.compile(r'Selecting device', re.I)
    pattern_devices = re.compile(r'icm.edu.pl', re.I)
    pattern_time = re.compile(r'Total duration: (\d+\.\d+)', re.I)
    pattern_speed = re.compile(r'(\d+\.\d+) MLBUps', re.I)

    data_frame = pd.DataFrame(
        columns=['Name', 'Model', 'Nodes', 'Devices', 'Speed', 'Time', 'X', 'Y', 'Z', 'Total size', 'Local size'])
    for txtFile in glob1(results, "*.out"):
        dane = os.path.join(results, txtFile)
        with open(dane, "r") as f:
            content = f.read()

        # find data that fits to patterns
        matches_nodes = pattern_nodes.finditer(content)
        matches_model = pattern_model.finditer(content)
        matches_size = pattern_size.finditer(content)
        matches_local_size = pattern_local_size.finditer(content)
        matches_devices = pattern_devices.findall(content)
        matches_time = pattern_time.finditer(content)
        matches_speed = pattern_speed.finditer(content)

        # need to capture data to a list
        Name = txtFile
        Speed = np.mean(np.array([float(match.group(1)) for match in matches_speed])[-100:])
        for match in matches_model:
            Model = match.group(1)
        if pattern_time.search(content) == None:
            Time = None
        else:
            for match in matches_time:
                Time = match.group(1)
        Node = []
        for match in matches_nodes:
            temp = match.group(1)
            Node.append(temp)
        Devices = len(matches_devices)
        for match in matches_size:
            SizeX = match.group(1)
            SizeY = match.group(2)
            SizeZ = match.group(3)

        Total_size = int(SizeX) * int(SizeY) * int(SizeZ)
        # Global_size = SizeX + "x" + SizeY + "x" + SizeZ

        for match in matches_local_size:
            LocalX = match.group(1)
            LocalY = match.group(2)
            LocalZ = match.group(3)

        Local_size = LocalX + "x" + LocalY + "x" + LocalZ

        # creating a list and pandas row to append to DataFrame
        to_append = [Name, Model, Node, Devices, Speed, Time, SizeX, SizeY, SizeZ, Total_size, Local_size]
        a_series = pd.Series(to_append, index=data_frame.columns)
        data_frame = data_frame.append(a_series, ignore_index=True)

    data_frame.to_csv(r'export_dataframe.csv', header=True, index=False)


def PlotCreator():
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 16
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    df1 = pd.read_csv('export_dataframe.csv')
    F1 = df1['Model'].unique().tolist()
    Filt_model = [(df1['Model'] == a) for a in F1]
    # For strong scaling
    x_uniq = df1['X'].unique().tolist()
    y_uniq = df1['Y'].unique().tolist()
    z_uniq = df1['Z'].unique().tolist()
    Filt_size_x = [(df1['X'] == i_x) for i_x in x_uniq]
    Filt_size_y = [(df1['Y'] == i_y) for i_y in y_uniq]
    Filt_size_z = [(df1['Z'] == i_z) for i_z in z_uniq]
    # For weak scaling
    size_uniq = df1['Local size'].unique().tolist()
    weak_filt = [df1['Local size'] == size for size in size_uniq]
    # for plot speedup(A/V)
    GPU_load = (df1['Total size'] / df1['Devices']).unique().tolist()
    Filt_GPU_load = [(df1['Total size'] / df1['Devices'] == load) for load in GPU_load]

    wd = os.getcwd()

    dirName= os.path.join(wd, 'plots')
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    os.chdir(dirName)
    # Making Weak Scaling Plot
    a = 0
    for i in range(len(Filt_model)):
        for iter in range(len(weak_filt)):
            temp_df = df1.loc[Filt_model[i]].loc[weak_filt[iter]]
            if not temp_df.empty and len(temp_df['Devices']) > 2:
                a += 1
                x = temp_df.sort_values(by='Devices', ascending=True)['Devices']
                y = temp_df.sort_values(by='Devices', ascending=True)['Speed']
                name = temp_df['Model'].unique()
                local_size = str(temp_df['Local size'].unique().tolist()[0])
                make_plot_weak(x, y, name[0] + str(a), local_size)

    # Making Strong Scaling Plot
    a = 0
    for i in range(len(Filt_model)):
        for i_x in range(len(Filt_size_x)):
            for i_y in range(len(Filt_size_y)):
                for i_z in range(len(Filt_size_z)):
                    temp_df = df1.loc[Filt_model[i]].loc[Filt_size_x[i_x]].loc[Filt_size_y[i_y]].loc[Filt_size_z[i_z]]
                    if not temp_df.empty and len(temp_df['Devices']) > 2:
                        x = temp_df.sort_values(by='Devices', ascending=True)['Devices']
                        y = temp_df.sort_values(by='Devices', ascending=True)['Speed']
                        name = temp_df['Model'].unique()
                        global_size = str(temp_df['X'].unique().tolist()[0]) + 'x' + str(
                            temp_df['Y'].unique().tolist()[0]) + 'x' + str(temp_df['Z'].unique().tolist()[0])
                        make_plot_strong(x, y, name[0] + str(a), global_size)
                        a += 1

    # Plot ghost layers
    a = 0
    for i in range(len(Filt_model)):
        for iter in range(len(Filt_GPU_load)):
            temp_df = df1.loc[Filt_model[i]].loc[Filt_GPU_load[iter]]
            temp_df = temp_df.sort_values(by='Devices', ascending=True)
            if not temp_df.empty and len(temp_df['Devices']) > 4:
                '''x = temp_df['Devices'] * temp_df['X'] * temp_df['Z'] / (temp_df['Total size'] / temp_df['Devices'])
                y = temp_df['Speed']
                name = temp_df['Model'].unique()
                #total_size = str((temp_df['Total size']/df1['Devices']).unique().tolist()[0])
                make_plot_ghost(x, y, name[0] + str(a))'''
                make_plot_ghost(temp_df, str(a))
            a += 1

    os.chdir("./")

def make_plot_weak(x, y, name, size):
    # fig = plt.plot(x, y, color='black', marker='x', markevery=1, markersize=7, linestyle=":", linewidth=2, label=f'bla')
    fig = plt.plot(x, y, 'kx')
    m, b = np.polyfit(x, y, 1)
    x = np.linspace(0, np.amax(x) + 1, 20)
    plt.plot(x, m * x + b, color='black', markevery=1, markersize=7, linestyle=":", linewidth=2, label=f'bla')
    plt.xlabel('Number of GPU', fontsize=24)
    plt.ylabel('MLBUps', fontsize=24)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.xticks(np.arange(0, max(x) + 1, 1.0))
    plt.grid(True)
    plt.savefig("Weak_scaling_" + name + "_" + size + "_2", dpi=600)
    plt.clf()


def make_plot_strong(x, y, name, size):
    fig = plt.plot(x, y, color='black', marker='x', markevery=1, markersize=7, linestyle=":", linewidth=2, label=f'bla')
    # fig = plt.plot(x, y, 'kx')
    # m, b = np.polyfit(x, y, 1)
    # x = np.linspace(0, np.amax(x) + 1, 20)
    # plt.plot(x, m * x + b, color='black', markevery=1, markersize=7, linestyle=":", linewidth=2, label=f'bla')
    plt.xlabel('Number of GPU', fontsize=24)
    plt.ylabel('MLBUps', fontsize=24)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.xticks(np.arange(0, max(x) + 1, 1.0))
    plt.grid(True)
    plt.savefig("Strong_scaling_" + name + '_' + size, dpi=600)
    plt.clf()


def make_plot_ghost(df, a):
    fig, ax = plt.subplots()
    A_V = df['Devices'] * df['X'] * df['Z'] / (df['Total size'] / df['Devices'])
    df2=pd.concat([df, A_V], axis=1)
    df2 = df2.rename(columns={0: 'A/V'})

    df2 = df2.sort_values(by='A/V', ascending=True)

    temp_df = df2.loc[df2['Devices'] == 1]
    if not temp_df.empty:
        x1 = temp_df['A/V']
        y1 = temp_df['Speed']
        line1 = ax.plot(x1, y1, color="black", marker="x", markevery=1, markersize=7, linestyle="-", linewidth=2,
                    label=f'1 GPU')

    temp_df = df2.loc[df2['Devices'] == 2]
    if not temp_df.empty:
        x2 =temp_df['A/V']
        y2 = temp_df['Speed']
        line2 = ax.plot(x2, y2, color="black", marker=">", markevery=1, markersize=7, linestyle=":", linewidth=2,
                        label=f'2 GPU')

    temp_df = df2.loc[df2['Devices'] == 4]
    if not temp_df.empty:
        x3 = temp_df['A/V']
        y3 = temp_df['Speed']

        line3 = ax.plot(x3, y3, color="black", marker="<", markevery=1, markersize=7, linestyle="-", linewidth=2,
                    label=f'4 GPU')

    temp_df = df2.loc[df2['Devices'] == 8]
    if not temp_df.empty and len(temp_df['Devices']) > 4:
        x4 = temp_df['Devices'] * temp_df['X'] * temp_df['Z'] / (temp_df['Total size'] / temp_df['Devices'])
        y4 = temp_df['Speed']
        line4 = ax.plot(x4, y4, color="black", marker="v", markevery=1, markersize=7, linestyle=".", linewidth=2,
                    label=f'8 GPU')

    ax.legend()
    plt.title('Speed in function of A/V', fontsize=32)
    plt.xlabel('A/V', fontsize=24)
    plt.ylabel('MLBUps', fontsize=24)
    plt.ylim(ymin=0)
    plt.grid(True)
    name = df['Model'].unique()
    plt.savefig("AV_plot_" + name[0]+a, dpi=600)
    plt.clf()


def __main__():
    start = timeit.default_timer()
    wd = os.getcwd()
    path = os.path.join(wd, 'export_dataframe.csv')
    if os.path.isfile(path):
        PlotCreator()
    else:
        FileDataCreator()

    t1 = float(timeit.default_timer() - start)
    print("Czas :", str(t1))


__main__()
