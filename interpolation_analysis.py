"""
df: dataframe
dfs: dataframeS
qdf: queried dataframe
gdf: grouped dataframe
dt: datetime
dts: datetimeS
"""
from IPython import embed
import pandas as pd
import pytz
import pyvista as pv
import matplotlib.pyplot as plt
from decimal import Decimal as D
from decimal import getcontext
import matplotlib.gridspec as gridspec

getcontext().prec = 6

# pv.global_theme.font.size = 30
# pv.global_theme.font.label_size = 30

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath, amssymb}'
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
# plt.rcParams['xtick.direction'] = 'out'
# plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['legend.handlelength'] = 5.0
# plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'grey'
plt.rcParams['grid.linewidth'] = 1.5

# Interpolation UniformGrid
# using the uniform grid allows volume views
# and it requires much less memory to store
# use decimal number to get it right
def build_uniformgrid():
    x = D(4.9)
    xs = D(0.1)
    xd = x / xs + D(1)
    y = D(1.2)
    ys = D(0.1)
    yd = y / ys + D(1)
    z = D(1.2)
    zs = D(0.1)
    zd = z / zs + D(1)
    xs = float(xs)
    ys = float(ys)
    zs = float(zs)
    xd = xd.to_integral_exact()
    yd = yd.to_integral_exact()
    zd = zd.to_integral_exact()
    grid = pv.UniformGrid(
        origin=(0, 0, 0),
        spacing=(xs, ys, zs),
        dims=(xd, yd, zd),
    )
    return(grid)


def df2polydata(df, xl='x', yl='y', zl='z', scalars=None, xlims=None, ylims=None, zlims=None):
    """
    Keyword arguments:
    df -- dataframe containing the data to use for the polydata
    scalars -- list list of scalars to be passed to the polydata
    xlims, ylims, zlims -- tuples containing min and max values
    """
    for lims in [xlims, ylims, zlims]:
        if lims is not None:
            df = df.loc[df[zl] >= lims[0]]
            df = df.loc[df[zl] <= lims[1]]
    polydata = pv.PolyData(df[[xl, yl, zl]].to_numpy())
    for s in scalars:
        polydata[s] = df[s]
    return(polydata)


def interpolate(grid, polydata, scalar, anisotropy=10):
    polydata.set_active_scalars(scalar)
    scale = (1 / anisotropy, 1, 1)
    polydata_scaled = polydata.scale(xyz=scale, inplace=False)
    grid_scaled = grid.scale(xyz=scale, inplace=False)
    inter_scaled = grid_scaled.interpolate(polydata_scaled, radius=1, sharpness=7.5, strategy='closest_point')
    grid[scalar] = inter_scaled[scalar]
    return(grid)


def plot_interpolated(grid, polydata, ID='testing', cmap='turbo', tmin=15, tmax=25):
    ID = str(ID)
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1400, 800))
    plotter.add_mesh(
        grid,
        scalars='temp',
        show_edges=False,
        edge_color='darkgrey',
        opacity=0.75,
        clim=(tmin, tmax),
        cmap=cmap,
    )
    plotter.show_bounds(
        mesh=grid,
        grid='k',
        location='outer',
        ticks='both',
        font_size=28,
        bold=True,
        font_family='arial',
        use_2d=False,
        padding=0.02,
        xlabel='\n\n X[m] ',
        ylabel='\n\n Y[m] ',
        zlabel='\n\n Z[m] ',
    )
    plotter.add_point_labels(polydata, 'labels', font_size=16, point_size=5, shape=None, point_color='k')
    plotter.add_points(points=polydata, scalars='temp', cmap=cmap, clim=(tmin, tmax), point_size=7)
    camera = [
        (-1.48216021381891, -3.6637146736069246, 3.312909422772074),
        (2.429032117488465, 0.8586964887351431, -0.047192369990395694),
        (0.34155633919364664, 0.3520678664255698, 0.8714284162151139),
    ]
    plotter.camera_position = camera
    fout = ID + '_screenshot.svg'
    plotter.save_graphic(fout)
    # plotter.show()
    # print(plotter.camera_position)


sens_spacing_types = {
    4: [0, 2, 7, 12, 17, 22, 27, 32, 37, 47, 57, 67, 77, 87, 97, 107, 117]
}
tz_mins = -8 * 60
testbed_height = 1.2
inner_csv = ';'
refresh_import = False
refresh_interpolate = False

if refresh_import:
    probe_info_csv_path = 'probe_info.csv'
    probe_info_csv_datatypes = {
        'id': int,
        'path': str,
        'valid': bool,
        'sens_num': int,
        'sens_spacing_type': int,
        'sens_bad': str,
        'x': float,
        'y': float,
        'z': float,
        'above_surface': float,
        'timezone': str,
    }
    probe_info = pd.read_csv(probe_info_csv_path, comment='#', converters=probe_info_csv_datatypes)
    # print(probe_info)
    # print(probe_info.info())
    paths: list = probe_info['path'].tolist()
    # print('probe paths are {}'.format(paths))
    datasets = []
    for rown, row in probe_info.iterrows():
        path = row['path']
        probe = row['id']
        sens_bad = row['sens_bad'].strip()
        if len(sens_bad) > 0:
            sens_bad = [int(n) for n in sens_bad.split(';')]
        # read dataset
        dataset = pd.read_csv(path, skiprows=4)
        # add header
        ncols = dataset.shape[1]
        nsens = ncols - 3
        header_fixed = ['original_index', 'datetime', 'battery']
        header_sens = [i for i in range(1, nsens + 1)]
        header = header_fixed + header_sens
        dataset.columns = header
        # fix time zone
        dataset['datetime'] = pd.to_datetime(dataset['datetime'])
        dataset['datetime'] = dataset['datetime'].dt.tz_localize(pytz.FixedOffset(0))
        dataset['datetime'] = dataset['datetime'].dt.tz_convert(pytz.FixedOffset(tz_mins))
        # check before melting
        # print(dataset)
        # melt the dataset, making sensors the value variables,
        # melt: large to long
        dataset = pd.melt(
            dataset,
            id_vars=header_fixed,
            value_vars=header_sens,
            var_name='sens',
            value_name='temp',
        )
        # check after melting
        # print(dataset)
        # add columns with information on probe and sensor
        dataset['probe'] = probe
        dataset['sens_id'] = 'p' + dataset['probe'].astype(str) + 's' + dataset['sens'].astype(str)
        dataset['x'] = row['x']
        dataset['y'] = row['y']
        dataset['z'] = row['z']
        dataset['above_surface'] = row['above_surface']
        sens_spacing_probe = sens_spacing_types[row['sens_spacing_type']]
        dataset['sens_spacing'] = [sens_spacing_probe[i - 1] for i in dataset['sens']]
        dataset['sens_z'] = dataset['z'] + (dataset['above_surface'] / 100) - (dataset['sens_spacing'] / 100)
        dataset['valid'] = True
        dataset.loc[dataset['sens'].isin(sens_bad), 'valid'] = False
        # final check
        # print(dataset)
        # append to list of datasets
        datasets.append(dataset)
    datasets = pd.concat(datasets)
    datasets = datasets.drop(['original_index', 'battery'], axis=1)
    datasets = datasets.loc[datasets['valid'] == True]
    datasets = datasets.sort_values(['datetime', 'probe', 'sens'], inplace=False)
    datasets = datasets.set_index(['datetime', 'probe', 'sens'], append=False)
    datasets.to_pickle('all_observations.pkl')

else:
    datasets = pd.read_pickle('all_observations.pkl')


if refresh_interpolate:

    # As the datetime values and ranges have to be flexible,
    # it is hard to express the comparison via grouping.
    # The current alternative is:
    # 1. list of datetimes
    # 2. set a range, so that each datetime becomes a window
    # 3. loop over the datetimes ranges and query the matching rows
    # 4. use the original (central) datetime to describe/name the quiried group
    relevant_datetimes = pd.date_range('2021-06-01 00:00:00', '2021-10-15 00:00:00', freq='1H')
    relevant_datetimes = [pd.to_datetime(dt).tz_localize(pytz.FixedOffset(tz_mins)) for dt in relevant_datetimes]
    interval = pd.Timedelta('30 min')
    relevant_intervals = [(dt - interval, dt + interval) for dt in relevant_datetimes]

    # Define a list to collect all the individual quired dfs of each datetime range
    gdf_bydt = []
    for interval, dt in zip(relevant_intervals, relevant_datetimes):
        qdf: pd.DataFrame = datasets.loc[interval[0]: interval[1]]
        qdf['gdt'] = dt
        print(
            'queried group {}\nfrom {} to {}\nfound {} results'
            .format(dt, interval[0], interval[1], qdf.shape[0])
        )
        if qdf.shape[0] == 0:
            raise UserWarning('No results for this datetime {}'.format(dt))
        gdf_bydt.append(qdf)

    # Concat to obtain a df with the relevant dts
    # it also as the gdt key to group them back
    df = pd.concat(gdf_bydt)

    # Reset the indeces after grouping, keep them as columns for later
    df = df.reset_index(inplace=False, drop=False)

    # The interpolation needs:
    # unique x, y, z, temp for each sensor of each probe
    # At the moment all of the 4 variables may be changing with time
    # within the datetime range.
    # It is necessary to drop this variability by taking the mean values.
    # done by grouping by datetime, probe, and sensor,
    # and then aggregate for the mean values.
    agg_dict = {
        'x': 'mean',
        'y': 'mean',
        'z': 'mean',
        'temp': 'mean',
        'sens': 'first',
        'sens_spacing': 'first',
        'sens_z': 'first',
    }
    agg_cols = ['gdt', 'probe', 'sens']
    df = df.groupby(agg_cols).agg(agg_dict)
    df['z'] = df['sens_z'] + testbed_height

    # The 3d grid interpolaiton can take a lot of memory
    # if there are many nodes and many datetimes are being intepolated.
    # Thus, interpolate one at the time, extract the 1d profile,
    # then drop the full 3d model before the next datetime.
    # Store the 1d information in this list.
    dfs1d = []

    gdf = df.groupby(level='gdt')
    for gkey, group in gdf:
        polydata = df2polydata(group, scalars=['temp', 'sens'])
        polydata_soil = df2polydata(group, scalars=['temp', 'sens'], zlims=(0, 1.2))
        grid = build_uniformgrid()
        grid = interpolate(grid, polydata_soil, 'temp')
        df3d = pd.DataFrame({'x': grid.x, 'y': grid.y, 'z': grid.z, 'temp': grid['temp']})
        df1d = df3d.groupby('z', as_index=False).agg({'temp': ['mean', 'std']})
        df1d.columns = ['_'.join(tup).rstrip('_') for tup in df1d.columns.values]
        df1d['gdt'] = gkey
        dfs1d.append(df1d)

    # For each dt there is the mean 1d profile of temp and its std
    df1d = pd.concat(dfs1d)

    # Add a column with only date for daily analysis
    # but keep it as datetime
    df1d['date'] = pd.to_datetime(df1d['gdt'].dt.date)

    df1d_daily = df1d.copy()

    df1d = df1d.set_index(['gdt', 'z'])

    df1d_daily = df1d.groupby(['date', 'z']).agg({'temp_mean': ['min', 'max', 'std']})
    df1d_daily.columns = ['_'.join(tup).rstrip('_') for tup in df1d_daily.columns.values]
    df1d_daily['temp_diff'] = df1d_daily['temp_mean_max'] - df1d_daily['temp_mean_min']

    df1d.to_pickle('df1d.pkl')
    df1d.to_csv('df1d.csv')

    df1d_daily.to_pickle('df1d_daily.pkl')
    df1d_daily.to_csv('df1d_daily.csv')


else:
    df1d = pd.read_pickle('df1d.pkl')
    df1d_daily = pd.read_pickle('df1d_daily.pkl')


# Month-range analysis
# this is just a grouping without aggregation,
# no need to save it as a file.
# It groups by year and month but also keeps the entire dt information,
# so that it is possible to get hourly data within each group.
df1d_monthly = df1d.groupby(
    by=[
        pd.Grouper(level=0, freq='M'),
        pd.Grouper(level=0, freq='Y'),
    ]
)

months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months_indices = range(1, 13)
months = {i: n for i, n in zip(months_indices, months_names)}

fig, axs = plt.subplots(3, df1d_monthly.ngroups, sharey=True, figsize=(16, 12))
fig.subplots_adjust(
    hspace=0.1,
    wspace=0.05,
    left=0.04,
    right=0.98,
    top=0.98,
    bottom=0.06,
)

axs[0, 0].set_ylabel('z vs lateral mean')
axs[1, 0].set_ylabel('z vs lateral std')
axs[2, 0].set_ylabel('z vs 24-h excursion')

# monthly visualization of hourly data
for i, (key, group) in enumerate(df1d_monthly):
    month = key[0].month
    year = key[1].year
    # print('i {}\nkey {}\nyear {}\nmonth {}'.format(i, key, year, month))
    for k, dthour in group.groupby(level=0):
        axs[0, i].plot(
            dthour['temp_mean'],
            dthour.index.get_level_values('z'),
            color=(0.7, 0.4, 0.4, 0.1),
            linewidth=2
        )
        axs[1, i].plot(
            dthour['temp_std'],
            dthour.index.get_level_values('z'),
            color=(0.4, 0.4, 0.7, 0.1),
            linewidth=2
        )


# Monthly visualization of 24-hour excursion:
# Pivot (unstack as the values are in the index)
# to have datetime as index, z as columns, ans temp_mean as measurement.
# Rooling 24 gives one day, get max and min
# Get the difference
# Stack back to the tidy form
# Groupby to plot
dftemp = df1d.drop(['temp_std', 'date'], axis=1).unstack(level='z')
dftemp_min = dftemp.rolling(24, min_periods=6, center=True).min()
dftemp_max = dftemp.rolling(24, min_periods=6, center=True).max()
df_excursion = dftemp_max - dftemp_min
df_excursion = df_excursion.stack(level='z')
df_excursion = df_excursion.rename(columns={'temp_mean': 'temp_diff'})
df_excursion_monthly = df_excursion.groupby(
    by=[
        pd.Grouper(level=0, freq='M'),
        pd.Grouper(level=0, freq='Y'),
    ]
)

for i, (key, group) in enumerate(df_excursion_monthly):
    month = key[0].month
    year = key[1].year
    # print('i {}\nkey {}\nyear {}\nmonth {}'.format(i, key, year, month))
    for k, hourly in group.groupby(level='gdt'):
        axs[2, i].plot(
            hourly['temp_diff'],
            hourly.index.get_level_values('z'),
            color=(0.4, 0.7, 0.4, 0.1),
            linewidth=3
        )
    axs[2, i].set_xlabel(months[month])

plt.savefig('summary_profiles.svg', dpi=300)
plt.savefig('summary_profiles.png', dpi=300)
plt.show()
