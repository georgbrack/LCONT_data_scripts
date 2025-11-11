import numpy as np
import xarray as xr
from scipy import stats
from scipy.interpolate import RectBivariateSpline as rbs
from datetime import datetime
import os
import sys


def open_LARA_zarr(filepath: str):
    """
    Opens a file as an xarray-dataset and extracts 'lon_av', 'lat_av', 'z_av' and 'hmix'.
    Fills NaN-Values with -1
    Parameters:
        filepath (str): The filepath (or URL) to the file
    
    Returns:
        lon_av (arr): A xarray-DataArray (shape = (nr. of particles, nr. of timesteps)) containing longitudinal data for each particle
        lat_av (arr): A xarray-DataArray (shape = (nr. of particles, nr. of timesteps)) conatining latitudinal data for each particle
        height_av (arr): A xarray-DataArray (shape = (nor. of particles, nr. of timestepes)) containing height above ground data for each particle
        hmix (arr): A 3 dimensional numpy array containing the temporal evolution of the mixing layer height on a 0.5° x 0.5° grid
    """
    variables = ['lon_av', 'lat_av', 'z_av', 'hmix']
    ds_dict = {var : xr.open_zarr(f'{filepath}/{var}') for var in variables}
    traj = xr.merge([ds_dict[var] for var in variables])
    lon_av = traj.lon_av.fillna(-1.)
    lat_av = traj.lat_av.fillna(-1.)
    height_av = traj.z_av.fillna(-1.)
    hmix = traj.hmix.fillna(-1.)

    return lon_av, lat_av, height_av, hmix


def calculate_mixing_layer_height(hmix, longitude_av, latitude_av) -> np.ndarray:
    """
    Interpolate the mixing layer height to the position of each particle
    
    Parameters:
        hmix (xr.DataArray): A 3 dimensional xarray-DataArray containing the temporal evolution of the mixing layer height on a grid
        longitude_av (xr.DataArray): A xarray-DataArray (shape = (nr. of particles, nr. of timesteps)) containing longitudinal data for each particle
        latitude_av (xt.DataArray): A xarray-DataArray (shape = (nr. of particles, nr. of timesteps)) conatining latitudinal data for each particle
    
    Returns:
        mixing_layer_height (np.ndarray): A 2 dimensional np-Array (shape = (nr. of particles, nr. of timesteps)) containing the mixing layer height at each particles position at each timestep
    """
    #allocate memory for topography per particle
    mixing_layer_height = np.zeros((len(longitude_av), len(longitude_av[0, :])))  # lon are the particle longitudes (particles, timesteps)

    #For each timestep, interpolate the position of the particle to the grid
    for itime in range(len(mixing_layer_height[0, :])):
        hmix_inter = interpolate2d(hmix, itime) 
        mixing_layer_height[:, itime]  = hmix_inter(latitude_av[:, itime], longitude_av[:, itime], grid=False) # lat are the latitudes of the particles (particles, timesteps)
        mini = np.min(hmix).values # The interpolation sometimes does weird stuff, so I set a max and min based on the topo data
        maxi = np.max(hmix).values
        mixing_layer_height[mixing_layer_height[:, itime] < mini, itime] = mini
        mixing_layer_height[mixing_layer_height[:, itime] > maxi, itime] = maxi
    return mixing_layer_height


def interpolate2d(field, itime):
    lon = np.arange(0., 360.5, 0.5)
    lat = np.arange(-90, 90.5, 0.5)
    if len(field.dims) == 2:
        info = field[:, :].values
    else:
        info = field[:, :, itime].values
    return rbs(lat, lon, info)


def get_is_land(landmask: xr.Dataset, lat: np.ndarray, lon: np.ndarray, time: datetime) -> np.ndarray:
    """
    Returns a numpy array to determine if particles are over land/sea ice or water
    
    Parameters:
        landmask (xr.Dataset): A landmask with the temporal evolution of sea ice concentration
        lat (np.ndarray): The latitudinal coordinates of all particles for a timestep as an array
        lon (np.ndarray): The longitudinal coordinates of all particles for a timestep as an array
        time (datetime): The time for which the result should be created
        
    Returns:
        is_land_ice (np.ndarray): An array (with the same shape as lat/lon) that contains True/False values depending on if the particle was over land/sea ice or water
    """

    particle_ds = xr.Dataset({"lat": ("points", lat), "lon": ("points", lon)})
    return landmask.sel(latitude=particle_ds.lat, longitude=particle_ds.lon, time=time, method='nearest').siconc.values.astype(bool)


def calculate_total_traveltime_particlecounter(longitude_av: np.ndarray,
                                               latitude_av: np.ndarray,
                                               height_av: np.ndarray,
                                               mixing_layer_height: np.ndarray,
                                               times: np.ndarray,
                                               lat_bins: np.ndarray,
                                               lon_bins: np.ndarray,
                                               init: bool,
                                               last_timecounter_timestep: np.ndarray,
                                               sea_ice_data: xr.Dataset):
    """
    Takes positional and time data for a list of particles and returns a grid that contains data according to how long each particle travels over land to reach each gridpoint (sum) and a sum of how many particles pass each gridpoint
    Only particles that are below the mixing layer height over land (regarding to the provided sea-ice data or land mask) are considered in the returned statistic, only time counters of particles below the mixing layer height over water are set to zero
    
    Parameters:
        longitude_av (np.ndarray): A numpy array (shape = (nr. of particles, nr. of timesteps)) containing longitudinal data for each particle and each timestep
        latitude_av (np.ndarray): A numpy array (shape = (nr. of particles, nr. of timesteps)) containing latitudinal data for each particle and each timestep
        height_av (np.ndarray): A numpy array (shape = (nr. of particles, nr. of timesteps)) containing altitude data for each particle and each timestep
        mixing_layer_height (np.ndarray): A numpy array (shape = (nr. of particles, nr. of timesteps)) containing the linearly interpolated mixing layer height for the position of each particle at each timestep
        times (np.ndarray): A numpy array (shape = (nr. of timesteps, ) or (1, nr. of timesteps)) containing the time corresponding to each timestep as a datetime64
        lat_bins (list): A list of values that serve as borders for the individual (latitudinal) bins
        lon_bins (list): A list of values that serve as borders for the individual (longitudinal) bins
        init (bool): True when the init file is being calculated, affects the counting of time over land
        last_timecounter_timestep (np.ndarray): Used when init = False, so that time over land is counted correctly
        sea_ice_data (xr.Dataset): An xarray dataset containing monthly sea ice data
        
    Returns:
        traveltime_total (np.ndarray): A grid created using the latitudinal and longitudinal bins provided that contains data for how long each particle travels over land until it reaches each gridpoint (sum)
        particle_counter (np.ndarray): A grid created using the latitudinal and longitudinal bins provided that contains data for how many particles pass each gridpoint
        last_overland_timestep (np.ndarray): The last timestep (shape = (nr. of particles, 1)) that can be used to recursively count onwards in the next loop
    """

    # allocating memory for timecounter-Array (this array calculates the time a particle has to give off moisture)
    timecounter = np.zeros_like(longitude_av, dtype=int)
    
    # creating a heightmask array
    heightmask = (height_av <= mixing_layer_height)
    
    # checking if init is True or False and calculating first timecounter array accordingly
    if init:
        timecounter[:, 0] = np.invert(np.logical_and(np.invert(get_is_land(sea_ice_data, latitude_av[:, 0], longitude_av[:, 0], time=times[0])), heightmask[:, 0]))
    elif not init:
        timecounter[:, 0] = (1 + last_timecounter_timestep[:]) * np.invert(np.logical_and(np.invert(get_is_land(sea_ice_data, latitude_av[:, 0], longitude_av[:, 0], time=times[0])), heightmask[:, 0]))
    
    for timeindex in range(1, len(times)):
        timecounter[:, timeindex] = (timecounter[:, timeindex - 1] + 1) * np.invert(np.logical_and(np.invert(get_is_land(sea_ice_data, latitude_av[:, timeindex], longitude_av[:, timeindex], time=times[timeindex])), heightmask[:, timeindex]))
    
    last_timecounter_timestep = timecounter[:, -1]
    
    # reshaping arrays so they are usable in the binned statistics function
    timecounter_reshaped = np.reshape(timecounter[heightmask], (-1, 1))[:, 0]
    timecounter_reshaped_ones = np.where(timecounter_reshaped != 0, 1, timecounter_reshaped)
    longitude_av_reshaped = np.reshape(longitude_av[heightmask], (-1, 1))[:, 0]
    latitude_av_reshaped = np.reshape(latitude_av[heightmask], (-1, 1))[:, 0]
    
    # creating mask to reduce computation time
    mask_reshaped = (timecounter_reshaped_ones == 1)
    
    # calculating the statistic
    traveltime_total = stats.binned_statistic_2d(x=latitude_av_reshaped[mask_reshaped],
                                                 y=longitude_av_reshaped[mask_reshaped],
                                                 values=timecounter_reshaped[mask_reshaped],
                                                 statistic='sum', bins=[lat_bins, lon_bins])
    particle_counter = stats.binned_statistic_2d(x=latitude_av_reshaped[mask_reshaped],
                                                 y=longitude_av_reshaped[mask_reshaped],
                                                 values=timecounter_reshaped_ones[mask_reshaped],
                                                 statistic='sum', bins=[lat_bins, lon_bins])

    return traveltime_total.statistic, particle_counter.statistic, last_timecounter_timestep


def save_arrays(traveltime_total: np.ndarray, particle_counter: np.ndarray, last_timestep: np.ndarray, savepath: str, resolution: float) -> None:
    """
    Takes an array and saves it (to the specified path) as an xarray dataset
    
    Parameters:
        traveltime_total (np.ndarray): A 2 dimensional numpy array containing the total travel time over land for each gridpoint
        particle_counter (np.ndarray): A 2 dimensional numpy array containing the particle count for each gridpoint
        last_timestep (np.ndarray): A 1 dimensional numpy array containing the last timestep data for each particle, if None it will not be saved
        savepath (str): The path (without file ending) where the dataset should be saved
        resolution (float): Resolution of the traveltime_total and particle_counter array
    Returns:
        None
    """
    
    # preparing coordinates for dataset
    lon = np.arange(-180, 180, step=resolution)
    lat = np.arange(-90, 90, step=resolution)
    # creating a dataset with the right coordinates and traveltime_total as well as particle_counter including a timestamp in the filename
    data = xr.Dataset({
        "traveltime_total": (
            ("lat", "lon"),
            traveltime_total),
        "particle_counter": (
            ("lat", "lon"),
            particle_counter)
        },
        coords = {"lat": lat, "lon": lon}
    )

    # creating a savename
    savename = savepath + '.nc'

    # saving the dataset as netcdf file
    data.to_netcdf(path=savename)

    # saving the last timestep (last column of last_timestep array) as csv if it is not None
    if last_timestep is not None:
        np.savetxt(savepath + '_last_timestep.csv', last_timestep, delimiter=',')

    return None


def calculation(directory: str,
                sea_ice_data: xr.Dataset,
                output_base_path: str,
                years_to_process: list,
                init: bool,
                last_timestep: np.ndarray) -> None:
    """
    Uses all other functions to calculate travel time for a specific list of years. It will create a directory for each year in the output_base_path and save the results there.

    Parameters:
        directory (str): The base directory where the LARA data for the specific interval is stored, can be a URL
        sea_ice_data (xr.Dataset): An xarray dataset containing monthly sea ice and land data
        output_base_path (str): The base path where the output directories and files will be saved
        years_to_process (list): A list of years (as strings) to process
        init (bool): True when the init file is being calculated, affects the counting of time
        last_timestep (np.ndarray): Used when init = False, so that time over land is counted correctly
    Returns:
        None
    """

    # setting resolution and bins
    resolution = 0.5
    lat_bins = np.arange(-90 - resolution/2, 90, step=resolution)
    lon_bins = np.arange(-180 - resolution/2, 180, step=resolution)

    # Loop over years
    for year in years_to_process:
        # set path
        yeardir = os.path.join(directory, year)
        # get sorted months
        months = [f'{m:02d}' for m in range(1, 13)]

        for month in months:
            # set path
            monthdir = os.path.join(yeardir, month)

            # import data
            lon_av, lat_av, height_av, hmix = open_LARA_zarr(monthdir)

            # set particlenumber
            if init:
                particlenumber = len(lon_av)
            elif last_timestep is not None:
                particlenumber = len(last_timestep)
            else:
                raise ValueError("init=False but last_timestep is None. Cannot determine particle number.")
            
            mixing_layer_height = calculate_mixing_layer_height(hmix, lon_av[:particlenumber], lat_av[:particlenumber])
            lon_av = (lon_av + 180) % 360 - 180
            traveltime_total, particle_counter, last_timestep = calculate_total_traveltime_particlecounter(
                                                                   lon_av[:particlenumber].values,
                                                                   lat_av[:particlenumber].values,
                                                                   height_av[:particlenumber].values,
                                                                   mixing_layer_height,
                                                                   lon_av.time.values,
                                                                   lat_bins,
                                                                   lon_bins,
                                                                   init,  # Use the current state
                                                                   last_timestep,
                                                                   sea_ice_data)
            
            # saving the file to its respective directory
            output_yeardir = os.path.join(output_base_path, year)
            os.makedirs(output_yeardir, exist_ok=True)
            filepath = os.path.join(output_yeardir, 'travel_times_' + str(month))

            # save arrays
            if month == months[-1]:
                save_arrays(traveltime_total, particle_counter, last_timestep, filepath, resolution)
            else: 
                save_arrays(traveltime_total, particle_counter, None, filepath, resolution)

            init = False
        
    return None


#------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    SEA_ICE_CONC_THRESHOLD = 0.9

    # parse arguments
    if len(sys.argv) < 3:
        print("Usage: calculate_travel_times.py <interval_index> <mode (full or spinup)>")
        sys.exit(1)

    intervalindex = int(sys.argv[1])
    mode = sys.argv[2].lower()

    if mode not in ['full', 'spinup']:
        print(f'Error: Invalid mode "{mode}". Choose "full" or "spinup".')
        sys.exit(1)

    # change this to your paths
    laradir = 'https://data.eodc.eu/collections/LARA'
    output_base_path = '.ADD OUTPUT BASE PATH HERE'
    sea_ice_data_path = './ERA5_sea-ice-cover.grib'  #Path to sea ice data file 
    # Sea-Ice Data: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download


    intervals = ['1940-1947',
                 '1947-1954',
                 '1954-1961',
                 '1961-1968',
                 '1968-1975',
                 '1975-1982',
                 '1982-1989',
                 '1989-1996',
                 '1996-2003',
                 '2003-2010',
                 '2010-2017',
                 '2017-2024']
    
    # load sea ice data
    print("Loading and processing sea ice data...")
    sea_ice = xr.open_dataset(sea_ice_data_path, engine='cfgrib')
    sea_ice = sea_ice.fillna(1)
    sea_ice = xr.where(sea_ice > SEA_ICE_CONC_THRESHOLD, 1, 0)
    # rearrange longitudes to (-180, 180)
    sea_ice['longitude'] = xr.where(sea_ice.longitude > 180, sea_ice.longitude - 360, sea_ice.longitude)
    sea_ice = sea_ice.sortby('longitude')

    # get current interval info
    current_interval_name = intervals[intervalindex]
    current_interval_datadir = os.path.join(laradir, current_interval_name)
    
    # extract start and end year and create interval
    start_year_str, end_year_str = current_interval_name.split('-')
    start_year, end_year = int(start_year_str), int(end_year_str)
    all_years_in_interval = [str(y) for y in range(start_year, end_year + 1)]

    # execute based on mode
    if mode == "full":
        # full run
        print(f"Running FULL calculation for interval {intervalindex} ({current_interval_name})")

        years_to_process = all_years_in_interval[:-1]
        print(f"Processing years: {years_to_process}")

        calculation(directory=current_interval_datadir,
                    sea_ice_data=sea_ice,
                    output_base_path=output_base_path,
                    years_to_process=years_to_process,
                    init=True,
                    last_timestep=None)
        
    elif mode == "spinup":
        # spin up recalculation using last month of previous interval as init
        print(f"Running SPINUP calculation for interval {intervalindex} ({current_interval_name})")

        if intervalindex == 0:
            print("Error: Cannot run spinup for the first interval as there is no previous interval.")
            sys.exit(1)

        # get previous interval info
        previous_interval_name = intervals[intervalindex - 1]
        previous_interval_datadir = os.path.join(laradir, previous_interval_name)


        previous_start_str, previous_end_str = previous_interval_name.split('-')
        previous_start, previous_end = int(previous_start_str), int(previous_end_str)
        previous_years = [str(y) for y in range(previous_start, previous_end + 1)]

        last_processed_year_str = previous_years[-2]

        # construct path to the timestep csv
        csv_filename = "travel_times_12_last_timestep.csv"
        last_timestep_filepath = os.path.join(output_base_path, last_processed_year_str, csv_filename)

        print(f"Loading last timestep data from {last_timestep_filepath}")

        # load the data
        try:
            initial_last_timestep = np.loadtxt(last_timestep_filepath, delimiter=",")
        except Exception as e:
            print(f"Error loading last timestep data: {last_timestep_filepath}")
            print(f"Did Stage 1 finish for interval {intervalindex - 1}? Error: {e}")
            sys.exit(1)
        
        # identify spin-up year to recalculate
        spinup_year = all_years_in_interval[0]
        print(f"Recalculating spin-up year: {spinup_year}")

        calculation(directory=current_interval_datadir,
                    sea_ice_data=sea_ice,
                    output_base_path=output_base_path,
                    years_to_process=[spinup_year],
                    init=False,
                    last_timestep=initial_last_timestep)
        
    print(f"Run completed for mode: {mode}")