"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple
import os

import numpy as np
import pandas as pd
from numba import njit

# Means and stds for later normalizing and re-scaling
SCALER = {
    'input_means': np.array([59.58, 42.17, 71.14, 2.56, 29.92, 145.11]),
    'input_stds': np.array([18.24, 19.42, 17.38, 3.23, 0.55, 75.52]),
    'output_mean': np.array([11.587152]), 
    'output_std': np.array([79.574436]) 
}

INVALID_ATTR = []

def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the 
        `data` directory in the main folder of this repository., by default None
    
    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """
    attributes_path = Path(camels_root) / 'demo.csv'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    #print('hello' + os.getcwd())
    df = pd.read_csv(attributes_path, dtype={'FIPS': str})
    df['FIPS'] = df['FIPS'].apply(lambda x: x.zfill(5))
    df = df.set_index('FIPS')

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql('basin_attributes', conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(db_path: str,
                    basins: List,
                    keep_features: List = None) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='FIPS')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)
    #print(df)
    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """

    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


#@njit(np.ndarray(int, int), np.ndarray(int, int), int)
#@njit(int[:,:], int[:,:], int)
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where 
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_forcing(camels_root: PosixPath, basin: str) -> Tuple[pd.DataFrame, int]:
    """Load Maurer forcing data from text files.

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the Maurer forcing
    area: int
        Catchment area (read-out from the header of the forcing file)

    Raises
    ------
    RuntimeError
        If not forcing file was found.
    """
    basindict = get_basin_dict()
    #print(basindict)
    forcing_path = camels_root / 'envirodata' 
    files = list(forcing_path.glob('*.csv'))
    file_path = [f for f in files if f.name[:-4] == basindict[basin]]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basindict[basin]} at {file_path}')
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path)
    df.index = pd.to_datetime(df['Date'])

    area = 1 # we don't need the area here
    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> pd.Series:
    """[summary]

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    area : int
        Catchment area, used to normalize the discharge to mm/day

    Returns
    -------
    pd.Series
        A Series containing the discharge values.

    Raises
    ------
    RuntimeError
        If no discharge file was found.
    """
    discharge_path = camels_root / 'dailycases.csv'
    df = pd.read_csv(discharge_path)
    df['FIPS'] = df['FIPS'].apply(str).apply(lambda x: x.zfill(5))
    QObs = df.loc[df['FIPS'] == basin].T.drop("FIPS").reset_index().iloc[1:,:]
    QObs.index = pd.to_datetime(QObs['index'])
    QObs = QObs.drop('index', axis=1).squeeze()
    #basin_dict = get_basin_dict()
    #fips = basin_dict['basin']
    #print(QObs)
    return QObs

def get_basin_dict():
    """Generate dictionary mapping station names to FIPS codes.

    Returns
    -------
    Dict
        Dict mapping stn : FIPS
    """
    basin_file = Path(__file__).absolute().parent.parent / "data/fips_stn.csv"
    fips_stn = pd.read_csv(basin_file)
    fips_stn['FIPS'] = fips_stn['FIPS'].apply(str).apply(lambda x: x.zfill(5))
    fips_stn = fips_stn.set_index('FIPS')
    fips_stn_dict = fips_stn.to_dict()
    fips_stn_dict = fips_stn_dict['stn']
    return fips_stn_dict
