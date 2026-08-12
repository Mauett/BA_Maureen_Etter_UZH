import numpy as np
import pandas as pd
import uproot
import concurrent.futures
import os
import glob
import warnings
from typing import Optional, Union


class ADCRawData:
    """General class to import the analog-to-digital converter (ADC) raw data.

    Attributes:
        verbose: Verbosity of output.
        raw_input_path: Path with the ROOT files to be imported.
        raw_input_fileslist: Array of ROOT files to be imported.
        adc_f: ADC sampling frequency in samples per second.
        adc_r: ADC resolution in volts per bin.
        adc_z: ADC input impedance.
        adc_a: Amplification factor of readout.
        elementary_charge: Electron charge in C.
        adc_area_to_e: Conversion factor pulse area in ADC units to charge in units of elementary charge.
    """

    def __init__(self, raw_input_path: str, raw_input_filepattern: str = '*.root',
                 adc_type: str = 'v1730d', verbose: bool = False):
        """Init of the ADCRawData class.

        Defines the list of files to be loaded and global parameters from the data acquisition.

        Args:
            raw_input_path: Path with the ROOT files to be imported.
            raw_input_filepattern: Name of file or pattern of files to be imported. Default: `*root` for
                all ROOT files in `raw_input_path`.
            adc_type: ADC model used for the data acquisition. Options: `v1724`, `v1730d` (default).
            verbose: Set verbose output.
        """
        self.verbose = verbose
        self.raw_input_path = raw_input_path
        self.raw_input_fileslist = glob.glob(os.path.join(raw_input_path, raw_input_filepattern))
        if len(self.raw_input_fileslist) == 0:
            raise ValueError('No files found to be loaded. Provide valid path and filename pattern.')
        elif '.root' not in ' '.join(self.raw_input_fileslist):
            raise TypeError('There seem to be no .root files among the selected files.')
        elif self.verbose:
            print('Selected following files to be loaded:')
            print(*[el.split(os.sep)[-1] for el in self.raw_input_fileslist], sep="\n")
        # Define conversion factors depending on ADC type
        self.get_conversion_factors(adc_type)

    def get_conversion_factors(self, adc_type: str):
        """Define global conversion factors depending on ADC type.

        Args:
            adc_type: ADC model used for the data acquisition. Options: `v1724`, `v1730d`.
        """
        self.elementary_charge = 1.60218e-19  # electron charge
        if str(adc_type).lower() in ['v1730d', 'v1730', '1730', '1730d']:  # CAEN V1730D
            self.adc_f = 500e6  # ADC sampling frequency: 500 MS/s digitization speed (2 ns bins)
            self.adc_r = 2.0 / 2 ** 14  # ADC resolution in volts per bin: 14 bit ADC, 2V voltage range
            self.adc_z = 50  # input impedance: 50 Ohm termination to ground
            self.adc_a = 10  # amplification factor: 10 times gain into 50 Ohm impedance
        elif str(adc_type).lower() in ['v1724', '1724']:  # CAEN V1730D
            self.adc_f = 100e6  # ADC sampling frequency: 100 MS/s digitization speed (10 ns bins)
            self.adc_r = 2.25 / 2 ** 14  # ADC resolution in volts per bin: 14 bit ADC, 2.25V voltage range
            self.adc_z = 50  # input impedance: 50 Ohm termination to ground
            self.adc_a = 10  # amplification factor: 10 times gain into 50 Ohm impedance
        else:
            raise ValueError(
                '{} is no valid option for `adc_type`. Select from (`v1724`, `v1730d`).'.format(adc_type))
        # Conversion factor pulse area in ADC units to charge in units of elementary charge.
        self.adc_area_to_e = self.adc_r / (self.adc_f * self.adc_z * self.adc_a * self.elementary_charge)

    def set_run_conditions(self):
        """Define the run specific conditions.
        TODO: fill, extract info from raw_input_path
        """
        pass

    def get_trees(self) -> str:
        """Find name of unique available tree in ROOT files to be loaded.

        Returns:
            tree: Unique ROOT tree name.
        """
        # Iterate over selected files.
        for i, input_file in enumerate(self.raw_input_fileslist):
            with uproot.open(input_file) as file:
                # Get tree names in file.
                trees_file = file.keys()
                if len(trees_file) == 0:
                    raise ValueError('No trees found in selected ROOT file {}.'.format(input_file))
                # Convert tree names to string.
                trees_file = [el for el in trees_file]
                # For large data sets, ROOT may generate additional copies of a
                # particular tree, consequently named e.g. `t1;1`, `t1;2`,...
                # We only want the name before the semicolon.
                trees_file = np.unique([el.split(';')[0] for el in trees_file])
            # Concatenate with previous iterations.
            if i != 0:
                trees = np.unique(np.concatenate([trees, trees_file], axis=0))
            else:
                trees = trees_file
        # Require and return only one unique tree name.
        if len(trees) > 1:
            raise ValueError('Multiple ({}) trees found in selected ROOT file.'
                             'Specify single tree to be loaded.'.format(len(trees)))
        else:
            tree = str(trees[0])
        return tree

    def get_branches(self, tree: Optional[str] = None) -> np.ndarray:
        """Find branch names in ROOT files to be loaded.

        Args:
            tree: Name of the ROOT tree to be inspected. If `None` deduce with
                `pmt_analysis.utils.input.ADCRawData.get_trees` method.

        Returns:
            branches: Array with branch names in selected ROOT files.
        """
        # Define tree to be inspected.
        if tree is None:
            tree = self.get_trees()
        else:
            if type(tree) != str:
                raise TypeError('Parameter `tree` must be of type `str`.')
        # Iterate over selected files.
        for i, input_file in enumerate(self.raw_input_fileslist):
            with uproot.open(input_file) as file:
                # Get branch names in given tree.
                branches_file = file[tree].keys()
                if len(branches_file) == 0:
                    raise ValueError('No branches found for tree {} in selected ROOT file {}.'.format(tree, input_file))
                # Convert branch names to string.
                branches_file = np.array([el for el in branches_file])
            # Concatenate with previous iterations.
            if i != 0:
                branches = np.unique(np.concatenate([branches, branches_file], axis=0))
            else:
                branches = branches_file
        return branches

    def get_branch_data(self, branch: Union[str, int], tree: Optional[str] = None) -> np.ndarray:
        """Retrieve data of specified branch and tree in all selected ROOT files.

        Args:
            branch: Branch of ROOT file to be loaded. Also allows for input of an ADC channel number (int).
            tree: ROOT tree to load. If `None` deduce with `pmt_analysis.utils.input.ADCRawData.get_trees` method.
        Returns:
            out: Array with data of selected branch. Typically, Unix timestamp for `branch = 'Time'` or
                ADC data of selected channel, e.g. waveforms of channel 0 for `branch = 'wf0'` or `branch = 0`.
        """
        # Define tree to be inspected.
        if tree is None:
            tree = self.get_trees()
        else:
            if type(tree) != str:
                raise TypeError('Parameter `tree` must be of type `str`.')
        # Make optional to pass channel number (int) for branch.
        if type(branch) == int:
            branch = 'wf{}'.format(branch)
        if type(branch) != str:
            raise TypeError('Parameter `branch` must be of type `str` or `int`.')
        # Check availability of selected branch.
        if branch not in self.get_branches(tree):
            raise ValueError('Branch {} not found in tree {} of selected ROOT files.'.format(branch, tree))
        # Iteratively load data from ROOT files.
        executor = concurrent.futures.ThreadPoolExecutor(8)
        out = uproot.concatenate(files={el: tree for el in self.raw_input_fileslist},
                                 expressions=branch,
                                 library='np',
                                 step_size=100000,
                                 allow_missing=True,
                                 decompression_executor=executor,
                                 interpretation_executor=executor
                                 )[branch]
        return out


class ScalerRawData:
    """General class to import the CAEN V260 scaler raw data from space-separated `.dat` files.

    Attributes:
        verbose: Verbosity of output.
        trim_empty: Remove columns for non-active channels.
        files: List of full file path and name of all `.dat` files to be loaded.
        t_int: Data acquisition interval in seconds.
    """

    def __init__(self, files: Union[str, list], trim_empty: bool = True, verbose: bool = True):
        """Init of the ScalerRawData class.

        Defines the list of files to be loaded and global parameters from the data acquisition.

        Args:
            files: Files to be loaded. Possible formats: 
                String of full file path and name for a single file to be loaded; 
                list of strings of full file paths and names for multiple files to be loaded; 
                string of full file path for the parent directory of all `.dat` files to be loaded.
            trim_empty: Remove columns for non-active channels.
            verbose: Verbosity of output.
        """
        self.verbose = verbose
        self.trim_empty = trim_empty
        self.files = self.convert_input_path(input_str_or_list=files)
        if verbose:
            print('Files to be loaded:')
            print(*self.files, sep="\n")
        self.t_int = self.get_t_int()
        if verbose:
            print('Data acquisition interval: {} s'.format(self.t_int))

    @staticmethod
    def convert_input_path(input_str_or_list: Union[str, list]) -> list:
        """Convert `input` parameter to appropriate format, 
        i.e. list of strings indicating full file paths and names.

        Args:
            input_str_or_list: Files to be loaded. Possible formats:
                String of full file path and name for a single file to be loaded;
                list of strings of full file paths and names for multiple files to be loaded;
                string of full file path for the parent directory of all `.dat` files to be loaded.
        """
        # Construct list of strings with paths and names of files to be loaded.
        if type(input_str_or_list) == str:
            if os.path.isfile(input_str_or_list):
                output_list = [input_str_or_list]  # single file name to list
            elif os.path.isdir(input_str_or_list):
                output_list = glob.glob(os.path.join(input_str_or_list, '*.dat'))  # find all .dat files in directory
                if len(output_list) < 1:
                    raise ValueError('No files found to be loaded.')
            else:
                raise ValueError('Cannot access {}: No such file or directory'.format(input_str_or_list))
        elif type(input_str_or_list) != list:
            raise TypeError('Values for file parameter must be of type str or list, '
                            'but is of type {}.'.format(type(input_str_or_list)))
        else:
            output_list = input_str_or_list

        # Remove possible file names not in .dat format.
        if np.any(~np.array(['.dat' in el for el in output_list])):
            warnings.warn('Removing files not in .dat format.')
            output_list = [el for el in output_list if '.dat' in el]
            if len(output_list) < 1:
                raise ValueError('No files found to be loaded.')

        return output_list

    def get_t_int(self) -> int:
        """Get data acquisition interval and ensure that all files have the same acquisition interval.

        Returns:
            t_int: Data acquisition interval in seconds.
        """
        # Obtain acquisition intervals from files to be loaded.
        t_int_list = []
        for file in self.files:
            with open(file) as f:
                first_line = f.readline().strip('\n').split(' ')
                t_int_list.append(int(first_line[first_line.index('Interval:') + 1]))
        # Ensure that all files have the same acquisition interval.
        if np.unique(t_int_list).shape[0] == 1:
            t_int = t_int_list[0]
        else:
            raise ValueError('Loaded files have different acquisition intervals.')

        return t_int

    def get_data(self) -> pd.DataFrame:
        """Load scaler data.

        Returns:
            df: Pandas data frame with scaler data. Contains timestamps, datetimes, as well as counts (`ch*_cnts`)
                and count rates (`ch*_freq`) in the respective data acquisition intervals for the individual channels.
        """
        scaler_column_names = np.concatenate([np.array(['timestamp']),
                                              np.array([['ch{}_cnts'.format(i), 'ch{}_freq'.format(i)]
                                                        for i in range(16)]).flatten()])
        df_list = (pd.read_csv(file, sep='\s+', lineterminator='\n', skiprows=1,
                               header=None, names=scaler_column_names) for file in self.files)
        df = pd.concat(df_list, ignore_index=True)

        # Remove columns with no counts
        if self.trim_empty:
            df.drop(df.columns[df.mean(axis=0) < 1e-3], axis=1, inplace=True)

        return df
