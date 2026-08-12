import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional


class Scaler:
    """Class for the analysis of CAEN V260 scaler data loaded with `pmt_analysis.utils.input.ScalerRawData`.

    Attributes:
        data: Pandas data frame with scaler data as returned by
            `pmt_analysis.utils.input.ScalerRawData.get_data` method.
        t_int: Data acquisition interval in seconds. Attribute of
            `pmt_analysis.utils.input.ScalerRawData` object.
        partition_t: List of UNIX timestamps partitioning the data or integer number of seconds indicating the temporal
            width of the consecutive partitions to split the data into.
    """

    def __init__(self, data: pd.DataFrame,  t_int: int,
                 partition_t: Optional[Union[np.ndarray, pd.Series, list, int]] = None):
        """Init of the Scaler class.

        Args:
            data: Pandas data frame with scaler data as returned by
                `pmt_analysis.utils.input.ScalerRawData.get_data` method.
            t_int: Data acquisition interval in seconds. Attribute of
                `pmt_analysis.utils.input.ScalerRawData` object.
            partition_t: Array of UNIX timestamps partitioning the data or integer number of seconds indicating the
                temporal width of the consecutive partitions to split the data into. Indicate first timestamp as NaN
                to automatically infer start time of the data taking. If set to None, handle full data set as one
                partition.
        """
        self.data = data
        self.t_int = t_int
        if type(partition_t) in [list, np.ndarray, pd.Series]:
            self.partition_t = np.array(partition_t)
            # Replace first entry if NaN in case of undefined start time
            if np.isnan(self.partition_t[0]):
                self.partition_t[0] = min(self.data['timestamp'])
        elif (type(partition_t) in [int]) or (partition_t is None):
            self.partition_t = partition_t
            self.get_partitions()
        else:
            raise TypeError('Parameter `partition_t` must be of type list or int.')
        self.partition_t = np.sort(self.partition_t)

    def get_partitions(self):
        """Get partition start timestamps for partitions of width `partition_t` seconds."""
        t_min = min(self.data['timestamp'])
        t_max = max(self.data['timestamp'])
        t_diff = t_max-t_min
        if self.partition_t is None:
            # Take full data set as one partition.
            self.partition_t = np.array([t_min])
        elif self.partition_t > t_diff:
            warnings.warn('Temporal extent of `data` ({}) '
                          'smaller than `partition_t` ({}).'.format(t_diff, self.partition_t))
            # Take full data set as one partition.
            self.partition_t = np.array([t_min])
        else:
            # Find partition start timestamps. The last partition may be larger than the provided `partition_t`.
            self.partition_t = np.arange(t_diff//self.partition_t)*self.partition_t + t_min

    def get_values(self, channel: int, give_rate: bool = False, verbose: bool = True,
                   margin_start: float = 0, margin_end: float = 0) -> dict:
        """Get characteristic values (median / most probable count / count rate value, standard deviations,...)
        in partitions.

        Args:
            channel: Scaler channel number.
            give_rate: If true, use count rates in Hz, otherwise use absolute count values.
            verbose: Verbosity of the output.
            margin_start: Margin in seconds of data to exclude after start of partition.
            margin_end: Margin in seconds of data to exclude before end of partition.

        Returns:
            values_dict: Dictionary with following keys:
                `t_start` UNIX timestamp start of partition,
                `t_end` UNIX timestamp end of partition,
                `bins_centers` bin centers histogram for mode determination,
                `cnts` counts histogram for mode determination,
                `w_window` rolling average window width for mode determination,
                `cnts_smoothed` rolling average histogram for mode determination,
                `mode` mode count / count rate value from smoothed histogram,
                `median` median count / count rate value,
                `mean` mean count / count rate value,
                `perc_25` first quartile count / count rate value,
                `perc_75` third quartile count / count rate value,
                `std` standard deviation count / count rate value,
                `std_mean` standard error of the mean count / count rate value
        """
        values_dict = {key: [] for key in ['t_start', 't_end', 'bins_centers', 'cnts', 'w_window', 'cnts_smoothed',
                                           'mode', 'median', 'mean', 'perc_25', 'perc_75', 'std', 'std_mean']}
        t_last_partition = max(self.partition_t)
        if give_rate:
            value_name = 'ch{}_freq'.format(channel)
        else:
            value_name = 'ch{}_cnts'.format(channel)
        if margin_start < 0:
            margin_start = np.abs(margin_start)
            warnings.warn('Converted margin_start to positive value.')
        if margin_end < 0:
            margin_end = np.abs(margin_end)
            warnings.warn('Converted margin_end to positive value.')
        if (margin_end > 300) or (margin_start > 300):
            warnings.warn('Values of more than 300s for margin_start and margin_end are discouraged.')
        for i, t_start_partition in enumerate(self.partition_t):
            values_dict['t_start'].append(t_start_partition+margin_start)
            if t_start_partition == t_last_partition:
                t_start_next_partition = max(self.data['timestamp'])
                if margin_start+margin_end >= t_start_next_partition-t_start_partition:
                    raise ValueError('Margins are larger than partition.')
                data_sel = self.data[value_name][(self.data['timestamp'] >= t_start_partition+margin_start) &
                                                 (self.data['timestamp'] <= t_start_next_partition-margin_end)]
            else:
                t_start_next_partition = self.partition_t[i+1]
                if margin_start+margin_end >= t_start_next_partition-t_start_partition:
                    raise ValueError('Margins are larger than partition.')
                data_sel = self.data[value_name][(self.data['timestamp'] >= t_start_partition+margin_start) &
                                                 (self.data['timestamp'] < t_start_next_partition-margin_end)]
            values_dict['t_end'].append(t_start_next_partition-margin_end)

            # Get modes from smoothed data
            p_05 = np.floor(np.percentile(data_sel, 5))
            p_25 = np.percentile(data_sel, 25)
            p_75 = np.percentile(data_sel, 75)
            p_95 = np.ceil(np.percentile(data_sel, 95))
            cnts, bins_edges = np.histogram(data_sel, bins=np.arange(p_05, p_95+1/self.t_int, 1/self.t_int))
            bins_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
            values_dict['bins_centers'].append(bins_centers)
            values_dict['cnts'].append(cnts)
            # Smoothing, get window size based on ideal bin number from Sturges’ Rule
            n_bins_ideal = np.ceil(np.log2(data_sel[(data_sel >= p_25) &
                                                    (data_sel <= p_75)].shape[0]) + 1)
            w_bins_ideal = (p_75 - p_25)/n_bins_ideal
            w_window = max(3, int(5*w_bins_ideal))  # smoothing window size ~ 5 ideal bin widths
            values_dict['w_window'].append(w_window)
            cnts_smoothed = pd.Series(cnts).rolling(window=w_window, center=True, min_periods=1).mean()
            values_dict['cnts_smoothed'].append(np.array(cnts_smoothed))
            mode_naive = np.mean(bins_centers[cnts_smoothed == np.max(cnts_smoothed)])
            values_dict['mode'].append(mode_naive)

            # Other characteristic values
            median = np.median(data_sel)
            values_dict['median'].append(median)
            values_dict['mean'].append(np.mean(data_sel))
            std = np.std(data_sel)
            values_dict['std'].append(std)
            values_dict['std_mean'].append(std/np.sqrt(data_sel.shape[0]))
            values_dict['perc_25'].append(p_25)
            values_dict['perc_75'].append(p_75)

            if np.unique([len(values_dict[el]) for el in values_dict.keys()]).shape[0] != 1:
                warnings.warn('Entries in values_dict have different lengths.')
            if verbose:
                print('time: {:.0f}-{:.0f}, smoothing window: {}, mode: {}, '
                      'median: {}'.format(int(t_start_partition), int(t_start_next_partition), w_window,
                                          mode_naive, median))

        return values_dict
