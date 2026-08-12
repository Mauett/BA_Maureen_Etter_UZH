import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
from typing import Optional


class PlottingScaler:
    """Class for plotting of CAEN V260 scaler data and related dark count results.

    Attributes:
        data: Pandas data frame with scaler data as returned by
            `pmt_analysis.utils.input.ScalerRawData.get_data` method.
        t_int: Data acquisition interval in seconds. Attribute of
            `pmt_analysis.utils.input.ScalerRawData` object.
        give_rate: If true, use count rates on vertical axis, otherwise use absolute count values.
        save_plots: Bool defining if plots are saved (as png and pdf).
        show_plots: Bool defining if plots are displayed.
        save_dir: Target directory for saving the plots.
        partition_t: List of UNIX timestamps partitioning the data, e.g. into different operation conditions,
            such as opposing PMT voltages for light emission tests.
        partition_v: Parameter values corresponding to different operation conditions set at times given in
            `partition_t`, such as opposing PMT voltage values for light emission tests.
        partition_v_unit: Unit of parameter represented by `partition_v`, e.g. `V` for the
            opposing PMT voltage values for light emission tests.
    """

    def __init__(self, data: pd.DataFrame, t_int: int, save_plots: bool = False, show_plots: bool = True,
                 save_dir: Optional[str] = None, partition_t: Optional[list] = None,
                 partition_v: Optional[list] = None, partition_v_unit: Optional[str] = None,
                 give_rate: bool = False):
        """Init of the PlottingScaler class.

        Args:
            data: Pandas data frame with scaler data as returned by
                `pmt_analysis.utils.input.ScalerRawData.get_data` method.
            t_int: Data acquisition interval in seconds. Attribute of
                `pmt_analysis.utils.input.ScalerRawData` object.
            save_plots: Bool defining if plots are saved (as png and pdf).
            show_plots: Bool defining if plots are displayed.
            save_dir: Target directory for saving the plots.
            partition_t: List of UNIX timestamps partitioning the data, e.g. into different operation conditions,
                such as opposing PMT voltages for light emission tests.
            partition_v: Parameter values corresponding to different operation conditions set at times given in
                `partition_t`, such as opposing PMT voltage values for light emission tests.
            partition_v_unit: Unit of parameter represented by `partition_v`, e.g. `'V'` for the
                opposing PMT voltage values for light emission tests.
            give_rate: If true, use count rates on vertical axis, otherwise use absolute count values.
        """
        self.data = data
        self.t_int = t_int
        self.give_rate = give_rate
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.save_dir = save_dir
        if save_plots & (save_dir is None):
            raise NameError('save_dir must be defined if save_plots is True.')
        self.partition_t = partition_t
        if self.partition_t is not None:
            self.partition_t = np.array(np.array(self.partition_t), dtype=float)
        self.partition_v = partition_v
        if self.partition_v is not None:
            if self.partition_t is None:
                warnings.warn('Ignore partition_v as partition_t is undefined.')
                self.partition_v = None
            elif np.array(self.partition_v).dtype == int:
                self.partition_v = np.array(self.partition_v)
            else:
                self.partition_v = np.array(np.array(self.partition_v), dtype=float)
        if self.partition_v is None:
            warnings.warn('Ignore partition_v_unit as partition_t or partition_v are undefined.')
            self.partition_v_unit = None
        else:
            self.partition_v_unit = partition_v_unit

    def plot_rate_evolution_hist2d(self, channel: int, t_step_s: int, time_format: str = 't_datetime_utc',
                                   values_dict: Optional[dict] = None, m_value: Optional[str] = 'median',
                                   plot_type: str = 'hist2d'):
        """Plot of time-dependent counts / count rates.

        Args:
            channel: Scaler channel number.
            t_step_s: Time bin width in seconds.
            time_format: Time axis format: Options:
                `t_s_rel` time difference in seconds since start of data period;
                `t_h_rel` time difference in hours since start of data period;
                `t_d_rel` time difference in days since start of data period;
                `t_datetime_utc` UTC datetime;
                `t_datetime_zh` datetime for Zurich time zone, i.e. CE(S)T
            values_dict: Dictionary with characteristic values in the partitions as obtained with
                `pmt_analysis.analysis.scaler.Scaler.get_values` method.
            m_value: Key of values from `values_dict` to use as characteristic values in partitions.
                options: `'median'`, `'mean'`, `'mode'`.
            plot_type: 'hist2d' for 2d histogram or 'scatter' for scatter plot.
        """
        if values_dict is not None:
            m_value = m_value.lower()
            if m_value not in ['median', 'mean', 'mode']:
                raise ValueError('Value {} for `m_value` not supported. Supported options: `median`, `mean`, `mode`')
            m_vals = np.array(values_dict[m_value])
            m_t_start = np.array(values_dict['t_start'])
            m_t_end = np.array(values_dict['t_end'])
            if np.unique([len(el) for el in [m_vals, m_t_start, m_t_end]]).shape[0] != 1:
                raise ValueError('Length for {}, t_start, and t_end not equal.'.format(m_value))
        # Convert time stamps to selected format
        if time_format == 't_s_rel':
            t_values = self.data['timestamp'] - self.data['timestamp'].min()
            if self.partition_t is not None:
                partition_t = self.partition_t - self.data['timestamp'].min()
            t_step = t_step_s
            if values_dict is not None:
                m_t_start = m_t_start - self.data['timestamp'].min()
                m_t_end = m_t_end - self.data['timestamp'].min()
        elif time_format == 't_h_rel':
            t_values = (self.data['timestamp'] - self.data['timestamp'].min()) / 3600
            if self.partition_t is not None:
                partition_t = (self.partition_t - self.data['timestamp'].min()) / 3600
            t_step = t_step_s / 3600
            if values_dict is not None:
                m_t_start = (m_t_start - self.data['timestamp'].min()) / 3600
                m_t_end = (m_t_end - self.data['timestamp'].min()) / 3600
        elif time_format == 't_d_rel':
            t_values = (self.data['timestamp'] - self.data['timestamp'].min()) / (24 * 3600)
            if self.partition_t is not None:
                partition_t = (self.partition_t - self.data['timestamp'].min()) / (24 * 3600)
            t_step = t_step_s / (24 * 3600)
            if values_dict is not None:
                m_t_start = (m_t_start - self.data['timestamp'].min()) / (24 * 3600)
                m_t_end = (m_t_end - self.data['timestamp'].min()) / (24 * 3600)
        elif time_format == 't_datetime_utc':
            t_values = pd.to_datetime(self.data['timestamp'], unit='s', utc=True)
            t_values = t_values.map(lambda x: x.tz_localize(None))  # remove timezone holding local time representations
            if self.partition_t is not None:
                partition_t = pd.to_datetime(self.partition_t, unit='s', utc=True, errors='coerce')
                partition_t = partition_t.map(lambda x: x.tz_localize(None))
            t_step = pd.Timedelta(t_step_s, 's')
            if values_dict is not None:
                m_t_start = pd.to_datetime(m_t_start, unit='s', utc=True, errors='coerce')
                m_t_start = m_t_start.map(lambda x: x.tz_localize(None))
                m_t_end = pd.to_datetime(m_t_end, unit='s', utc=True, errors='coerce')
                m_t_end = m_t_end.map(lambda x: x.tz_localize(None))
        elif time_format == 't_datetime_zh':
            t_values = pd.to_datetime(self.data['timestamp'], unit='s', utc=True)
            t_values = t_values.map(lambda x: x.tz_convert('Europe/Zurich'))
            t_values = t_values.map(lambda x: x.tz_localize(None))  # remove timezone holding local time representations
            if self.partition_t is not None:
                partition_t = pd.to_datetime(self.partition_t, unit='s', utc=True, errors='coerce')
                partition_t = partition_t.map(lambda x: x.tz_convert('Europe/Zurich'))
                partition_t = partition_t.map(lambda x: x.tz_localize(None))
            t_step = pd.Timedelta(t_step_s, 's')
            if values_dict is not None:
                m_t_start = pd.to_datetime(m_t_start, unit='s', utc=True, errors='coerce')
                m_t_start = m_t_start.map(lambda x: x.tz_convert('Europe/Zurich'))
                m_t_start = m_t_start.map(lambda x: x.tz_localize(None))
                m_t_end = pd.to_datetime(m_t_end, unit='s', utc=True, errors='coerce')
                m_t_end = m_t_end.map(lambda x: x.tz_convert('Europe/Zurich'))
                m_t_end = m_t_end.map(lambda x: x.tz_localize(None))
        else:
            raise ValueError('Value {} for time_format unsupported.'
                             'Supported values: [t_s_rel, t_h_rel, t_d_rel, t_datetime_utc, '
                             't_datetime_zh]'.format(time_format))

        if plot_type == 'hist2d':
            # Define time axis bins
            bins_x = np.arange(t_values.min(),
                               t_values.max() + t_step,
                               t_step)
            if time_format in ['t_datetime_utc', 't_datetime_zh']:
                bins_x = pd.to_datetime(bins_x)

            # Define count / count rate axis bins
            if self.give_rate:
                bins_y_05p = np.percentile(self.data['ch{}_freq'.format(channel)], 5)
                bins_y_95p = np.percentile(self.data['ch{}_freq'.format(channel)], 95)
            else:
                bins_y_05p = np.percentile(self.data['ch{}_cnts'.format(channel)], 5)
                bins_y_95p = np.percentile(self.data['ch{}_cnts'.format(channel)], 95)
            bins_y_min = bins_y_05p - (bins_y_95p - bins_y_05p)
            bins_y_max = bins_y_95p + 1.2*(bins_y_95p - bins_y_05p)
            if self.give_rate:
                bins_y = np.arange(bins_y_min, bins_y_max, 1)
            else:
                bins_y = np.arange(bins_y_min, bins_y_max, self.t_int)

            # Generate plot
            fig, ax = plt.subplots()
            if self.give_rate:
                plt.hist2d(t_values, self.data['ch{}_freq'.format(channel)],
                           bins=[bins_x, bins_y], cmap='Blues')
            else:
                plt.hist2d(t_values, self.data['ch{}_cnts'.format(channel)],
                           bins=[bins_x, bins_y], cmap='Blues')
        elif plot_type == 'scatter':
            # Generate plot
            fig, ax = plt.subplots()
            bins_x = np.array([min(t_values), max(t_values)])
            if self.give_rate:
                y_values = self.data['ch{}_freq'.format(channel)]
            else:
                y_values = self.data['ch{}_cnts'.format(channel)]
            bins_y = np.array([min(y_values) - 0.05*(max(y_values)-min(y_values)),
                               max(y_values) + 0.05*(max(y_values)-min(y_values))])
            plt.scatter(t_values, y_values, s=1, alpha = 0.5)
        else:
            raise ValueError('Plot type `{}` not supported, use `hist2d` or `scatter` instead.'.format(plot_type))

        # Mark partitions
        if self.partition_v is not None:
            annotation_style = dict(size=6, color='gray', rotation='vertical',
                                    horizontalalignment='left', verticalalignment='top')
            annotation_y = max(bins_y) - (max(bins_y) - min(bins_y))/300
        if self.partition_t is not None:
            for i, part in enumerate(partition_t):
                if not pd.isnull(part):
                    plt.axvline(x=part, color='gray', linestyle='solid', alpha=0.7)
                if (self.partition_v is not None) and (not pd.isnull(np.array(self.partition_v)[i])):
                    if pd.isnull(part):
                        annotation_x = min(bins_x)
                    else:
                        annotation_x = part
                    annotation_x += (max(bins_x) - min(bins_x))/300
                    if time_format in ['t_datetime_utc', 't_datetime_zh']:
                        annotation_x = matplotlib.dates.date2num(annotation_x)
                    if self.partition_v_unit is None:
                        partition_v_unit = ''
                    else:
                        partition_v_unit = ' {}'.format(self.partition_v_unit)
                    ax.text(annotation_x, annotation_y,
                            '{}{}'.format(self.partition_v[i], partition_v_unit),
                            **annotation_style)

        # Give characteristic value partitions
        if values_dict is not None:
            for i, m_val in enumerate(m_vals):
                plt.plot([m_t_start[i], m_t_end[i]], 2*[m_val], c='k', linewidth=1, linestyle='solid')

        # Adjust axes and labels
        if plot_type == 'scatter':
            plt.xlim(bins_x[0], bins_x[1])
            plt.ylim(bins_y[0], bins_y[1])

        if time_format == 't_s_rel':
            plt.xlabel('Time [s]')
        elif time_format == 't_h_rel':
            plt.xlabel('Time [h]')
        elif time_format == 't_d_rel':
            plt.xlabel('Time [d]')
        elif time_format == 't_datetime_utc':
            plt.xlabel('Date [UTC]')
            fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        elif time_format == 't_datetime_zh':
            plt.xlabel('Date')
            fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        if self.give_rate:
            plt.ylabel('Count Rate [s$^{-1}$]')
        else:
            if self.t_int == 1:
                plt.ylabel('Counts Per 1 Second')
            else:
                plt.ylabel('Counts Per {} Seconds'.format(self.t_int))

        if plot_type == 'hist2d':
            cbar = plt.colorbar()
            cbar.set_label('Entries')

        # Output / save
        plt.tight_layout()
        filename = 'scaler_channel_{}_'.format(channel)
        if self.give_rate:
            filename += 'rate_vs_'
        else:
            filename += 'counts_vs_'
        filename += time_format
        filename += '_'+plot_type
        filename += '_{}-{}'.format(self.data['timestamp'].min(), self.data['timestamp'].max())
        if m_value is not None:
            filename += '_'+m_value
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_dist_in_partition(self, values_dict: dict, i: int = 0):
        """Plot count (rate) distribution in partition.

        Args:
            values_dict: Dictionary with characteristic values in the partitions as obtained with
                `pmt_analysis.analysis.scaler.Scaler.get_values` method.
            i: Partition index.
        """
        plt.step(values_dict['bins_centers'][i], values_dict['cnts'][i], where='mid', color='C0')
        plt.fill_between(values_dict['bins_centers'][i],
                         values_dict['cnts'][i] - np.sqrt(values_dict['cnts'][i]),
                         values_dict['cnts'][i] + np.sqrt(values_dict['cnts'][i]),
                         step='mid', color='C0', alpha=0.33, linewidth=0
                         )
        plt.plot(values_dict['bins_centers'][i], values_dict['cnts_smoothed'][i], color='C1')
        plt.axvline(values_dict['mode'][i], color='gray', linestyle='solid', label='Mode')
        plt.axvline(values_dict['median'][i], color='gray', linestyle='dashed', label='Median')
        plt.axvline(values_dict['mean'][i], color='gray', linestyle='dotted', label='Mean')
        plt.xlim(min(values_dict['bins_centers'][i]), max(values_dict['bins_centers'][i]))
        legend = plt.legend(loc=1, bbox_to_anchor=(0.99, 0.99), frameon=True, shadow=False, edgecolor='black',
                            fancybox=False)
        legend.get_frame().set_linewidth(0.75)
        plt.ylabel('Entries')
        if self.give_rate:
            plt.xlabel('Count Rate [s$^{-1}$]')
        else:
            if self.t_int == 1:
                plt.xlabel('Counts Per 1 Second')
            else:
                plt.xlabel('Counts Per {} Seconds'.format(self.t_int))

        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_dists_in_partitions(self, values_dict: dict):
        """Plot count (rate) distributions in all partitions.

        Args:
            values_dict: Dictionary with characteristic values in the partitions as obtained with
                `pmt_analysis.analysis.scaler.Scaler.get_values` method.
        """
        if (type(self.partition_t) in [int]) or (self.partition_t is None):
            rng = 1
        else:
            rng = self.partition_t.shape[0]
        for i in range(rng):
            if self.show_plots:
                print('Plot partition {} ({:.0f} - {:.0f})'.format(i, values_dict['t_start'][i],
                                                                   values_dict['t_end'][i]))
            self.plot_dist_in_partition(values_dict, i=i)
