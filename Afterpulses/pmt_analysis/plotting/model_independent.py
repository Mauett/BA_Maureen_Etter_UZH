import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Optional


class PlottingGainModelIndependent:
    """
    Class for plots related to the model independent occupancy and gain calculation.

    Attributes:
        input_dict: Dictionary with output results from the
            `pmt_analysis.analysis.model_independent.GainModelIndependent.compute` method.
        save_plots: Bool defining if plots are saved (as png and pdf).
        show_plots: Bool defining if plots are displayed.
        save_dir: Target directory for saving the plots.
        save_name_suffix: Measurement specific figure name suffix.
    """

    def __init__(self, input_dict: dict, save_plots: bool = False, show_plots: bool = True,
                 save_dir: Optional[str] = None, save_name_suffix: Optional[str] = None):
        """Init of the GainModelIndependent class.

        Generation of standard plots with respect to the model independent occupancy
        and gain calculation.

        Args:
            input_dict: Dictionary with output results from the
                `pmt_analysis.analysis.model_independent.GainModelIndependent.compute` method.
            save_plots: Bool defining if plots are saved (as png and pdf).
            show_plots: Bool defining if plots are displayed.
            save_dir: Target directory for saving the plots.
            save_name_suffix: Measurement specific figure name suffix.
        """
        self.input_dict = input_dict
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.save_dir = save_dir
        self.save_name_suffix = save_name_suffix
        if save_plots & (save_dir is None):
            raise NameError('save_dir must be defined if save_plots is True.')
        if save_plots & (save_name_suffix is None):
            raise NameError('save_name_suffix must be defined if save_plots is True.')

    def plot_occupancy_vs_threshold(self):
        """Plot occupancy as a function of integration threshold."""
        thresholds_list = np.asarray(self.input_dict['iterations']['threshold'])
        occupancy_list = np.asarray(self.input_dict['iterations']['occupancy'])
        occupancy_err_list = np.asarray(self.input_dict['iterations']['occupancy_err'])
        occupancy_list_smooth = np.asarray(self.input_dict['iterations']['occupancy_smoothed'])
        occupancy = self.input_dict['occupancy']
        threshold_occ_det = self.input_dict['threshold_occupancy_determination']

        plt.plot(thresholds_list, occupancy_list, color='C0')
        plt.fill_between(thresholds_list, occupancy_list - occupancy_err_list,
                         occupancy_list + occupancy_err_list,
                         alpha=0.3, color='C0', linewidth=0)
        plt.plot(thresholds_list[occupancy_list_smooth > 0.1],
                 occupancy_list_smooth[occupancy_list_smooth > 0.1],
                 color='C1', linewidth=1)
        plt.axhline(y=occupancy, color='gray', linestyle='dotted')
        plt.axvline(x=threshold_occ_det, color='gray', linestyle='dotted')
        plt.xlim(thresholds_list[0], thresholds_list[-1])
        plt.xlabel('Area Threshold [ADC Units]')
        plt.ylabel('Occupancy [PE/Trigger]')
        filename = 'occupancy_vs_threshold_{}'.format(self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_rel_occ_err_vs_threshold(self):
        """Plot relative occupancy error as a function of integration threshold."""
        thresholds_list = np.asarray(self.input_dict['iterations']['threshold'])
        occupancy_list = np.asarray(self.input_dict['iterations']['occupancy'])
        occupancy_err_list = np.asarray(self.input_dict['iterations']['occupancy_err'])
        occupancy = self.input_dict['occupancy']
        occupancy_err = self.input_dict['occupancy_err']
        threshold_occ_det = self.input_dict['threshold_occupancy_determination']

        plt.plot(thresholds_list, occupancy_err_list / occupancy_list, linewidth=1)
        plt.axhline(y=0.01, color='gray', linestyle='dashed')
        plt.axhline(y=occupancy_err / occupancy, color='gray', linestyle='dotted')
        plt.axvline(x=threshold_occ_det, color='gray', linestyle='dotted')
        plt.xlim(thresholds_list[0], thresholds_list[-1])
        plt.yscale('log')
        plt.xlabel('Area Threshold [ADC Units]')
        plt.ylabel('Relative Error Occupancy')
        filename = 'rel_occ_err_vs_threshold_{}'.format(self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_occ_and_rel_err_vs_threshold(self):
        """Plot occupancy and relative occupancy error as a function of integration threshold
        in two vertically stacked subplots."""
        thresholds_list = np.asarray(self.input_dict['iterations']['threshold'])
        occupancy_list = np.asarray(self.input_dict['iterations']['occupancy'])
        occupancy_err_list = np.asarray(self.input_dict['iterations']['occupancy_err'])
        occupancy_list_smooth = np.asarray(self.input_dict['iterations']['occupancy_smoothed'])
        occupancy = self.input_dict['occupancy']
        occupancy_err = self.input_dict['occupancy_err']
        threshold_occ_det = self.input_dict['threshold_occupancy_determination']

        height = matplotlib.rcParams['figure.figsize'][-1]
        gs = gridspec.GridSpec(2, 1, height_ratios=[0.65 * height, 0.35 * height])

        ax0 = plt.subplot(gs[0])
        ax0.plot(thresholds_list, occupancy_list)
        ax0.fill_between(thresholds_list, occupancy_list - occupancy_err_list, occupancy_list + occupancy_err_list,
                         alpha=0.3, color='C0', linewidth=0)
        ax0.plot(thresholds_list[occupancy_list_smooth > 0.1],
                 occupancy_list_smooth[occupancy_list_smooth > 0.1], linewidth=1)
        ax0.axhline(y=occupancy, color='gray', linestyle='dotted')
        ax0.axvline(x=threshold_occ_det, color='gray', linestyle='dotted')
        ax0.set_ylabel('Occupancy [PE/Trigger]')
        ax0.tick_params(axis='x', labelbottom=False)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.plot(thresholds_list, occupancy_err_list / occupancy_list, linewidth=1)
        ax1.axhline(y=0.01, color='gray', linestyle='dashed')
        ax1.axhline(y=occupancy_err / occupancy, color='gray', linestyle='dotted')
        ax1.axvline(x=threshold_occ_det, color='gray', linestyle='dotted')
        ax1.set_xlabel('Area Threshold [ADC Units]')
        ax1.set_ylabel('Rel. Error')
        ax1.set_yscale('log')

        plt.xlim(thresholds_list[0], thresholds_list[-1])
        plt.subplots_adjust(hspace=.0)

        filename = 'occ_and_rel_err_vs_threshold_{}'.format(self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_area_hist_occ_threshold_det(self):
        """Area histogram illustrating the threshold for the occupancy determination
        and resulting 'LED on' and 'LED off' integrals."""
        hist_bin_centers = np.asarray(self.input_dict['histograms']['bin_centers'])
        hist_counts_led_on = np.asarray(self.input_dict['histograms']['counts_led_on'])
        hist_counts_led_off = np.asarray(self.input_dict['histograms']['counts_led_off'])
        threshold_occ_det = self.input_dict['threshold_occupancy_determination']
        outlier_threshold_lower = self.input_dict['outlier_thresholds'][0]

        plt.fill_between(hist_bin_centers,
                         hist_counts_led_off - (np.sqrt(hist_counts_led_off)),
                         hist_counts_led_off + (np.sqrt(hist_counts_led_off)),
                         step='mid', color='C0', alpha=0.4, linewidth=0)
        plt.fill_between(hist_bin_centers,
                         hist_counts_led_on - (np.sqrt(hist_counts_led_on)),
                         hist_counts_led_on + (np.sqrt(hist_counts_led_on)),
                         step='mid', color='C1', alpha=0.4, linewidth=0)
        mask = hist_bin_centers <= threshold_occ_det + 1
        plt.fill_between(hist_bin_centers[mask], hist_counts_led_off[mask],
                         step='mid', hatch=5 * '/', color='none',
                         edgecolor='C0', alpha=0.3)
        plt.fill_between(hist_bin_centers[mask], hist_counts_led_on[mask],
                         step='mid', hatch=5 * '\\', color='none',
                         edgecolor='C1', alpha=0.3)
        plt.step(hist_bin_centers, hist_counts_led_off, where='mid',
                 color='C0', label='LED Off')
        plt.step(hist_bin_centers, hist_counts_led_on, where='mid',
                 color='C1', label='LED On')
        plt.axvline(x=threshold_occ_det, color='gray', linestyle='dashed',
                    label='Threshold Occupancy\nCalculation: {}'.format(threshold_occ_det))
        plt.axvline(x=outlier_threshold_lower / 2, color='gray', linestyle='dotted',
                    label='Range Probed\nThresholds')
        plt.axvline(x=-outlier_threshold_lower / 2, color='gray', linestyle='dotted')
        plt.xlim(outlier_threshold_lower, -outlier_threshold_lower)
        plt.yscale('log')
        plt.xlabel('Area [ADC Units]')
        plt.ylabel('Entries')
        legend = plt.legend(loc=4, bbox_to_anchor=(0.99, 0.025),
                            framealpha=1)
        legend.get_frame().set_linewidth(matplotlib.rcParams['axes.linewidth'])
        filename = 'area_hist_occ_threshold_det_{}'.format(self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_area_hist_model_independent_gain(self):
        """Area histogram of the 'LED on' and 'LED off' data illustrating the
        model independent gain determination."""
        hist_bin_centers = np.asarray(self.input_dict['histograms']['bin_centers'])
        hist_counts_led_on = np.asarray(self.input_dict['histograms']['counts_led_on'])
        hist_counts_led_off = np.asarray(self.input_dict['histograms']['counts_led_off'])
        mean_s = self.input_dict['moments_s']['mean']
        mean_b = self.input_dict['moments_b']['mean']
        mean_psi = self.input_dict['mean_psi']

        plt.fill_between(hist_bin_centers,
                         hist_counts_led_off - (np.sqrt(hist_counts_led_off)),
                         hist_counts_led_off + (np.sqrt(hist_counts_led_off)),
                         step='mid', color='C0', alpha=0.4, linewidth=0)
        plt.fill_between(hist_bin_centers,
                         hist_counts_led_on - (np.sqrt(hist_counts_led_on)),
                         hist_counts_led_on + (np.sqrt(hist_counts_led_on)),
                         step='mid', color='C1', alpha=0.4, linewidth=0)
        plt.step(hist_bin_centers, hist_counts_led_off, where='mid',
                 color='C0', label='LED Off')
        plt.step(hist_bin_centers, hist_counts_led_on, where='mid',
                 color='C1', label='LED On')
        plt.axvline(x=mean_b, color='C0', linestyle='dotted',
                    linewidth=1, label='Mean LED Off')
        plt.axvline(x=mean_s, color='C1', linestyle='dotted',
                    linewidth=1, label='Mean LED On')
        plt.axvline(x=mean_psi, color='gray', linestyle='dashed',
                    linewidth=1, label='Mean Single\nPhotoelectron\nResponse')
        plt.xlim(min(hist_bin_centers), max(hist_bin_centers))
        plt.yscale('log')
        plt.xlabel('Area [ADC Units]')
        plt.ylabel('Entries')
        legend = plt.legend(loc=1, bbox_to_anchor=(0.99, 0.99))
        legend.get_frame().set_linewidth(matplotlib.rcParams['axes.linewidth'])
        filename = 'area_hist_model_independent_gain_{}'.format(self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_essentials(self):
        """Plot essential plots for model independent gain and occupancy estimation."""
        self.plot_occ_and_rel_err_vs_threshold()
        self.plot_area_hist_occ_threshold_det()
        self.plot_area_hist_model_independent_gain()
