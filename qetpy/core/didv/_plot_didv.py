import numpy as np
import matplotlib.pyplot as plt

from ._base_didv import squarewaveresponse, complexadmittance


class _PlotDIDV(object):
    """Class that contains all plotting functions for DIDV."""

    def _get_best_time_offset(self):
        """
        Helper method for returning the time offset value that
        corresponds to the fit with the lowest chi-square.

        """

        cost_lambda = lambda x: x['cost'] if x is not None else None
        dt_lambda = lambda x: x['params']['dt'] if x is not None else None

        cost_vals = [
            cost_lambda(self._1poleresult),
            cost_lambda(self._2poleresult),
            cost_lambda(self._3poleresult),
        ]
        dt_vals = [
            dt_lambda(self._1poleresult),
            dt_lambda(self._2poleresult),
            dt_lambda(self._3poleresult),
        ]
        if all(fv is None for fv in cost_vals):
            best_time_offset = 0
        else:
            min_cost_idx = min(
                (val, ii) for ii, val in enumerate(
                    cost_vals
                ) if val is not None
            )[1]
            best_time_offset = dt_vals[min_cost_idx]

        return best_time_offset


    def _plot_time_domain(self, poles):
        """Helper function for plotting the fits in time domain."""

        if poles == "all":
            poleslist = np.array([1, 2, 3])
        else:
            poleslist = np.array(poles)

        ## plot the entire trace with fits
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            self._time * 1e6,
            (self._tmean - self._offset) * 1e6,
            color='k',
            label='Mean',
        )

        if (self._1poleresult is not None) and (1 in poleslist):
            if 'smallsignalparams' in self._1poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit1_timedomain = squarewaveresponse(
                self._time,
                self._sgamp,
                self._sgfreq,
                self._dutycycle,
                **self._1poleresult[key],
            )
            ax.plot(
                (self._time + self._1poleresult['params']['dt']) * 1e6,
                didvfit1_timedomain * 1e6,
                color='magenta',
                alpha=0.9,
                label='1-Pole Fit',
            )

        if (self._2poleresult is not None) and (2 in poleslist):
            if 'smallsignalparams' in self._2poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit2_timedomain = squarewaveresponse(
                self._time,
                self._sgamp,
                self._sgfreq,
                self._dutycycle,
                **self._2poleresult[key],
            )
            ax.plot(
                (self._time + self._2poleresult['params']['dt']) * 1e6,
                didvfit2_timedomain * 1e6,
                color='green',
                alpha=0.9,
                label='2-Pole Fit',
            )

        if (self._3poleresult is not None) and (3 in poleslist):
            if 'smallsignalparams' in self._3poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit3_timedomain = squarewaveresponse(
                self._time,
                self._sgamp,
                self._sgfreq,
                self._dutycycle,
                **self._3poleresult[key],
            )
            ax.plot(
                (self._time + self._3poleresult['params']['dt']) * 1e6,
                didvfit3_timedomain * 1e6,
                color='orange',
                alpha=0.9,
                label='3-Pole Fit',
            )

        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')
        ax.legend(loc='upper left')
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both', direction='in', right=True, top=True)

        return fig, ax


    def plot_full_trace(self, poles="all", saveplot=False, savepath="",
                        savename=""):
        """
        Function to plot the entire trace in time domain

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        saveplot : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        """

        fig, ax = self._plot_time_domain(poles)

        ax.set_xlim([self._time[0] * 1e6, self._time[-1] * 1e6])
        ax.set_title("Full Trace of dIdV")

        if saveplot:
            fig.savefig(savepath + f"full_trace_{savename}.png")
            plt.close(fig)
        else:
            plt.show()


    def plot_single_period_of_trace(self, poles="all", saveplot=False,
                                    savepath="", savename=""):
        """
        Function to plot a single period of the trace in time domain

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        saveplot : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        """

        fig, ax = self._plot_time_domain(poles)

        period = 1.0/self._sgfreq

        ax.set_xlim([self._time[0] * 1e6, self._time[0] * 1e6 + period * 1e6])
        ax.set_title("Single Period of Trace")

        if saveplot:
            fig.savefig(savepath + f"trace_one_period_{savename}.png")
            plt.close(fig)
        else:
            plt.show()


    def plot_zoomed_in_trace(self, poles="all", zoomfactor=0.1,
                             saveplot=False, savepath="", savename=""):
        """
        Function to plot a zoomed in portion of the trace in time
        domain. This plot zooms in on the overshoot of the DIDV.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        zoomfactor : float, optional, optional
            Number between zero and 1 to show different amounts of the
            zoomed in trace.
        saveplot : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        """

        period = 1.0 / self._sgfreq

        best_time_offset = self._get_best_time_offset()

        fig, ax = self._plot_time_domain(poles)

        ax.set_xlim(
            (best_time_offset + self._time[0] + (
                0.5 - zoomfactor / 2
            ) * period) * 1e6,
            (best_time_offset + self._time[0] + (
                0.5 + zoomfactor / 2) * period
            ) * 1e6,
        )

        ax.set_title("Zoomed In Portion of Trace")

        if saveplot:
            fig.savefig(savepath + f"zoomed_in_trace_{savename}.png")
            plt.close(fig)
        else:
            plt.show()


    def plot_didv_flipped(self, poles="all", saveplot=False, savepath="",
                          savename=""):
        """
        Function to plot the flipped trace in time domain. This
        function should be used to test if there are nonlinearities in
        the didv.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        saveplot : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        """

        fig, ax = self._plot_time_domain(poles)

        period = 1.0 / self._sgfreq
        time_flipped = self._time - period / 2.0
        tmean_flipped = -(self._tmean - self._offset)

        ax.plot(
            time_flipped * 1e6,
            tmean_flipped * 1e6,
            color='blue',
            label='Flipped Data',
        )

        ax.set_title("Flipped Traces to Check Asymmetry")

        if saveplot:
            fig.savefig(savepath + f"flipped_trace_{savename}.png")
            plt.close(fig)
        else:
            plt.show()


    def _plot_freq_domain(self, function, poles):
        """Helper method for plotting data in frequency domain."""
        
        if poles == "all":
            poleslist = np.array([1, 2, 3])
        else:
            poleslist = np.array(poles)

        # remove values set with placeholder mean
        goodinds = (
            self._didvmean != 0.5 - 0.5j
        ) & (
            self._didvmean != - 0.5 + 0.5j
        )
        fitinds = self._freq > 0
        plotinds = np.logical_and(fitinds, goodinds)

        best_time_offset = self._get_best_time_offset()

        time_phase = np.exp(2.0j * np.pi * best_time_offset * self._freq)

        ## plot the real part of the dIdV in frequency domain
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            self._freq[plotinds],
            function(self._didvmean * time_phase)[plotinds],
            color='blue',
            label='Mean',
            s=5,
        )
        ## plot error in real part of dIdV
        ax.plot(
            self._freq[plotinds],
            function((self._didvmean + self._didvstd) * time_phase)[plotinds],
            color='black',
            label='1-$\sigma$ Bounds',
            alpha=0.1,
        )
        ax.plot(
            self._freq[plotinds],
            function((self._didvmean - self._didvstd) * time_phase)[plotinds],
            color='black',
            alpha=0.1,
        )

        if (self._1poleresult is not None) and (1 in poleslist):
            if 'smallsignalparams' in self._1poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit1_freqdomain = complexadmittance(
                self._freq, **self._1poleresult[key],
            )
            ax.plot(
                self._freq[fitinds],
                function(didvfit1_freqdomain)[fitinds],
                color='magenta',
                label='1-Pole Fit',
            )

        if (self._2poleresult is not None) and (2 in poleslist):
            if 'smallsignalparams' in self._2poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit2_freqdomain = complexadmittance(
                self._freq, **self._2poleresult[key],
            )
            ax.plot(
                self._freq[fitinds],
                function(didvfit2_freqdomain)[fitinds],
                color='green',
                label='2-Pole Fit',
            )

        if (self._3poleresult is not None) and (3 in poleslist):
            if 'smallsignalparams' in self._3poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit3_freqdomain = complexadmittance(
                self._freq, **self._3poleresult[key],
            )
            ax.plot(
                self._freq[fitinds],
                function(didvfit3_freqdomain)[fitinds],
                color='orange',
                label='3-Pole Fit',
            )

        ax.set_xlabel('Frequency (Hz)')
        ax.set_xscale('log')

        yhigh = max(function(
            self._didvmean * time_phase
        )[plotinds][self._freq[plotinds] < 1e5])
        ylow = min(function(
            self._didvmean * time_phase
        )[plotinds][self._freq[plotinds] < 1e5])

        ybnd = np.max([yhigh, -ylow])

        ax.set_ylim([-ybnd, ybnd])
        ax.set_xlim([min(self._freq[fitinds]), max(self._freq[fitinds])])
        ax.legend(loc='upper left')
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(which='major')
        ax.grid(which='minor', linestyle='dotted', alpha=0.3)

        return fig, ax


    def plot_re_im_didv(self, poles="all", saveplot=False, savepath="",
                        savename=""):
        """
        Function to plot the real and imaginary parts of the didv in
        frequency space. Currently creates two different plots.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        saveplot : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        """

        fig, ax = self._plot_freq_domain(np.real, poles)

        ax.set_title("Real Part of dIdV")
        ax.set_ylabel('Re($dI/dV$) ($\Omega^{-1}$)')

        if saveplot:
            fig.savefig(savepath + f"didv_real_{savename}.png")
            plt.close(fig)
        else:
            plt.show()

        
        fig, ax = self._plot_freq_domain(np.imag, poles)

        ax.set_title("Imaginary Part of dIdV")
        ax.set_ylabel('Im($dI/dV$) ($\Omega^{-1}$)')

        if saveplot:
            fig.savefig(savepath + f"didv_imag_{savename}.png")
            plt.close(fig)
        else:
            plt.show()


    def plot_abs_phase_didv(self, poles="all", saveplot=False, savepath="",
                            savename=""):
        """
        Function to plot the absolute value and the phase of the dIdV
        in frequency space. Currently creates two different plots.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        saveplot : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        """

        fig, ax = self._plot_freq_domain(np.abs, poles)

        ax.set_title("|dIdV|")
        ax.set_ylabel('Abs($dI/dV$) ($\Omega^{-1}$)')
        ax.set_ylim(0)

        if saveplot:
            fig.savefig(savepath + f"didv_abs_{savename}.png")
            plt.close(fig)
        else:
            plt.show()


        fig, ax = self._plot_freq_domain(np.angle, poles)

        ax.set_title("Phase of dIdV")
        ax.set_ylabel('Arg($dI/dV$)')
        ax.set_ylim(-np.pi, np.pi)

        if saveplot:
            fig.savefig(savepath + f"didv_phase_{savename}.png")
            plt.close(fig)
        else:
            plt.show()


    def plot_re_vs_im_dvdi(self, poles='all', saveplot=False, savepath="",
                           savename=""):
        """
        Function to plot the real vs imaginary parts of the complex
        impedance.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        saveplot : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        """
        
        if poles == "all":
            poleslist = np.array([1, 2, 3])
        else:
            poleslist = np.array(poles)

        ## don't plot points with huge errors
        goodinds = np.abs(self._didvmean / self._didvstd) > 2.0
        fitinds = self._freq > 0
        plotinds = np.logical_and(fitinds, goodinds)

        best_time_offset = self._get_best_time_offset()

        time_phase = np.exp(2.0j * np.pi * best_time_offset * self._freq)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            np.real(1 / (self._didvmean * time_phase))[plotinds],
            np.imag(1 / (self._didvmean * time_phase))[plotinds],
            color='blue',
            label='Mean',
            s=5,
        )

        if (self._1poleresult is not None) and (1 in poleslist):
            if 'smallsignalparams' in self._1poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit1_freqdomain = complexadmittance(
                self._freq, **self._1poleresult[key],
            )
            ax.plot(
                np.real(1 / didvfit1_freqdomain)[fitinds],
                np.imag(1 / didvfit1_freqdomain)[fitinds],
                color='magenta',
                label='1-Pole Fit',
            )

        if (self._2poleresult is not None) and (2 in poleslist):
            if 'smallsignalparams' in self._2poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit2_freqdomain = complexadmittance(
                self._freq, **self._2poleresult[key],
            )
            ax.plot(
                np.real(1 / didvfit2_freqdomain)[fitinds],
                np.imag(1 / didvfit2_freqdomain)[fitinds],
                color='green',
                label='2-Pole Fit',
            )

        if (self._3poleresult is not None) and (3 in poleslist):
            if 'smallsignalparams' in self._3poleresult:
                key = 'smallsignalparams'
            else:
                key = 'params'
            didvfit3_freqdomain = complexadmittance(
                self._freq, **self._3poleresult[key],
            )
            ax.plot(
                np.real(1 / didvfit3_freqdomain)[fitinds],
                np.imag(1 / didvfit3_freqdomain)[fitinds],
                color='orange',
                label='3-Pole Fit',
            )

        ax.set_xlabel('Re($dV/dI$) ($\Omega$)')
        ax.set_ylabel('Im($dV/dI$) ($\Omega$)')
        ax.set_title('Re($dV/dI$) vs. Im($dV/dI$)')

        ax.legend(loc='upper left')
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(which='major')
        ax.grid(which='minor', linestyle='dotted', alpha=0.3)

        if saveplot:
            fig.savefig(savepath+f"dvdi_real_vs_imag_{savename}.png")
            plt.close(fig)
        else:
            plt.show()
