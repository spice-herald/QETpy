import numpy as np
import matplotlib.pyplot as plt
from qetpy.utils import lowpassfilter


__all__ = [
    "plotnsmb",
]


def plotnsmb(pulset, fs, tdelmin, amin, sbTemplatef, nS, nB, nt, psddnu,
             lpFiltFreq=None, lgcsaveplots=False, xlim=None, figPrefix='testFit',
             background_templates_shifts=None, savepath='./'):
    """
    Diagnostic plotting function for the OF nSmB code.

    Parameters
    ----------
    pulset : tuple
        Data to be plotted
        Dimensions: 1 X (time bins)
    omega : tuple
        Angular frequency of the pulse in the frequency domain
        Dimensions: (time bins) X ()
    fs : tuple
        Sampling frequency in Hz
    tdelmin : tuple
        The best fit time delay of the signal, offset from the original signal template
    amin : tuple
        The best fit amplitude of the signal
        Dimensions: (nS + nB) X 1
    sbTemplatef : ndarray
        The frequency domain signal and background templates
        Dimensions: (nS + nB) X (time bins)
    nt : tuple
        The number of time domain points of the signal and template
    lpFiltFreq : tuple, optional
        The frequency of the LF filter to pulset (default None)
    lgcsaveplots : bool, int, optional
        Flag for whether or not to save plots (default False)
        If not False, should be int to append to saved fig filename
    figPrefix : str, optional
        The saved fig filename prefix (whatever text before the number)
    background_templates_shifts : ndarray, optional
        The indices at which the background templates start
        Dimensions: m X ()
    savepath : str, optional
        Path to save directory. Default is current working directory.

    """

    nSB = nS + nB

    if (lpFiltFreq!=None):
        pulseFilt = lowpassfilter(pulset, lpFiltFreq, fs)

    # === DAQ Setup ===
    dt = float(1) / fs
    dnu = float(1) / (nt * dt)
    nu = np.arange(0, float(nt)) * dnu
    lgc = nu > nt * dnu / 2
    nu[lgc] = nu[lgc] - nt * dnu
    omega = (2 * np.pi) * nu

    # create a phase shift matrix
    # The signal gets phase shifted by tdelmin
    # The background templates have no phase shift
    phase = np.exp(-1j * omega * tdelmin)
    phaseAr = np.ones((nS, 1)) @ phase[None,:]
    phaseMat= np.concatenate((phaseAr, np.ones((nB, nt))), axis=0)
    ampMat = amin @ np.ones((1, nt))
    fitf = ampMat * sbTemplatef * phaseMat
    fittotf = np.sum(fitf, axis=0, keepdims=True)

    # get baseline of pulset
    pulsetBL = np.mean(pulset[0,0:300])

    # now invert
    fitt = np.real(np.fft.ifft(fitf, axis=1) * nt)
    fittott = np.real(np.fft.ifft(fittotf, axis=1) * nt);

    # make residual 
    residT = pulset - fittott
    # check the chi2
    residTf = np.fft.fft(residT, axis=1) / nt
    chi2T = np.real(np.sum(np.conj(residTf.T) / psddnu.T * residTf.T, 0))


    chi2TFloat = float(chi2T)

    # ===Time Domain ==================================================
    bins = np.arange(nt)
    bins = bins[None,:]
    timeP = bins * dt
    plt.figure(figsize=(12, 7));
    if (lpFiltFreq):
        plt.plot(bins.T, pulseFilt.T, '-k', label='Data (LP filtered)')
        plt.plot(bins.T, pulset.T, '-b', alpha=0.4, linewidth=0.5, label='Data')
    else:
        plt.plot(bins.T, pulset.T, '-k', label='Data')
    plt.plot(bins.T, fittott.T, '-g', label=f'Total Fit. $\chi^2$={chi2TFloat:.1f}', linewidth=5)    
    # plot the background template best fit
    # note:
    #    1) the loop starts at nS and only goes to nSB - 2
    #       thereby leaving out the DC and slope component
    #    2) the DC and slope components are added to the
    #       pulse background templates
    for jSB in range(nS, nSB - 2):
        if(jSB==nS):
            plt.plot(
                bins.T,
                fitt[jSB, :, None] + fitt[-1, :, None] + fitt[-2, :, None],
                '-',
                c='r',
                label='Background Fit',
            )
        else:
            plt.plot(
                bins.T,
                fitt[jSB, :, None] + fitt[-1, :, None] + fitt[-2, :, None],
                '-',
                c='r',
            )
    # plot the signal template fit if the amplitude
    # is larger than a small threshold
    if (np.abs(amin[0]) > 1e-20):
        plt.plot(
            bins.T,
            fitt[0, :, None]  + fitt[-1, :, None] + fitt[-2, :, None],
            '-c',
            label='Signal Fit',
        )

    if background_templates_shifts is not None:
            for ii in range(nB):
                if (ii==0):
                    plt.axvline(
                        x=background_templates_shifts[ii],
                        linestyle='--',
                        color='m',
                        linewidth=1,
                        label='Bckg. Start Times',
                    )
                else:
                    plt.axvline(
                        x=background_templates_shifts[ii],
                        linestyle='--',
                        color='m',
                        linewidth=1,
                    )

    plt.xlabel(f'bins (1 bin = {(dt * 1e6):.2f} $\mu$s)');
    plt.ylabel('Amps');

    if xlim is not None: 
        plt.xlim(xlim)
    plt.legend()
    plt.grid()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    if lgcsaveplots:
        plt.savefig(savepath + figPrefix + str(lgcsaveplots) + '.png', bbox_inches='tight')
