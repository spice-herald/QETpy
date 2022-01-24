import numpy as np
from math import ceil
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


__all__ = [
    "plot_psd",
    "plot_reim_psd",
    "plot_corrcoeff",
    "plot_csd",
    "plot_decorrelatednoise",
    "compare_noise",
    "plot_noise_sim",
]


def plot_psd(noise, lgcoverlay=True, lgcsave=False, savepath=None):
    """
    Function to plot the noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz).

    Parameters
    ----------
    noise : Object
        Noise object to be plotted
    lgcoverlay : boolean, optional
        If True, psd's for all channels are overlayed in a single plot,
        If False, each psd for each channel is plotted in a seperate subplot
    lgcsave : boolean, optional
        If True, the figure is saved in the user provided directory
    savepath : str, optional
        Absolute path for the figure to be saved

    """

    if noise.psd is None:
        print('Need to calculate the psd first')
        return
    else:
        ### Overlay plot
        if lgcoverlay:
            plt.figure(figsize=(12, 8))
            plt.title(f'{noise.name} PSD')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
            plt.grid(which='both')
            for ichan, channel in enumerate(noise.channames):
                plt.loglog(noise.freqs[1:], np.sqrt(noise.psd[ichan][1:]), label=channel)
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            if lgcsave:
                try:
                    plt.savefig(
                        savepath + noise.name.replace(" ", "_") + '_PSD_overlay.png',
                        bbox_extra_artists=(lgd, ),
                        bbox_inches='tight',
                    )
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()
        ### Subplots            
        else:
            num_subplots = len(noise.channames)
            nrows = int(ceil(num_subplots / 2))
            ncolumns = 2
            fig, axes = plt.subplots(nrows, ncolumns, figsize=(6 * num_subplots, 6 * num_subplots))
            plt.suptitle(f'{noise.name} PSD', fontsize=40)
            for ii in range(nrows * 2):
                if ii < nrows:
                    irow = ii
                    jcolumn = 0
                else:
                    irow = ii - nrows
                    jcolumn = 1
                if ii < num_subplots and nrows > 1:    
                    axes[irow,jcolumn].set_title(noise.channames[ii], fontsize=30)
                    axes[irow,jcolumn].set_xlabel('Frequency [Hz]', fontsize=25)
                    axes[irow,jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]', fontsize=25)
                    axes[irow,jcolumn].grid(which='both')
                    axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.psd[ii][1:]))
                elif ii < num_subplots and nrows==1:
                    axes[jcolumn].set_title(noise.channames[ii], fontsize=30)
                    axes[jcolumn].set_xlabel('Frequency [Hz]', fontsize=25)
                    axes[jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]', fontsize=25)
                    axes[jcolumn].grid(which = 'both')
                    axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.psd[ii][1:]))
                elif nrows==1:
                    axes[jcolumn].axis('off')
                else:
                    axes[irow,jcolumn].axis('off')
            plt.tight_layout() 
            plt.subplots_adjust(top=0.95)

            if lgcsave:
                try:
                    plt.savefig(savepath + noise.name.replace(" ", "_") + '_PSD_subplot.png')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()



def plot_reim_psd(noise, lgcsave=False, savepath=None):
    """
    Function to plot the real vs imaginary noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz).
    This is done to check for thermal muon tails making it passed the quality cuts

    Parameters
    ----------
    noise : Object
        Noise object to be plotted
    lgcsave : boolean, optional
        If True, the figure is saved in the user provided directory
    savepath : str, optional
        Absolute path for the figure to be saved

    """

    if noise.real_psd is None:
        print('Need to calculate the psd first')
        return
    else:
        num_subplots = len(noise.channames)
        nrows = int(ceil(num_subplots / 2))
        ncolumns = 2
        fig, axes = plt.subplots(nrows, ncolumns, figsize=(6 * num_subplots, 6 * num_subplots)) 
        plt.suptitle('{} Real vs Imaginary PSD'.format(noise.name), fontsize=40)
        for ii in range(nrows * 2):
            if ii < nrows:
                irow = ii
                jcolumn = 0
            else:
                irow = ii - nrows
                jcolumn = 1
            if ii < num_subplots and nrows > 1:
                axes[irow,jcolumn].set_title(noise.channames[ii], fontsize=30)
                axes[irow,jcolumn].set_xlabel('Frequency [Hz]', fontsize=25)
                axes[irow,jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]', fontsize=25)
                axes[irow,jcolumn].grid(which='both')
                axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.real_psd[ii][1:]), label='real')
                axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.imag_psd[ii][1:]), label='imag')
                axes[irow,jcolumn].legend()
            elif ii < num_subplots and nrows==1:
                axes[jcolumn].set_title(noise.channames[ii], fontsize=30)
                axes[jcolumn].set_xlabel('Frequency [Hz]', fontsize=25)
                axes[jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]', fontsize=25)
                axes[jcolumn].grid(which='both')
                axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.real_psd[ii][1:]), label='real')
                axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.imag_psd[ii][1:]), label='imag')
                axes[jcolumn].legend()
            elif nrows==1:
                axes[jcolumn].axis('off')
            else:
                axes[irow,jcolumn].axis('off')
        plt.tight_layout() 
        plt.subplots_adjust(top=0.95)

        if lgcsave:
            try:
                plt.savefig(savepath + noise.name.replace(" ", "_") + '_ReIm_PSD.png')
            except:
                print('Invalid save path. Figure not saved')
        plt.show()        



def plot_corrcoeff(noise, lgcsmooth=True, nwindow=7, lgcsave=False, savepath=None):
    """
    Function to plot the cross channel correlation coefficients. Since there are typically few traces,
    the correlations are often noisy. a savgol_filter is used to smooth out some of the noise

    Parameters
    ----------
    noise : Object
        noise object to be plotted
    lgcsmooth : boolean, optional
        If True, a savgol_filter will be used when plotting.
    nwindow : int, optional
        the number of bins used for the window in the savgol_filter
    lgcsave : boolean, optional
        If True, the figure is saved in the user provided directory
    savepath : str, optional
        Absolute path for the figure to be saved

    """

    if (noise.corrcoeff is None):
        print('Need to calculate the corrcoeff first')
        return
    else:
        plt.figure(figsize=(12, 8))
        plt.title('{} \n Cross Channel Correlation Coefficients'.format(noise.name))
        for ii in range(noise.corrcoeff.shape[0]):
            for jj in range(noise.corrcoeff.shape[1]):
                if ii > jj:
                    label = '{} - {}'.format(noise.channames[ii], noise.channames[jj])
                    if lgcsmooth:
                        plt.plot(
                            noise.freqs[1:],
                            savgol_filter(noise.corrcoeff[ii][jj][1:], nwindow, 3, mode = 'nearest'),
                            label=label,
                            alpha=0.5,
                        )
                    else:
                        plt.plot(noise.freqs[1:], noise.corrcoeff[ii][jj][1:] , label=label, alpha=0.5)
                        
                    plt.xscale('log')
        plt.xlabel('frequency [Hz]')
        plt.ylabel(r'Correlation Coeff [COV(x,y)/$\sigma_x \sigma_y$]')
        plt.grid(which = 'both')
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1.)

        if lgcsave:
            try:
                plt.savefig(
                    savepath + noise.name.replace(" ", "_") + '_corrcoeff.png',
                    bbox_extra_artists=(lgd, ),
                    bbox_inches='tight',
                )
            except:
                print('Invalid save path. Figure not saved')

        plt.show()

def plot_csd(noise, whichcsd=['01'], lgcreal=True, lgcsave=False, savepath=None):
    """
    Function to plot the cross channel noise spectrum referenced to the TES line in
    units of Amperes^2/Hz

    Parameters
    ----------
    noise : Object
        Noise object to be plotted
    whichcsd : list, optional
        a list of strings, where each element of the list refers to the pair of 
        indices of the desired csd plot
    lgcreal : boolean, optional
        If True, the Re(csd) is plotted. If False, the Im(csd) is plotted
    lgcsave : boolean, optional
        If True, the figure is saved in the user provided directory
    savepath : str, optional
        Absolute path for the figure to be saved

    """

    if noise.csd is None:
        print('Must calculate the csd first')
        return
    else:
        x_plt_label = []
        y_plt_label = []
        for label in whichcsd:
            if type(label) == str:
                x_plt_label.append(int(label[0]))
                y_plt_label.append(int(label[1]))
                if ((int(label[0]) > noise.real_csd.shape[0] - 1) or (int(label[1]) > noise.real_csd.shape[1] - 1)):
                    print('index out of range')
                    return
            else:
                print("Invalid selection. Please provide a list of strings for the desired plots. Ex: ['01','02'] ")
                return

        for ii in range(len(x_plt_label)):
            plt.figure(figsize=(12, 8))
            if lgcreal:
                title = '{} Re(CSD) for channels: {}-{}'.format(
                    noise.name,
                    noise.channames[x_plt_label[ii]],
                    noise.channames[y_plt_label[ii]],
                )

                plt.loglog(noise.freqs[1:], noise.real_csd[x_plt_label[ii]][y_plt_label[ii]][1:])
            else:
                title = '{} Im(CSD) for channels: {}-{}'.format(
                    noise.name,
                    noise.channames[x_plt_label[ii]],
                    noise.channames[y_plt_label[ii]],
                )

                plt.loglog(noise.freqs[1:], noise.imag_csd[x_plt_label[ii]][y_plt_label[ii]][1:])
            plt.title(title)
            plt.grid(which='both')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(r'CSD [A$^2$/Hz]')

            if lgcsave:
                try:
                    plt.savefig(savepath + noise.name.replace(" ", "_") + '_csd{}.png'.format(ii))
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()


def plot_decorrelatednoise(noise, lgcoverlay=False, lgcdata=True, lgcuncorrnoise=True, lgccorrelated=False,
                           lgcsum=False, lgcsave=False, savepath=None):
    """
    Function to plot the de-correlated noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz) 
    from fitted parameters calculated calculate_deCorrelated_noise

    Parameters
    ----------
    noise : Object
        Noise object to be plotted
    lgcoverlay : boolean, optional
        If True, de-correlated for all channels are overlayed in a single plot,
        If False, the noise for each channel is plotted in a seperate subplot
    lgcdata : boolean, optional
        Only applies when lgcoverlay = False. If True, the csd data is plotted
    lgcuncorrnoise : boolean, optional
        Only applies when lgcoverlay = False. If True, the de-correlated noise is plotted
    lgccorrelated : boolean, optional
        Only applies when lgcoverlay = False. If True, the correlated component of the fitted noise
        is plotted
    lgcsum : boolean, optional
        Only applies when lgcoverlay = False. If True, the sum of the fitted de-correlated noise and
        and correlated noise is plotted
    lgcsave : boolean, optional
        If True, the figure is saved in the user provided directory
    savepath : str, optional
        Absolute path for the figure to be saved

    """

    if noise.uncorrnoise is None:
        print('Need to de-correlate the noise first')
        return
    else:

        ### Overlay plot
        if lgcoverlay:
            plt.figure(figsize=(12, 8))
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
            plt.grid(which='both')
            plt.title('{} de-correlated noise'.format(noise.name))
            for ichan, channel in enumerate(noise.channames):
                    plt.loglog(noise.freqs[1:], np.sqrt(noise.uncorrnoise[ichan][1:]), label=channel)
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

            if lgcsave:
                try:
                    plt.savefig(
                        savepath + noise.name.replace(" ", "_") + '_deCorrNoise_overlay.png',
                        bbox_extra_artists=(lgd, ),
                        bbox_inches='tight',
                    ) 
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()
        ### Subplots
        else:
            num_subplots = len(noise.channames)
            nrows = int(ceil(num_subplots / 2))
            ncolumns = 2
            fig, axes = plt.subplots(nrows, ncolumns, figsize=(6 * num_subplots, 6 * num_subplots)) 
            plt.suptitle('{} de-correlated noise'.format(noise.name), fontsize=40)
            for ii in range(nrows * 2):
                if ii < nrows:
                    irow = ii
                    jcolumn = 0
                else:
                    irow = ii - nrows
                    jcolumn = 1
                if ii < num_subplots and nrows > 1:    
                    axes[irow,jcolumn].set_title(noise.channames[ii], fontsize=30)
                    axes[irow,jcolumn].set_xlabel('Frequency [Hz]', fontsize=25)
                    axes[irow,jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]', fontsize=25)
                    axes[irow,jcolumn].grid(which='both')
                    if lgcdata:
                        axes[irow,jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.real_csd[ii][ii][1:]),
                            label='Data',
                            alpha=0.4,
                        )
                    if lgcuncorrnoise:
                        axes[irow,jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.uncorrnoise[ii][1:]),
                            label='Uncorrelated Noise',alpha = 0.6)
                    if lgccorrelated:
                        axes[irow,jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.corrnoise[ii][1:]),
                            label='Correlated Noise',
                            alpha=0.6,
                        )
                    if lgcsum:
                        axes[irow,jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.uncorrnoise[ii][1:] + noise.corrnoise[ii][1:]),
                            label='Total Noise',
                            alpha=0.6,
                        )
                    axes[irow,jcolumn].legend()
                elif ii < num_subplots and nrows==1:
                    axes[jcolumn].set_title(noise.channames[ii], fontsize=30)
                    axes[jcolumn].set_xlabel('Frequency [Hz]', fontsize=25)
                    axes[jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]', fontsize=25)
                    axes[jcolumn].grid(which='both')
                    if lgcdata:
                        axes[jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.real_csd[ii][ii][1:]),
                            label='Data',
                            alpha=0.4,
                        )
                    if lgcuncorrnoise:
                        axes[jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.uncorrnoise[ii][1:]),
                            label='Uncorrelated Noise',
                            alpha=0.6,
                        )
                    if lgccorrelated:
                        axes[jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.corrnoise[ii][1:]),
                            label='Correlated Noise',
                            alpha=0.6,
                        )
                    if lgcsum:
                        axes[jcolumn].loglog(
                            noise.freqs[1:],
                            np.sqrt(noise.uncorrnoise[ii][1:] + noise.corrnoise[ii][1:]),
                            label='Total Noise',
                            alpha=0.6,
                        )
                    axes[jcolumn].legend()
                elif nrows==1:
                    axes[jcolumn].axis('off')
                else:
                    axes[irow,jcolumn].axis('off')
            plt.tight_layout() 
            plt.subplots_adjust(top=0.95)

            if lgcsave:
                try:
                    plt.savefig(savepath + noise.name.replace(" ", "_") + '_deCorrNoise_subplot.png')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()


def compare_noise(arr, channels, lgcdecorrelatednoise=False, lgcsave=False, savepath=None):
    """
    Function to plot multiple psds from different noise objects on the same figure. Each channel will
    be plotted in its own figure.

    Parameters
    ----------
    arr : array_like
        Array of noise objects
    channels : list
        List of strings, each string is a channel to plot. ex ['PSA1','PAS2']
    lgcdecorrelatednoise : boolean, optional
        If False, the psd is for each channel is plotted. If True, the calculated
        de-correlated noise is plotted
    lgcsave : boolean , optional
        If True, the figure is saved in the user provided directory
    savepath : str, optional
        Absolute path for the figure to be saved

    """


    #check to make sure channels is a list or array
    if (type(channels) == np.ndarray or type(channels) == list):
        for chan in channels:
            if chan in set(sum([arr[ii].channames for ii in range(len(arr))], [])): #check if channel is in 
                # any of the noise objects before creating a figure
                plt.figure(figsize=(12, 8))
                plt.title(chan) 
                plt.xlabel('Frequency [Hz]')
                plt.ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                plt.grid(which = 'both')
                for ii in range(len(arr)):
                    if chan in arr[ii].chann_dict:
                        chan_index = arr[ii].chann_dict[chan] #check that the channel is in the noise object before plotting
                        #plot the de correlated noise if desired
                        if lgcdecorrelatednoise:
                            #check that de correlated noise has been calculated
                            if arr[ii].uncorrnoise is not None:
                                plt.loglog(
                                    arr[ii].freqs[1:],
                                    np.sqrt(arr[ii].uncorrnoise[chan_index][1:]),
                                    label=arr[ii].name + ' Decorrelated Noise',
                                )
                            else:
                                print('The decorrelated noise for file: {} has not been calculated yet'.format(arr[ii].name))
                        else:
                            plt.loglog(arr[ii].freqs[1:], np.sqrt(arr[ii].psd[chan_index][1:]), label=arr[ii].name)
                    else:
                        print('channel: {} not found for file: {} '.format(chan, arr[ii].name)) 
                lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
                if lgcsave:
                    try:
                        plt.savefig(
                            savepath + chan + '_PSD_comparison.png',
                            bbox_extra_artists=(lgd, ),
                            bbox_inches='tight',
                        )
                    except:
                        print('Invalid save path. Figure not saved')
                plt.show()
            else:
                print('Invalid channel name: {} '.format(chan))
    else:
        print("Please provide desired channels in format of a list of numpy.ndarray. ex ['PSA1','PAS2']")   





def plot_noise_sim(f, psd, noise_sim, istype, qetbias, lgcsave=False, figsavepath='', xlims=None, ylims=None):
    """
    Plots psd with simulated noise model

    Parameters
    ----------
    f : array_like
        Frequency bins for psd
    psd : array_like 
        Power spectral density
    istype : str
        Must be 'current', 'power', 'sc', or 'normal'
        If 'current' the noise is plotted referenced to TES current
        If 'power' the noise is plotted referenced to TES power
    qetbias : float
        Applied QET bias
    lgcsave : boolean, optional
        If True, plot is saved
    figsavepath : str, optional
        Directory to save trace
    xlims : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim()

    Returns
    -------
    fig : Object
        fig object from matplotlib.pyplot
    ax : Object
        ax object from matplotlib.pyplot

    """

    if istype == 'current':
        fig, ax = _plot_ti_noise(f, psd, noise_sim, xlims, ylims)
        if lgcsave:
            fig.savefig(figsavepath+f'current_qetbias{qetbias*1e6:.3f}muA.png', bbox_inches='tight')
        else:
            return fig, ax 
    elif istype == 'power':
        fig, ax = _plot_tp_noise(f, psd, noise_sim, xlims, ylims)
        if lgcsave:
            fig.savefig(figsavepath+f'power_noise_qetbias{qetbias*1e6:.3f}muA.png', bbox_inches='tight')
        else:
            return fig, ax 
    elif istype == 'sc':
        fig, ax = _plot_sc_noise(f, psd, noise_sim, qetbias, xlims, ylims)    
        if lgcsave:
            plt.savefig(f'{figsavepath}SC_noise_qetbias{qetbias*1e6:.3f}muA.png')
        else:
            return fig, ax 
    elif istype == 'normal':
        fig, ax = _plot_n_noise(f, psd, noise_sim, qetbias, xlims, ylims)    
        if lgcsave:
            plt.savefig(f'{figsavepath}normal_noise_qetbias{qetbias*1e6:.3f}muA.png')
        else:
            return fig, ax 

def _plot_ti_noise(f, psd, noise_sim, xlims, ylims):
    """
    Helper function to plot transition noise in units of current

    Parameters
    ----------
    f : ndarray
        Array of frequency values
    psd : ndarray
        One sided Power spectral density
    noise_sim : TESnoise object
        The noise simulation object
    xlims : NoneType, tuple
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple
        Limits to be passed to ax.set_ylim()

    Returns
    -------
    fig : Object
        fig object from matplotlib.pyplot
    ax : Object
        ax object from matplotlib.pyplot

    """

    freqs = f[1:]
    psd = psd[1:]
    noise_sim.freqs = freqs

    fig, ax = plt.subplots(figsize=(11, 6))
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.grid(which="major", linestyle='--')
    ax.grid(which="minor", linestyle="dotted", alpha=0.5)
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_title(f"Current Noise For $R_0$ : {noise_sim.r0*1e3:.2f} $m\Omega$")

    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_ites())),
        color='#1f77b4',
        linewidth=1.5,
        label='TES Johnson Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_iload())),
        color='#ff7f0e',
        linewidth=1.5,
        label='Load Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_itfn())),
        color='#2ca02c',
        linewidth=1.5,
        label='TFN Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_itot())),
        color='#d62728',
        linewidth=1.5,
        label='Total Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_isquid())),
        color='#9467bd',
        linewidth=1.5,
        label='Squid+Electronics Noise',
    )
    ax.loglog(freqs, np.sqrt(psd), color='#8c564b', alpha=0.8, label='Raw Data')

    ax.set_ylabel('TES Current Noise $[A/\sqrt{\mathrm{Hz}}]$')

    lgd = plt.legend(loc='upper right')

    return fig, ax
    
    
def _plot_tp_noise(f, psd, noise_sim, xlims, ylims):
    """
    Helper function to plot transition noise in units of power

    Parameters
    ----------
    f : ndarray
        Array of frequency values
    psd : ndarray
        One sided Power spectral density
    noise_sim : TESnoise object
        The noise simulation object
    qetbias : float
        Applied QET bias
    xlims : NoneType, tuple
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple
        Limits to be passed to ax.set_ylim()

    Returns
    -------
    fig : Object
        fig object from matplotlib.pyplot
    ax : Object
        ax object from matplotlib.pyplot

    """

    freqs = f[1:]
    psd = psd[1:]
    noise_sim.freqs = freqs

    fig, ax = plt.subplots(figsize=(11, 6))
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.grid(which="major", linestyle='--')
    ax.grid(which="minor", linestyle="dotted", alpha=0.5)
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_title(f"Power Noise For $R_0$ : {noise_sim.r0*1e3:.2f} $m\Omega$")

    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_ptes())),
        color='#1f77b4',
        linewidth=1.5,
        label='TES Johnson Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_pload())),
        color='#ff7f0e',
        linewidth=1.5,
        label='Load Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_ptfn())),
        color='#2ca02c',
        linewidth=1.5,
        label='TFN Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_ptot())),
        color='#d62728',
        linewidth=1.5,
        label='Total Noise',
    )
    ax.loglog(
        noise_sim.freqs,
        np.sqrt(np.abs(noise_sim.s_psquid())),
        color='#9467bd',
        linewidth=1.5,
        label='Squid+Electronics Noise',
    )
    ax.loglog(
        freqs,
        np.sqrt(psd / (np.abs(noise_sim.dIdP(freqs))**2)),
        color='#8c564b',
        alpha=0.8,
        label='Raw Data',
    )
    ax.set_ylabel(r'Input Referenced Power Noise [W/$\sqrt{\mathrm{Hz}}$]')

    lgd = plt.legend(loc='upper right')

    return fig, ax

def _plot_sc_noise(f, psd, noise_sim, qetbias, xlims, ylims):
    """
    Helper function to plot SC noise 

    Parameters
    ----------
    f : ndarray
        Array of frequency values
    psd : ndarray
        One sided Power spectral density
    noise_sim : TESnoise object
        The noise simulation object
    qetbias : float
        Applied QET bias
    xlims : NoneType, tuple
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple
        Limits to be passed to ax.set_ylim()

    Returns
    -------
    fig : Object
        fig object from matplotlib.pyplot
    ax : Object
        ax object from matplotlib.pyplot

    """

    f = f[1:]
    psd = psd[1:]
    fig, ax = plt.subplots(figsize=(11, 6))
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.grid(which="major", linestyle = '--')
    ax.grid(which="minor", linestyle="dotted", alpha=0.5)
    ax.loglog(
        f, np.sqrt(psd), alpha=0.5, color='#8c564b', label='Raw Data',
    )
    ax.loglog(
        f, np.sqrt(noise_sim.s_isquid(f)), color='#9467bd', label='Squid+Electronics Noise',
    )
    ax.loglog(
        f, np.sqrt(noise_sim.s_iloadsc(f)), color='#ff7f0e', label='Load Noise',
    )
    ax.loglog(
        f, np.sqrt(noise_sim.s_itotsc(f)), color='#d62728', label='Total Noise',
    )
    ax.legend()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Input Referenced Current Noise [A/$\sqrt{\mathrm{Hz}}$]')
    ax.set_title(f'Superconducting State noise for QETbias: {qetbias*1e6} $\mu$A')
    ax.tick_params(which="both", direction="in", right=True, top=True)

    return fig, ax    

def _plot_n_noise(f, psd, noise_sim, qetbias, xlims, ylims):
    """
    Helper function to plot normal state noise 

    Parameters
    ----------
    f : ndarray
        Array of frequency values
    psd : ndarray
        One sided Power spectral density
    noise_sim : TESnoise object
        The noise simulation object
    qetbias : float
        Applied QET bias
    xlims : NoneType, tuple
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple
        Limits to be passed to ax.set_ylim()    

    Returns
    -------
    fig : Object
        fig object from matplotlib.pyplot
    ax : Object
        ax object from matplotlib.pyplot

    """

    f = f[1:]
    psd = psd[1:]
    fig, ax = plt.subplots(figsize=(11, 6))
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.grid(which="major", linestyle = '--')
    ax.grid(which="minor", linestyle="dotted", alpha=0.5)
    ax.loglog(
        f, np.sqrt(psd), alpha=0.5, color='#8c564b', label='Raw Data',
    )
    ax.loglog(
        f, np.sqrt(noise_sim.s_isquid(f)), color='#9467bd', label='Squid+Electronics Noise',
    )
    ax.loglog(
        f, np.sqrt(noise_sim.s_itesnormal(f)), color='#1f77b4', label= 'TES Johnson Noise',
    )
    ax.loglog(
        f, np.sqrt(noise_sim.s_iloadnormal(f)), color='#ff7f0e', label='Load Noise',
    )
    ax.loglog(
        f, np.sqrt(noise_sim.s_itotnormal(f)), color='#d62728', label='Total Noise',
    )
    ax.legend()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Input Referenced Current Noise [A/$\sqrt{\mathrm{Hz}}$]')
    ax.set_title(f'Normal State noise for QETbias: {qetbias*1e6} $\mu$A')
    ax.tick_params(which="both", direction="in", right=True, top=True)

    return fig, ax

