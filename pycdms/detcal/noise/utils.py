"""
A collection of utility functions to be used with the noise class, mostly functions to plot noise objects. 

Created by Caleb Fink 5/9/2018

"""
import numpy as np
import pickle 

from math import ceil
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns

def plot_psd(noise, lgcoverlay = True, lgcsave = False, savepath = None):
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
            sns.set_style('white')
            sns.set_context('notebook')
            plt.figure(figsize = (12,8))
            plt.title('{} PSD'.format(noise.name))
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
            plt.grid(which = 'both')
            for ichan, channel in enumerate(noise.channames):
                plt.loglog(noise.freqs[1:], np.sqrt(noise.psd[ichan][1:]), label = channel)
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            if lgcsave:
                try:
                    plt.savefig(savepath+noise.name.replace(" ", "_")+'_PSD_overlay.png',
                                bbox_extra_artists=(lgd,), bbox_inches='tight')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()
        ### Subplots            
        else:
            sns.set_style('white')
            sns.set_context('poster', font_scale = 1.9)
            num_subplots = len(noise.channames)
            nrows = int(ceil(num_subplots/2))
            ncolumns = 2
            fig, axes = plt.subplots(nrows, ncolumns, figsize = (6*num_subplots,6*num_subplots)) 
            plt.suptitle('{} PSD'.format(noise.name),  fontsize=40)
            for ii in range(nrows*2):
                if ii < nrows:
                    irow = ii
                    jcolumn = 0
                else:
                    irow = ii - nrows
                    jcolumn = 1
                if ii < num_subplots and nrows > 1:    
                    axes[irow,jcolumn].set_title(noise.channames[ii])
                    axes[irow,jcolumn].set_xlabel('frequency [Hz]')
                    axes[irow,jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                    axes[irow,jcolumn].grid(which = 'both')
                    axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.psd[ii][1:]))
                elif ii < num_subplots and nrows==1:
                    axes[jcolumn].set_title(noise.channames[ii])
                    axes[jcolumn].set_xlabel('frequency [Hz]')
                    axes[jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
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
                    plt.savefig(savepath+noise.name.replace(" ", "_")+'_PSD_subplot.png')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()

            
            
def plot_reim_psd(noise, lgcsave = False, savepath = None):
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
        sns.set_style('white')
        sns.set_context('poster', font_scale = 1.9)
        num_subplots = len(noise.channames)
        nrows = int(ceil(num_subplots/2))
        ncolumns = 2
        fig, axes = plt.subplots(nrows, ncolumns, figsize = (6*num_subplots,6*num_subplots)) 
        plt.suptitle('{} Real vs Imaginary PSD'.format(noise.name),  fontsize=40)
        for ii in range(nrows*2):
            if ii < nrows:
                irow = ii
                jcolumn = 0
            else:
                irow = ii - nrows
                jcolumn = 1
            if ii < num_subplots and nrows > 1:    
                axes[irow,jcolumn].set_title(noise.channames[ii])
                axes[irow,jcolumn].set_xlabel('frequency [Hz]')
                axes[irow,jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                axes[irow,jcolumn].grid(which = 'both')
                axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.real_psd[ii][1:]), label = 'real')
                axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.imag_psd[ii][1:]), label = 'imag')
                axes[irow,jcolumn].legend()
            elif ii < num_subplots and nrows==1:
                axes[jcolumn].set_title(noise.channames[ii])
                axes[jcolumn].set_xlabel('frequency [Hz]')
                axes[jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                axes[jcolumn].grid(which = 'both')
                axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.real_psd[ii][1:]), label = 'real')
                axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.imag_psd[ii][1:]), label = 'imag')
                axes[jcolumn].legend()
            elif nrows==1:
                axes[jcolumn].axis('off')
            else:
                axes[irow,jcolumn].axis('off')
        plt.tight_layout() 
        plt.subplots_adjust(top=0.95)

        if lgcsave:
            try:
                plt.savefig(savepath+noise.name.replace(" ", "_")+'_ReIm_PSD.png')
            except:
                print('Invalid save path. Figure not saved')
        plt.show()        
            
            
                
def plot_corrcoeff(noise, lgcsmooth = True, nwindow = 7, lgcsave = False, savepath = None):
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
        sns.set_style('white')
        sns.set_context('notebook')
        plt.figure(figsize = (12,8))
        plt.title('{} \n Cross Channel Correlation Coefficients'.format(noise.name) )
        for ii in range(noise.corrcoeff.shape[0]):
            for jj in range(noise.corrcoeff.shape[1]):
                if ii > jj:
                    label = '{} - {}'.format(noise.channames[ii],noise.channames[jj])
                    if lgcsmooth:
                        plt.plot(noise.freqs[1:], savgol_filter(noise.corrcoeff[ii][jj][1:], nwindow, 3, mode = 'nearest'),
                                 label = label, alpha = .5)
                    else:
                        plt.plot(noise.freqs[1:], noise.corrcoeff[ii][jj][1:] , label = label, alpha = .5)
                        
                    plt.xscale('log')
        plt.xlabel('frequency [Hz]')
        plt.ylabel(r'Correlation Coeff [COV(x,y)/$\sigma_x \sigma_y$]')
        plt.grid(which = 'both')
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1.)
        
        if lgcsave:
            try:
                plt.savefig(savepath+noise.name.replace(" ", "_")+'_corrcoeff.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
            except:
                print('Invalid save path. Figure not saved')
        
        plt.show()

def plot_csd(noise, whichcsd = ['01'], lgcreal = True, lgcsave = False, savepath = None):
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
        sns.set_style('white')
        sns.set_context('notebook')
        x_plt_label = []
        y_plt_label = []
        for label in whichcsd:
            if type(label) == str:
                x_plt_label.append(int(label[0]))
                y_plt_label.append(int(label[1]))
                if ((int(label[0]) > noise.real_csd.shape[0]-1) or (int(label[1]) > noise.real_csd.shape[1]-1)):
                    print('index out of range')
                    return
            else:
                print("Invalid selection. Please provide a list of strings for the desired plots. Ex: ['01','02'] ")
                return

        for ii in range(len(x_plt_label)):
            plt.figure(figsize = (12,8))
            if lgcreal:
                title = '{} Re(CSD) for channels: {}-{}'.format(noise.name, noise.channames[x_plt_label[ii]],
                                                                noise.channames[y_plt_label[ii]])

                plt.loglog(noise.freqs[1:],noise.real_csd[x_plt_label[ii]][y_plt_label[ii]][1:])
            else:
                title = '{} Im(CSD) for channels: {}-{}'.format(noise.name, noise.channames[x_plt_label[ii]],
                                                                noise.channames[y_plt_label[ii]])

                plt.loglog(noise.freqs[1:],noise.imag_csd[x_plt_label[ii]][y_plt_label[ii]][1:])
            plt.title(title)
            plt.grid(True, which = 'both')
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'CSD [A$^2$/Hz]')
            
            if lgcsave:
                try:
                    plt.savefig(savepath+noise.name.replace(" ", "_")+'_csd{}.png'.format(ii))
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()

def plot_decorrelatednoise(noise, lgcoverlay = False, lgcdata = True, lgcuncorrnoise = True, lgccorrelated = False,
                           lgcsum = False,lgcsave = False, savepath = None):
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
            sns.set_style('white')
            sns.set_context('notebook')
            plt.figure(figsize = (12,8))
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
            plt.grid(which = 'both')
            plt.title('{} de-correlated noise'.format(noise.name))
            for ichan, channel in enumerate(noise.channames):
                    plt.loglog(noise.freqs[1:], np.sqrt(noise.uncorrnoise[ichan][1:]), label = channel)
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            if lgcsave:
                try:
                    plt.savefig(savepath+noise.name.replace(" ", "_")+'_deCorrNoise_overlay.png', 
                                bbox_extra_artists=(lgd,), bbox_inches='tight') 
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()
        ### Subplots
        else:
            sns.set_style('white')
            sns.set_context('poster', font_scale = 1.9)
            num_subplots = len(noise.channames)
            nrows = int(ceil(num_subplots/2))
            ncolumns = 2
            fig, axes = plt.subplots(nrows, ncolumns, figsize = (6*num_subplots,6*num_subplots)) 
            plt.suptitle('{} de-correlated noise'.format(noise.name),  fontsize=40)
            for ii in range(nrows*2):
                if ii < nrows:
                    irow = ii
                    jcolumn = 0
                else:
                    irow = ii - nrows
                    jcolumn = 1
                if ii < num_subplots and nrows > 1:    
                    axes[irow,jcolumn].set_title(noise.channames[ii])
                    axes[irow,jcolumn].set_xlabel('frequency [Hz]')
                    axes[irow,jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                    axes[irow,jcolumn].grid(which = 'both')
                    if lgcdata:
                        axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.real_csd[ii][ii][1:]) \
                                                  , label = 'data' ,alpha = 0.4)
                    if lgcuncorrnoise:
                        axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.uncorrnoise[ii][1:]) \
                                                  , label = 'uncorrelated noise',alpha = 0.6)
                    if lgccorrelated:
                        axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.corrnoise[ii][1:]) \
                                                  , label = 'correlated noise' ,alpha = 0.6)
                    if lgcsum:
                        axes[irow,jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.uncorrnoise[ii][1:]+noise.corrnoise[ii][1:]) \
                                   , label = 'total noise' ,alpha = 0.6)
                    axes[irow,jcolumn].legend()
                elif ii < num_subplots and nrows==1:
                    axes[jcolumn].set_title(noise.channames[ii])
                    axes[jcolumn].set_xlabel('frequency [Hz]')
                    axes[jcolumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                    axes[jcolumn].grid(which = 'both')
                    if lgcdata:
                        axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.real_csd[ii][ii][1:]) \
                                                  , label = 'data' ,alpha = 0.4)
                    if lgcuncorrnoise:
                        axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.uncorrnoise[ii][1:]) \
                                                  , label = 'uncorrelated noise',alpha = 0.6)
                    if lgccorrelated:
                        axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.corrnoise[ii][1:]) \
                                                  , label = 'correlated noise' ,alpha = 0.6)
                    if lgcsum:
                        axes[jcolumn].loglog(noise.freqs[1:], np.sqrt(noise.uncorrnoise[ii][1:]+noise.corrnoise[ii][1:]) \
                                   , label = 'total noise' ,alpha = 0.6)
                    axes[jcolumn].legend()
                elif nrows==1:
                    axes[jcolumn].axis('off')
                else:
                    axes[irow,jcolumn].axis('off')
            plt.tight_layout() 
            plt.subplots_adjust(top=0.95)
            
            if lgcsave:
                try:
                    plt.savefig(savepath+noise.name.replace(" ", "_")+'_deCorrNoise_subplot.png')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()
            
            
def compare_noise(arr, channels, lgcdecorrelatednoise = False, lgcsave = False, savepath = None):
    """
    Function to plot multiple psd's from different noise objects on the same figure. Each channel will
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
    sns.set_style('white')
    sns.set_context('notebook')
    
    #check to make sure channels is a list or array
    if (type(channels) == np.ndarray or type(channels) == list):
        for chan in channels:
            if chan in set(sum([arr[ii].channames for ii in range(len(arr))],[])): #check if channel is in 
                # any of the noise objects before creating a figure
                plt.figure(figsize = (12,8))
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
                                plt.loglog(arr[ii].freqs[1:], np.sqrt(arr[ii].uncorrnoise[chan_index][1:]) \
                                           , label = arr[ii].name+' de-correlated noise')
                            else:
                                print('The de-correlated noise for file: {} has not been calculated yet'.format(arr[ii].name))
                        else:
                            plt.loglog(arr[ii].freqs[1:], np.sqrt(arr[ii].psd[chan_index][1:]), label = arr[ii].name)
                    else:
                        print('channel: {} not found for file: {} '.format(chan, arr[ii].name)) 
                lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                if lgcsave:
                    try:
                        plt.savefig(savepath+chan+'_PSD_comparison.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
                    except:
                        print('Invalid save path. Figure not saved')
                plt.show()
            else:
                print('Invalid channel name: {} '.format(chan))
    else:
        print("Please provide desired channels in format of a list of numpy.ndarray. ex ['PSA1','PAS2']")   

def fill_negatives(arr):
    """
    Simple helper function to remove negative and zero values from psd's.
    
    Parameters
    ----------
        arr : ndarray 
            1d array
    Returns
    -------
        arr : ndarray
            arr with the negative and zero values replaced by interpolated values
    """
    zeros = np.array(arr <= 0)
    inds_zero = np.where(zeros)[0]
    inds_not_zero = np.where(~zeros)[0]
    good_vals = arr[~zeros]       
    if len(good_vals) != 0:
        arr[zeros] = np.interp(inds_zero, inds_not_zero, good_vals)  
    return arr



def load_noise(file_str):
    """
    Load noise object that has been previously saved as pickle file
    
    Parameters
    ----------
        file_str : str
            The full path to the file to be loaded.
            
    Returns
    -------
        f : Object
            The loaded noise object.
    """
    with open(file_str,'rb') as savefile:
        f = pickle.load(savefile)
    return f
    
    
    
    
def plot_noise_sim(f, psd, noise_sim, istype, figsize = (12,8),lgcsave=False, savepath = ''):
    """
    Plots psd with simulated noise model
    
    Parameters
    ----------
        f : array_like
            Frequency bins for psd
        psd : array_like 
            Power spectral density
        istype : str
            Must be 'current' or 'power'
            If 'current' the noise is plotted referenced to TES current
            If 'power' the noise is plotted referenced to TES power
        figsize : tuple, optional
            Desired size of figure
        lgcsave : boolean, optional
            If True, plot is saved
        savepath : str, optional
            Directory to save trace
    Returns
    -------
        fig : Object
            fig object from matplotlib.pyplot
        ax : Object
            ax object from matplotlib.pyplot
    """
    
    freqs = f[1:]
    psd = psd[1:]
    
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"{istype} noise for $R_0$ : {noise_sim.r0*1e3:.0f} $m\Omega$")
    ax.grid(True, which = 'both')
    ax.set_xlabel(r'Frequency [Hz]')
    
    if istype is 'current':
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_ites())), label=r'$\sqrt{S_{ITES}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_iload())), label=r'$\sqrt{S_{ILoad}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_itfn())), label=r'$\sqrt{S_{ITFN}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_itot())), label=r'$\sqrt{S_{Itot}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_isquid())), label=r'$\sqrt{S_{Isquid}}$')
        ax.loglog(freqs, np.sqrt(psd), label ='data')
    
    elif istype is 'power':
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_ptes())), label=r'$\sqrt{S_{PTES}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_pload())), label=r'$\sqrt{S_{PLoad}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_ptfn())), label=r'$\sqrt{S_{PTFN}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_ptot())), label=r'$\sqrt{S_{Ptot}}$')
        ax.loglog(noise_sim.freqs, np.sqrt(np.abs(noise_sim.s_psquid())), label=r'$\sqrt{S_{Psquid}}$')
        ax.loglog(freqs, np.sqrt(psd/(np.abs(noise_sim.dIdP(freqs))**2)), label ='data')
        ax.set_ylabel(r'Input Referenced Power Noise [W/$\sqrt{\mathrm{Hz}}$]')
        
        
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if lgcsave:
        plt.savefig(savepath+f'{istype}_noise_{noise_sim.R0:.0f}.png', bbox_extra_artists=(lgd,), 
                    bbox_inches='tight')
    else:
        plt.show()
        return fig, ax
            