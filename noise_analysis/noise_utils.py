"""
A collection of utility functions to be used with the noise class, mostly functions to plot noise objects. 

Created by Caleb Fink 5/9/2018

"""
import numpy as np
from math import ceil
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')





def plot_PSD(noise, lgc_overlay = True, lgcSave = False, savePath = None):
    '''
    Function to plot the noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz)

    Input parameters:
    lgc_overlay: boolian value. If True, PSD's for all channels are overlayed in a single plot, 
                 if False, each PSD for each channel is plotted in a seperate subplot
    lgcSave: boolian value. If True, the figure is saved in the user provided directory
    savePath: absolute path for the figure to be saved

    Returns:
    None
    '''
    if noise.PSD is None:
        print('Need to calculate the PSD first')
        return
    else:
        ### Overlay plot
        if lgc_overlay:
            sns.set_context('notebook')
            plt.figure()
            plt.title('{} PSD'.format(noise.name))
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
            plt.grid(which = 'both')
            for ichan, channel in enumerate(noise.channNames):
                plt.loglog(noise.freqs[1:], np.sqrt(noise.PSD[ichan][1:]), label = channel)
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            if lgcSave:
                try:
                    plt.savefig(savePath+noise.name.replace(" ", "_")+'_PSD_overlay.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()
        ### Subplots            
        else:
            sns.set_context('poster', font_scale = 1.9)
            num_subplots = len(noise.channNames)
            nRows = int(ceil(num_subplots/2))
            nColumns = 2
            fig, axes = plt.subplots(nRows, nColumns, figsize = (6*num_subplots,6*num_subplots)) 
            plt.suptitle('{} PSD'.format(noise.name),  fontsize=40)
            for ii in range(nRows*2):
                if ii < nRows:
                    iRow = ii
                    jColumn = 0
                else:
                    iRow = ii - nRows
                    jColumn = 1
                if ii < num_subplots:    
                    axes[iRow,jColumn].set_title(noise.channNames[ii])
                    axes[iRow,jColumn].set_xlabel('frequency [Hz]')
                    axes[iRow,jColumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                    axes[iRow,jColumn].grid(which = 'both')
                    axes[iRow,jColumn].loglog(noise.freqs[1:], np.sqrt(noise.PSD[ii][1:]))
                else:
                    axes[iRow,jColumn].axis('off')
            plt.tight_layout() 
            plt.subplots_adjust(top=0.95)
            
            if lgcSave:
                try:
                    plt.savefig(savePath+noise.name.replace(" ", "_")+'_PSD_subplot.png')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()

                
def plot_corrCoeff(noise, lgcSave = False, savePath = None):
    '''
    Function to plot the cross channel correlation coefficients. Since there are typically few traces,
    the correlations are often noisy. a savgol_filter is used to smooth out some of the noise

    Input parameters:
    lgcSave: boolian value. If True, the figure is saved in the user provided directory
    savePath: absolute path for the figure to be saved

    Returns:
    None
    '''
    if (noise.corrCoeff is None):
        print('Need to calculate the corrCoeff first')
        return
    else:
        sns.set_context('notebook')
        plt.figure()
        plt.title('{} \n Cross Channel Correlation Coefficients'.format(noise.name) )
        for ii in range(noise.corrCoeff.shape[0]):
            for jj in range(noise.corrCoeff.shape[1]):
                if ii > jj:
                    label = '{} - {}'.format(noise.channNames[ii],noise.channNames[jj])
                    plt.plot(noise.freqs, savgol_filter(noise.corrCoeff[ii][jj], 51,3), label = label, alpha = .5)
                    plt.xscale('log')
        plt.xlabel('frequency [Hz]')
        plt.ylabel(r'Correlation Coeff [COV(x,y)/$\sigma_x \sigma_y$]')
        plt.grid(which = 'both')
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1.)
        
        if lgcSave:
            try:
                plt.savefig(savePath+noise.name.replace(" ", "_")+'_corrCoeff.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
            except:
                print('Invalid save path. Figure not saved')
        
        plt.show()


def plot_CSD(noise, whichCSD = ['01'],lgcReal = True,lgcSave = False, savePath = None):
    '''
    Function to plot the cross channel noise spectrum referenced to the 
    TES line in units of Amperes^2/Hz

    Input parameters:
    whichCSD: a list of strings, where each element of the list refers to the pair of indices of 
            the desired CSD plot
    lgcReal: boolian value. If Ture, the Re(CSD) is plotted. If False, the Im(CSD) is plotted
    lgcSave: boolian value. If True, the figure is saved in the user provided directory
    savePath: absolute path for the figure to be saved

    Returns:
    None
    '''
    if ((noise.CSD is None) or (noise.freqs_CSD is None)):
        print('Must calculate the CSD first')
        return
    else:
        sns.set_context('notebook')
        x_plt_label = []
        y_plt_label = []
        for label in whichCSD:
            if type(label) == str:
                x_plt_label.append(int(label[0]))
                y_plt_label.append(int(label[1]))
                if ((int(label[0]) > noise.real_CSD.shape[0]-1) or (int(label[1]) > noise.real_CSD.shape[1]-1)):
                    print('index out of range')
                    return
            else:
                print("Invalid selection. Please provide a list of strings for the desired plots. Ex: ['01','02'] ")
                return

        for ii in range(len(x_plt_label)):
            plt.figure()
            if lgcReal:
                title = '{} Re(CSD) for channels: {}-{}'.format(noise.name,noise.channNames[x_plt_label[ii]] \
                                                              ,noise.channNames[y_plt_label[ii]])

                plt.loglog(noise.freqs_CSD[1:],noise.real_CSD[x_plt_label[ii]][y_plt_label[ii]][1:])
            else:
                title = '{} Im(CSD) for channels: {}-{}'.format(noise.name,noise.channNames[x_plt_label[ii]] \
                                                              ,noise.channNames[y_plt_label[ii]])

                plt.loglog(noise.freqs_CSD[1:],noise.imag_CSD[x_plt_label[ii]][y_plt_label[ii]][1:])
            plt.title(title)
            plt.grid(which = 'both')
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'CSD [A$^2$/Hz]')
            
            if lgcSave:
                try:
                    plt.savefig(savePath + title.replace(" ", "_"))
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()

def plot_deCorrelatedNoise(noise, lgc_overlay = False, lgcData = True,lgcUnCorrNoise = True, lgcCorrelated = False \
                               , lgcSum = False,lgcSave = False, savePath = None):
    '''
    Function to plot the de-correlated noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz) 
    from fitted parameters calculated calculate_deCorrelated_noise

    Input parameters:
    lgc_overlay: boolian value. If True, de-correlated for all channels are overlayed in a single plot, 
                 if False, the noise for each channel is plotted in a seperate subplot
    lgcData: boolian value. Only applies when lgc_overlay = False. If True, the CSD data is plotted
    lgcUnCorrNoise: boolian value. Only applies when lgc_overlay = False. If True, the de-correlated noise is plotted
    lgcCorrelated: boolian value. Only applies when lgc_overlay = False. If True, the correlated component of the fitted noise 
                    is plotted
    lgcSum: boolian value. Only applies when lgc_overlay = False. If True, the sum of the fitted de-correlated noise and
            and correlated noise is plotted
    lgcSave: boolian value. If True, the figure is saved in the user provided directory
    savePath: absolute path for the figure to be saved

    Returns:
    None
    '''  
    if noise.unCorrNoise is None:
        print('Need to de-correlate the noise first')
        return
    else:
    
        ### Overlay plot
        if lgc_overlay:
            sns.set_context('notebook')
            plt.figure()
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
            plt.grid(which = 'both')
            plt.title('{} de-correlated noise'.format(noise.name))
            for ichan, channel in enumerate(noise.channNames):
                    plt.loglog(noise.freqs_fit[1:], np.sqrt(noise.unCorrNoise[ichan][1:]), label = channel)
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            if lgcSave:
                try:
                    plt.savefig(savePath+noise.name.replace(" ", "_")+'_deCorrNoise_overlay.png',bbox_extra_artists=(lgd,), bbox_inches='tight') 
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()
        ### Subplots
        else:
            sns.set_context('poster', font_scale = 1.9)
            num_subplots = len(noise.channNames)
            nRows = int(ceil(num_subplots/2))
            nColumns = 2
            fig, axes = plt.subplots(nRows, nColumns, figsize = (6*num_subplots,6*num_subplots)) 
            plt.suptitle('{} de-correlated noise'.format(noise.name),  fontsize=40)
            for ii in range(nRows*2):
                if ii < nRows:
                    iRow = ii
                    jColumn = 0
                else:
                    iRow = ii - nRows
                    jColumn = 1
                if ii < num_subplots:    
                    axes[iRow,jColumn].set_title(noise.channNames[ii])
                    axes[iRow,jColumn].set_xlabel('frequency [Hz]')
                    axes[iRow,jColumn].set_ylabel(r'Input Referenced Noise [A/$\sqrt{\mathrm{Hz}}$]')
                    axes[iRow,jColumn].grid(which = 'both')
                    if lgcData:
                        axes[iRow,jColumn].loglog(noise.freqs_CSD[1:], np.sqrt(noise.real_CSD[ii][ii][1:]) \
                                                  , label = 'data' ,alpha = 0.4)
                    if lgcUnCorrNoise:
                        axes[iRow,jColumn].loglog(noise.freqs_fit[1:], np.sqrt(noise.unCorrNoise[ii][1:]) \
                                                  , label = 'uncorrelated noise',alpha = 0.6)
                    if lgcCorrelated:
                        axes[iRow,jColumn].loglog(noise.freqs_fit[1:], np.sqrt(noise.corrNoise[ii][1:]) \
                                                  , label = 'correlated noise' ,alpha = 0.6)
                    if lgcSum:
                        axes[iRow,jColumn].loglog(noise.freqs_fit[1:], np.sqrt(noise.unCorrNoise[ii][1:]+noise.corrNoise[ii][1:]) \
                                   , label = 'total noise' ,alpha = 0.6)
                    axes[iRow][jColumn].legend()
                else:
                    axes[iRow,jColumn].axis('off')
            plt.tight_layout() 
            plt.subplots_adjust(top=0.95)
            
            if lgcSave:
                try:
                    plt.savefig(savePath+noise.name.replace(" ", "_")+'_deCorrNoise_subplot.png')
                except:
                    print('Invalid save path. Figure not saved')
            plt.show()

def fill_negatives(arr):
    '''
    Simple helper function to remove negative and zero values from PSD's.
    Input:
        arr: 1d array
    Returns:
        arr: arr with the negative and zero values replace by interpelate values
    '''
    zeros = np.array(arr <= 0)
    inds_zero = np.where(zeros)[0]
    inds_not_zero = np.where(~zeros)[0]
    good_vals = arr[~zeros]       
    if len(good_vals) != 0:
        arr[zeros] = np.interp(inds_zero, inds_not_zero, good_vals)  
    return arr
    