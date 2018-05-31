import numpy as np
import matplotlib.pyplot as plt

    def plot_full_trace(didv, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        
        if poles = "all":
            poleslist = np.array([1,2,3])
        else:
            poleslist = np.array(poles)
        
        ## plot the entire trace with fits
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(didv.time*1e6, didv.tmean*1e6 - didv.offset*1e6, color='black', label='mean')
        
        if (didv.fitparams1 is not None) and (1 in poles):
            ax.plot(didv.time*1e6+didv.fitparams1[2]*1e6, (didv.didvfit1_timedomain-didv.offset)*1e6, color='magenta', alpha=0.9, label='1-pole fit')
        if (didv.fitparams2 is not None) and (2 in poles):
            ax.plot(didv.time*1e6+didv.fitparams2[4]*1e6, (didv.didvfit2_timedomain-didv.offset)*1e6, color='green', alpha=0.9, label='2-pole fit')
        if (didv.fitparams3 is not None) and (3 in poles):
            ax.plot(didv.time*1e6+didv.fitparams3[6]*1e6, (didv.didvfit3_timedomain-didv.offset)*1e6, color='orange', alpha=0.9, label='3-pole fit')
        if (didv.irwinparams2priors is not None) and (plotpriors):
            ax.plot(didv.time*1e6+didv.irwinparams2priors[6]*1e6, (didv.didvfit2priors_timedomain-didv.offset)*1e6, color='cyan', alpha=0.9, label='2-pole fit with priors')
        
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')
        ax.set_xlim(0,max(didv.time)*1e6)
        
        ax.legend(loc='upper left')
        ax.set_title("Full Trace of dIdV")
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        
        if lgcsave:
            fig.savefig(savepath+'dIsTraces.png')
            plt.close(fig)
        else:
            plt.show()
        
    def plot_single_period_of_trace(didv, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        
        if poles = "all":
            poleslist = np.array([1,2,3])
        else:
            poleslist = np.array(poles)
        
        ## plot a single period of the trace
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(didv.time*1e6, didv.tmean*1e6 - didv.offset*1e6, color='black', label='mean')
        
        if (didv.fitparams1 is not None) and (1 in poles):
            ax.plot(didv.time*1e6+didv.fitparams1[2]*1e6, (didv.didvfit1_timedomain-didv.offset)*1e6, color='magenta', alpha=0.9, label='1-pole fit')
        if (didv.fitparams2 is not None) and (2 in poles):
            ax.plot(didv.time*1e6+didv.fitparams2[4]*1e6, (didv.didvfit2_timedomain-didv.offset)*1e6, color='green', alpha=0.9, label='2-pole fit')
        if (didv.fitparams3 is not None) and (3 in poles):
            ax.plot(didv.time*1e6+didv.fitparams3[6]*1e6, (didv.didvfit3_timedomain-didv.offset)*1e6, color='orange', alpha=0.9, label='3-pole fit')
        if (didv.irwinparams2priors is not None) and (plotpriors):
            ax.plot(didv.time*1e6+didv.irwinparams2priors[6]*1e6, (didv.didvfit2priors_timedomain-didv.offset)*1e6, color='cyan', alpha=0.9, label='2-pole fit with priors')
        
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')
        
        period = 1.0/didv.sgfreq
        halfRange=0.6*period
        ax.set_xlim((period-halfRange)*1e6,(period+halfRange)*1e6)

        ax.legend(loc='upper left')
        ax.set_title("Single Period of Trace")
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        
        if lgcsave:
            fig.savefig(savepath+'dIsTracesFit.png')
            plt.close(fig)
        else:
            plt.show()

    def plot_zoomed_in_trace(didv, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        
        if poles = "all":
            poleslist = np.array([1,2,3])
        else:
            poleslist = np.array(poles)
            
        ## plot zoomed in on the trace
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(didv.time*1e6, didv.tmean*1e6 - didv.offset*1e6, color='black', label='mean')
        
        if (didv.fitparams1 is not None) and (1 in poles):
            ax.plot(didv.time*1e6+didv.fitparams1[2]*1e6, (didv.didvfit1_timedomain-didv.offset)*1e6, color='magenta', alpha=0.9, label='1-pole fit')
        if (didv.fitparams2 is not None) and (2 in poles):
            ax.plot(didv.time*1e6+didv.fitparams2[4]*1e6, (didv.didvfit2_timedomain-didv.offset)*1e6, color='green', alpha=0.9, label='2-pole fit')
        if (didv.fitparams3 is not None) and (3 in poles):
            ax.plot(didv.time*1e6+didv.fitparams3[6]*1e6, (didv.didvfit3_timedomain-didv.offset)*1e6, color='orange', alpha=0.9, label='3-pole fit')
        if (didv.irwinparams2priors is not None) and (plotpriors):
            ax.plot(didv.time*1e6+didv.irwinparams2priors[6]*1e6, (didv.didvfit2priors_timedomain-didv.offset)*1e6, color='cyan', alpha=0.9, label='2-pole fit with priors')
            
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')

        period = 1.0/didv.sgfreq
        halfRange=0.1*period
        ax.set_xlim((period-halfRange)*1e6,(period+halfRange)*1e6)

        ax.legend(loc='upper left')
        ax.set_title("Zoomed In Portion of Trace")
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        if lgcsave:
            fig.savefig(savepath+'dIsTracesZoomFit.png')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_didv_flipped(didv, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        
        if poles = "all":
            poleslist = np.array([1,2,3])
        else:
            poleslist = np.array(poles)
        
        ## plot the traces as well as the traces flipped in order to check asymmetry
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(didv.time*1e6,(didv.tmean-didv.offset)*1e6,color='black',label='data')
        
        period = 1.0/didv.sgfreq
        time_flipped=didv.time-period/2.0
        tmean_flipped=-(didv.tmean-didv.offset)
        ax.plot(time_flipped*1e6,tmean_flipped*1e6,color='blue',label='flipped data')
        
        if (didv.fitparams1 is not None) and (1 in poles):
            ax.plot(didv.time*1e6+didv.fitparams1[2]*1e6, (didv.didvfit1_timedomain-didv.offset)*1e6, color='magenta', alpha=0.9, label='1-pole fit')
        if (didv.fitparams2 is not None) and (2 in poles):
            ax.plot(didv.time*1e6+didv.fitparams2[4]*1e6, (didv.didvfit2_timedomain-didv.offset)*1e6, color='green', alpha=0.9, label='2-pole fit')
        if (didv.fitparams3 is not None) and (3 in poles):
            ax.plot(didv.time*1e6+didv.fitparams3[6]*1e6, (didv.didvfit3_timedomain-didv.offset)*1e6, color='orange', alpha=0.9, label='3-pole fit')
        if (didv.irwinparams2priors is not None) and (plotpriors):
            ax.plot(didv.time*1e6+didv.irwinparams2priors[6]*1e6, (didv.didvfit2priors_timedomain-didv.offset)*1e6, color='cyan', alpha=0.9, label='2-pole fit with priors')
            
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')
        ax.legend(loc='upper left')
        ax.set_title("Flipped Traces to Check Asymmetry")
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        if lgcsave:
            fig.savefig(savepath+'dIsTracesFlipped.png')
            plt.close(fig)
        else:
            plt.show()
        
    def plot_re_im_didv(didv, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        
        if poles = "all":
            poleslist = np.array([1,2,3])
        else:
            poleslist = np.array(poles)
        
        goodinds=np.abs(didv.didvmean/didv.didvstd) > 2.0 ## don't plot points with huge errors
        plotinds=didv.freq[goodinds]>0
    
        freqinds = didv.freq>0
        
        ## plot the real part of the dIdV in frequency domain
        fig,ax=plt.subplots(figsize=(10,6))
        
        ax.scatter(didv.freq[freqinds],np.real(dIdV)[plotinds],color='blue',label='mean',s=5)
        ## plot error in real part of dIdV
        ax.plot(didv.freq[freqinds],np.real(dIdV+sdIdV)[plotinds],color='black',label='1-$\sigma$ bounds',alpha=0.1)
        ax.plot(didv.freq[freqinds],np.real(dIdV-sdIdV)[plotinds],color='black',alpha=0.1)
        
        if (didv.fitparams1 is not None) and (1 in poleslist):
            ax.plot(didv.freq[freqinds],np.real(didv.didvfit1)[freqinds],color='magenta',label='1-pole fit')
        if (didv.fitparams2 is not None) and (2 in poleslist):
            ax.plot(didv.freq[freqinds],np.real(didv.didvfit2)[freqinds],color='green',label='2-pole fit')
        if (didv.fitparams3 is not None) and (3 in poleslist):
            ax.plot(didv.freq[freqinds],np.real(didv.didvfit3)[freqinds],color='orange',label='3-pole fit')
        if (didv.irwinparams2priors is not None) and (plotpriors):
            ax.plot(didv.freq[freqinds],np.real(didv.didvfit2priors)[freqinds],color='cyan',label='2-pole fit with priors')
        
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Re($dI/dV$) ($\Omega^{-1}$)')
        ax.set_xscale('log')
        ax.set_xlim(min(didv.freq[freqinds]),max(didv.freq[freqinds]))
        ax.legend(loc='upper left')
        ax.set_title("Real Part of dIdV")
        ax.tick_params(right='on',top='on')
        ax.grid(which='major')
        ax.grid(which='minor',linestyle='dotted',alpha=0.3)
        
        if lgcsave:
            fig.savefig(savepath+'dIdV_Real.png')
            plt.close(fig)
        else:
            plt.show()
            
        ## plot the imaginary part of the dIdV in frequency domain
        fig,ax=plt.subplots(figsize=(10,6))
        
        ax.scatter(didv.freq[freqinds],np.imag(dIdV)[plotinds],color='blue',label='x(f) mean',s=5)
        
        ## plot error in imaginary part of dIdV
        ax.plot(didv.freq[freqinds],np.imag(dIdV+sdIdV)[plotinds],color='black',label='x(f) 1-$\sigma$ bounds',alpha=0.1)
        ax.plot(didv.freq[freqinds],np.imag(dIdV-sdIdV)[plotinds],color='black',alpha=0.1)
        
        if (didv.fitparams1 is not None) and (1 in poleslist):
            ax.plot(didv.freq[freqinds],np.imag(didv.didvfit1)[freqinds],color='magenta',label='1-pole fit')
        if (didv.fitparams2 is not None) and (2 in poleslist):
            ax.plot(didv.freq[freqinds],np.imag(didv.didvfit2)[freqinds],color='green',label='2-pole fit')
        if (didv.fitparams3 is not None) and (3 in poleslist):
            ax.plot(didv.freq[freqinds],np.imag(didv.didvfit3)[freqinds],color='orange',label='3-pole fit')
        if (didv.irwinparams2priors is not None) and (plotpriors):
            ax.plot(didv.freq[freqinds],np.imag(didv.didvfit2priors)[freqinds],color='cyan',label='2-pole fit with priors')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Im($dI/dV$) ($\Omega^{-1}$)')
        ax.set_xscale('log')
        ax.set_xlim(min(didv.freq[freqinds]),max(didv.freq[freqinds]))
        ax.legend(loc='upper left')
        ax.set_title("Imaginary Part of dIdV")
        ax.tick_params(which='both',direction='in',right='on',top='on')
        ax.grid(which='major')
        ax.grid(which='minor',linestyle='dotted',alpha=0.3)
        
        if lgcsave:
            fig.savefig(savepath+'dIdV_Imag.png')
            plt.close(fig)
        else:
            plt.show()