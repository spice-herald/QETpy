import numpy as np
import matplotlib.pyplot as plt


__all__ = ["_plot_iv", "_plot_rv", "_plot_pv", "_plot_all_curves"]


def _plot_iv(IVobject, temps="all", chans="all", showfit=True, lgcsave=False, savepath="", savename=""):
    """
    Function to plot the IV curves for the data in an IV (IBIS) object.
    
    Parameters
    ----------
    IVobject : class
        The IV class object that the data is stored in.
    temps : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    chans : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    showfit : boolean, optional
        Boolean flag to also plot the linear fit to the normal data
    lgcsave : boolean, optional
        Boolean flag to save the plot
    savepath : string, optional
        Path to save the plot to, saves it to the current directory by default
    savename : string, optional
        Name to append to the plot file name, if saving

    """
    
    ntemps, nch, niters = IVobject.dites.shape
    chan_names = IVobject.chan_names
    
    if temps == "all":
        trange = range(ntemps)
    elif np.isscalar(temps):
        trange = np.array([temps])
    else:
        trange = np.array(temps)
        
    if chans == "all":
        chrange = range(nch)
    elif np.isscalar(chans):
        chrange = np.array([chans])
    else:
        chrange = np.array(chans)
    
    ch_colors = plt.cm.plasma(np.linspace(0, 1, num=len(trange)*len(chrange)+1))[:-1]
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    for it, t in enumerate(trange):
        for ich, ch in enumerate(chrange):
            channel_name = chan_names[ch]
            channel_name = channel_name.split(' ')[0]
            if ntemps > 1:
                label_str="Temp {}, Channel {}".format(t, chan_names[ch])
            else:
                label_str="Channel {}".format(chan_names[ch])
            ax.scatter(IVobject.vb[t, ch]*1e6, IVobject.ites[t, ch]*1e6,label=label_str ,
                        color=ch_colors[it*len(chrange) + ich], s=20.0)
            ax.plot(IVobject.vb[t, ch]*1e6, IVobject.ites[t, ch]*1e6, color=ch_colors[it*len(chrange) + ich],
                    linewidth=2.5, alpha=0.5)
            ax.errorbar(IVobject.vb[t, ch]*1e6, IVobject.ites[t, ch]*1e6, yerr=IVobject.ites_err[t, ch]*1e6,
                        xerr=IVobject.vb_err[t, ch]*1e6,
                         linestyle='None', color='k')
            if showfit:
                maxind = np.argmax(abs(IVobject.vb[t, ch]))
                
                if IVobject.vb[t, ch, maxind] > 0:
                    vbfit = np.linspace(0, max(IVobject.vb[t,ch]*1e6), num=10)
                else:
                    vbfit = np.linspace(min(IVobject.vb[t,ch]*1e6), 0, num=10)
                    
                ax.plot(vbfit, 1.0/(IVobject.rfit[t, ch])*vbfit, color=ch_colors[it*len(chrange) + ich], alpha=0.1)

            ax.legend(loc='best')
            ax.set_xlabel(r'Bias Voltage [$\mu V$]', fontsize=14)
            ax.set_ylabel(r'Current through TES [$\mu A$]', fontsize=14)
            ax.grid(linestyle='dotted')
            ax.tick_params(which='both',direction='in',right=True,top=True)
            ax.set_title(f"{channel_name}  $I_0$ vs. $V_b$", fontsize=16)
    
    if lgcsave:
        fig.savefig(savepath+"iv_curve_{}.png".format(savename))
        plt.close(fig)
    else:
        #plt.show()
        return fig, ax

def _plot_rv(IVobject, temps="all", chans="all", lgcsave=False, savepath="", savename=""):
    """
    Function to plot the resistance curves for the data in an IV object.
    
    Parameters
    ----------
    IVobject : class
        The IV class object that the data is stored in.
    temps : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    chans : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    lgcsave : boolean, optional
        Boolean flag to save the plot
    savepath : string, optional
        Path to save the plot to, saves it to the current directory by default
    savename : string, optional
        Name to append to the plot file name, if saving

    """

    ntemps, nch, niters = IVobject.r0.shape
    chan_names = IVobject.chan_names
    
    if temps == "all":
        trange = range(ntemps)
    elif np.isscalar(temps):
        trange = np.array([temps])
    else:
        trange = np.array(temps)
        
    if chans == "all":
        chrange = range(nch)
    elif np.isscalar(chans):
        chrange = np.array([chans])
    else:
        chrange = np.array(chans)
    
    ch_colors = plt.cm.plasma(np.linspace(0, 1, num=len(trange)*len(chrange)+1))[:-1]
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    for it, t in enumerate(trange):
        for ich, ch in enumerate(chrange):
            
            channel_name = chan_names[ch]
            channel_name = channel_name.split(' ')[0]
            
            if ntemps > 1:
                label_str="Temp {}, Channel {}".format(t,chan_names[ch])
            else:
                label_str="Channel {}".format(chan_names[ch])
            ax.scatter(IVobject.vb[t, ch]*1e6, IVobject.r0[t, ch]*1e3, label=label_str,
                        color=ch_colors[it*len(chrange) + ich], s=20.0)
            ax.plot(IVobject.vb[t, ch]*1e6, IVobject.r0[t, ch]*1e3, color=ch_colors[it*len(chrange) + ich], 
                    linewidth=2.5, alpha=0.5)
            ax.errorbar(IVobject.vb[t, ch]*1e6, IVobject.r0[t, ch]*1e3, yerr=IVobject.r0_err[t, ch]*1e3,
                        xerr=IVobject.vb_err[t, ch]*1e6,
                         linestyle='None', color='k')

            ax.legend(loc='best')
            ax.set_xlabel(r'Bias Voltage [$\mu V$]', fontsize=14)
            ax.set_ylabel(r'Resistance of TES [$m \Omega$]', fontsize=14)
            ax.grid(linestyle='dotted')
            ax.tick_params(which='both',direction='in',right=True,top=True)
            ax.set_title(f"{channel_name}  $R_0$ vs. $V_b$", fontsize=16)
    
    if lgcsave:
        fig.savefig(savepath+"rv_curve_{}.png".format(savename))
        plt.close(fig)
    else:
        #plt.show()
        return fig, ax    

def _plot_pv(IVobject,  temps="all", chans="all",
             percent_rn_range=None, 
             lgcsave=False, savepath="", savename=""):
    """
    Function to plot the power curves for the data in an IV object.
    
    Parameters
    ----------
    IVobject : class
        The IV class object that the data is stored in.
    temps : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    chans : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    lgcsave : boolean, optional
        Boolean flag to save the plot
    savepath : string, optional
        Path to save the plot to, saves it to the current directory by default
    savename : string, optional
        Name to append to the plot file name, if saving

    """

    ntemps, nch, niters = IVobject.ptes.shape
    chan_names = IVobject.chan_names
    
    if temps == "all":
        trange = range(ntemps)
    elif np.isscalar(temps):
        trange = np.array([temps])
    else:
        trange = np.array(temps)
        
    if chans == "all":
        chrange = range(nch)
    elif np.isscalar(chans):
        chrange = np.array([chans])
    else:
        chrange = np.array(chans)
    
    ch_colors = plt.cm.plasma(np.linspace(0, 1, num=len(trange)*len(chrange)+1))[:-1]
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    for it, t in enumerate(trange):
        for ich, ch in enumerate(chrange):

            channel_name = chan_names[ch]
            channel_name = channel_name.split(' ')[0]
            
            if ntemps > 1:
                label_str="Temp {}, Channel {}".format(t,chan_names[ch])
            else:
                label_str="Channel {}".format(chan_names[ch])
                
            v0 = IVobject.vb[t, ch]*1e6
            v0_err = IVobject.vb_err[t, ch]*1e6
            p0 = IVobject.ptes[t, ch]*1e12
            p0_err = IVobject.ptes_err[t, ch]*1e12

            unit_label = 'pW'
            if np.median(p0) < 1:
                p0 = p0*1e3
                p0_err = p0_err*1e3
                unit_label = 'fW'
                       
            # percent  rn
            rn = IVobject.rnorm[t, ch]
            pc = 100*IVobject.r0[t, ch]/rn
            if percent_rn_range is not None:

                if len( percent_rn_range) != 2:
                    raise ValueError(
                        'ERROR: "percent_rn_range" should be '
                        'a tuple!')
                
                mask = ((pc >= percent_rn_range[0])
                        & (pc <= percent_rn_range[1]))

                v0 = v0[mask]
                v0_err = v0_err[mask]
                p0 = p0[mask]
                p0_err = p0_err[mask]
                pc =  pc[mask]

            if percent_rn_range is None:
                
                ax.scatter(v0, p0, label=label_str,
                           color=ch_colors[it*len(chrange) + ich],
                           s=20.0)
                ax.plot(v0, p0, color=ch_colors[it*len(chrange) + ich], 
                        linewidth=2.5, alpha=0.5)
                ax.errorbar(v0, p0, yerr=p0_err, xerr=v0_err,
                            linestyle='None', color='k')

                ax.legend(loc='best')
                ax.set_xlabel(r'Bias Voltage [$\mu V$]', fontsize=14)
                ax.set_ylabel(f'Power [{unit_label}]', fontsize=14)
                ax.grid(linestyle='dotted')
                ax.tick_params(which='both',direction='in',right=True,
                               top=True)
                ax.set_title(f"{channel_name}  $P_0$ vs. $V_b$",
                             fontsize=16)

            else:

                ax.scatter(pc, p0, label=label_str,
                           color=ch_colors[it*len(chrange) + ich],
                           s=20.0)
                ax.plot(pc, p0, color=ch_colors[it*len(chrange) + ich], 
                        linewidth=2.5, alpha=0.5)
                ax.errorbar(pc, p0, yerr=p0_err, xerr=0,
                            linestyle='None', color='k')

                ax.legend(loc='best')
                ax.set_xlabel(r'Percent Rn', fontsize=14)
                ax.set_ylabel(f'Power [{unit_label}]', fontsize=14)
                ax.grid(linestyle='dotted')
                ax.tick_params(which='both',direction='in',right=True,
                               top=True)
                ax.set_title(f"{channel_name}  $P_0$ vs. %$Rn$ (zoom)",
                             fontsize=16)
                       
    if lgcsave:
        fig.savefig(savepath+"pv_curve_{}.png".format(savename))
        plt.close(fig)
    else:
        #plt.show()
        return fig, ax

        
def _plot_all_curves(IVobject,  temps="all", chans="all", showfit=True, lgcsave=False, savepath="", savename=""):
    """
    Function to plot the IV, resistance, and power curves for the data in an IV object.
    
    Parameters
    ----------
    IVobject : class
        The IV class object that the data is stored in.
    temps : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    chans : string, array_like, int, optional
        Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
        to a subset of bath temperatures, or just one
    showfit : boolean, optional
        Boolean flag to also plot the linear fit to the normal data
    lgcsave : boolean, optional
        Boolean flag to save the plot
    savepath : string, optional
        Path to save the plot to, saves it to the current directory by default
    savename : string, optional
        Name to append to the plot file name, if saving

    """
    
    _plot_iv(IVobject,  temps=temps, chans=chans, showfit=showfit, lgcsave=lgcsave, savepath=savepath, savename=savename)
    _plot_rv(IVobject,  temps=temps, chans=chans, lgcsave=lgcsave, savepath=savepath, savename=savename)
    _plot_pv(IVobject,  temps=temps, chans=chans, lgcsave=lgcsave, savepath=savepath, savename=savename)
    _plot_pv(IVobject,  temps=temps, chans=chans, percent_rn_range=(3,40),
             lgcsave=lgcsave, savepath=savepath, savename=savename)

  
