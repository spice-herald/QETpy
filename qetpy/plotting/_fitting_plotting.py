import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "plotnonlin",
]


def plotnonlin(OFnonlinOBJ, pulse, params, errors):
    """
    Diagnostic plotting of non-linear pulse fitting

    Parameters
    ----------
    OFnonlinOBJ: OFnonlin object
        The OFnonlin fit object to be plotted
    pulse: ndarray
        The raw trace to be fit
    params: tuple
        Tuple containing best fit paramters

    """

    if (OFnonlinOBJ.npolefit==4):
        A, B, C, tau_r,tau_f1, tau_f2, tau_f3, t0 = params
        A_err, B_err, C_err, tau_r_err, tau_f1_err, tau_f2_err, tau_f3_err, t0_err = errors
    elif (OFnonlinOBJ.npolefit==3):
        A, B, tau_r, tau_f1, tau_f2,t0 = params
        A_err, B_err, tau_r_err, tau_f1_err, tau_f2_err, t0_err = errors
    elif (OFnonlinOBJ.npolefit==2):
        A,tau_r, tau_f, t0 = params
        A_err, tau_r_err, tau_f_err, t0_err = errors
    else:
        A, tau_f, t0 = params
        A_err, tau_f_err, t0_err = errors
        tau_r = OFnonlinOBJ.taurise
        tau_r_err = 0.0

    if (OFnonlinOBJ.npolefit==4):
        variables = [A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0]
    elif (OFnonlinOBJ.npolefit==3):
        variables = [A, B, tau_r, tau_f1, tau_f2, t0]
    else:
        variables = [A, tau_r, tau_f, t0]
    ## get indices to define window ##
    t0ind = int(t0 * OFnonlinOBJ.fs) #location of timeoffset

    nmin = t0ind - int(5 * tau_r * OFnonlinOBJ.fs) # 5 risetimes before offset
    if (OFnonlinOBJ.npolefit==3 or OFnonlinOBJ.npolefit==4):
        nmax = t0ind + int(9 * tau_f1 * OFnonlinOBJ.fs) # 9 falltimes after offset
    else:
        nmax = t0ind + int(7 * tau_f * OFnonlinOBJ.fs) # 7 falltimes after offset

    nbaseline = int(OFnonlinOBJ.fs * t0) - 1000
    if nbaseline > 0:
        pulse = pulse - np.mean(pulse[:nbaseline])
    else:
        pulse = pulse - np.mean(pulse[nbaseline + 10000:])

    f = OFnonlinOBJ.freqs
    cf = f > 0
    f = f[cf]
    error = OFnonlinOBJ.error[cf]

    fig, axes = plt.subplots(2, 2, figsize = (12,8))
    if (OFnonlinOBJ.npolefit==4):
        fig.suptitle('Non-Linear Four Pole Fit', fontsize=18)
    elif (OFnonlinOBJ.npolefit==3):
        fig.suptitle('Non-Linear Three Pole Fit', fontsize=18)
    elif (OFnonlinOBJ.npolefit==2):
        fig.suptitle('Non-Linear Two Pole Fit', fontsize=18)
    elif (OFnonlinOBJ.npolefit==1):
        fig.suptitle('Non-Linear Two Pole Fit (Fixed Rise Time)', fontsize=18)

    axes[0][0].grid(linestyle='dashed')
    axes[0][0].set_title(f'Frequency Domain Trace')
    axes[0][0].set_xlabel(f'Frequency [Hz]')
    axes[0][0].set_ylabel('Amplitude [A/$\sqrt{\mathrm{Hz}}$]')
    axes[0][0].loglog(
        f, np.abs(OFnonlinOBJ.data[cf]), c='g', label='Pulse', alpha=0.75,
    )
    if (OFnonlinOBJ.npolefit==4):
        axes[0][0].loglog(
            f, np.abs(OFnonlinOBJ.fourpole(*variables))[cf], c='r', label='Fit',
        )
    elif (OFnonlinOBJ.npolefit==3):
        axes[0][0].loglog(
            f, np.abs(OFnonlinOBJ.threepole(*variables))[cf], c='r', label='Fit',
        )
    else:
        axes[0][0].loglog(
            f, np.abs(OFnonlinOBJ.twopole(*variables))[cf], c='r', label='Fit',
        )

    axes[0][0].loglog(f, error, c='b', label='$\sqrt{PSD}$', alpha=0.75)
    axes[0][0].tick_params(which='both', direction='in', right=True, top=True)

    axes[0][1].grid(linestyle = 'dashed')
    axes[0][1].set_title(f'Time Series Trace (Zoomed)')
    axes[0][1].set_xlabel(f'Time [ms]')
    axes[0][1].set_ylabel(f'Amplitude [Amps]')

    axes[0][1].plot(
        OFnonlinOBJ.time[nmin:nmax] * 1e3,
        pulse[nmin:nmax],
        c='g',
        label='Pulse',
        alpha=0.75,
    )

    if (OFnonlinOBJ.npolefit==4):
        axes[0][1].plot(
            OFnonlinOBJ.time[nmin:nmax] * 1e3,
            OFnonlinOBJ.fourpoletime(*variables)[nmin:nmax],
            c='r',
            label='Time Domain',
        )
    elif (OFnonlinOBJ.npolefit==3):
        axes[0][1].plot(
            OFnonlinOBJ.time[nmin:nmax] * 1e3,
            OFnonlinOBJ.threepoletime(*variables)[nmin:nmax],
            c='r',
            label='Time Domain',
        )
    else:
        axes[0][1].plot(
            OFnonlinOBJ.time[nmin:nmax] * 1e3,
            OFnonlinOBJ.twopoletime(*variables)[nmin:nmax],
            c='r',
            label='Time Domain',
        )
    axes[0][1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[0][1].tick_params(which='both', direction='in', right=True, top=True)


    axes[1][0].grid(linestyle='dashed')
    axes[1][0].set_title(f'Time Series Trace (Full)')
    axes[1][0].set_xlabel(f'Time [ms]')
    axes[1][0].set_ylabel(f'Amplitude [Amps]')

    axes[1][0].plot(
        OFnonlinOBJ.time * 1e3, pulse, c='g', label='Pulse', alpha=0.75,
    )

    if (OFnonlinOBJ.npolefit==4):
        axes[1][0].plot(
            OFnonlinOBJ.time * 1e3,
            OFnonlinOBJ.fourpoletime(*variables),
            c='r',
            label='Time Domain',
        )
    elif (OFnonlinOBJ.npolefit==3):
        axes[1][0].plot(
            OFnonlinOBJ.time * 1e3,
            OFnonlinOBJ.threepoletime(*variables),
            c='r',
            label='Time Domain',
        )
    else:
        axes[1][0].plot(
            OFnonlinOBJ.time * 1e3,
            OFnonlinOBJ.twopoletime(*variables),
            c='r',
            label='Time Domain',
        )
    axes[1][0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[1][0].tick_params(which='both', direction='in', right=True, top=True)

    axes[1][1].plot([], [], c='r', label='Best Fit')
    axes[1][1].plot([], [], c='g', label='Raw Data')
    axes[1][1].plot([], [], c='b', label='$\sqrt{PSD}$')

    for ii in range(len(params)):
        axes[1][1].plot([], [], linestyle=' ')

    if (OFnonlinOBJ.npolefit==4):
        labels = [
            f'A: ({A * 1e6:.4f} +\- {A_err * 1e6:.4f}) [$\mu$A]',
            f'B: ({B * 1e6:.4f} +\- {B_err * 1e6:.4f}) [$\mu$A]',
            f'C: ({C * 1e6:.4f} +\- {C_err * 1e6:.4f}) [$\mu$A]',
            f'τ, f1: ({tau_f1 * 1e6:.4f} +\- {tau_f1_err * 1e6:.4f}) [$\mu$s]',
            f'τ, f2: ({tau_f2 * 1e6:.4f} +\- {tau_f2_err * 1e6:.4f}) [$\mu$s]',
            f'τ, f3: ({tau_f3 * 1e6:.4f} +\- {tau_f3_err * 1e6:.4f}) [$\mu$s]',
            f'$t_0$: ({t0 * 1e3:.4f} +\- {t0_err * 1e3:.4f}) [ms]',
            f'τ$_r$: ({tau_r * 1e6:.4f} +\- {tau_r_err * 1e6:.4f}) [$\mu$s]',
        ]
    elif (OFnonlinOBJ.npolefit==3):
        labels = [
            f'A: ({A * 1e6:.4f} +\- {A_err * 1e6:.4f}) [$\mu$A]',
            f'B: ({B * 1e6:.4f} +\- {B_err * 1e6:.4f}) [$\mu$A]',
            f'τ, f1: ({tau_f1 * 1e6:.4f} +\- {tau_f1_err * 1e6:.4f}) [$\mu$s]',
            f'τ, f2: ({tau_f2 * 1e6:.4f} +\- {tau_f2_err * 1e6:.4f}) [$\mu$s]',
            f'$t_0$: ({t0 * 1e3:.4f} +\- {t0_err * 1e3:.4f}) [ms]',
            f'τ$_r$: ({tau_r * 1e6:.4f} +\- {tau_r_err * 1e6:.4f}) [$\mu$s]',
        ]
    else:
        labels = [
            f'A: ({A * 1e6:.4f} +\- {A_err * 1e6:.4f}) [$\mu$A]',
            f'τ$_f$: ({tau_f * 1e6:.4f} +\- {tau_f_err * 1e6:.4f}) [$\mu$s]',
            f'$t_0$: ({t0 * 1e3:.4f} +\- {t0_err * 1e3:.4f}) [ms]',
            f'τ$_r$: ({tau_r * 1e6:.4f} +\- {tau_r_err * 1e6:.4f}) [$\mu$s]',
        ]
    lines = axes[1][1].get_lines()
    legend1 = plt.legend(
        [lines[i] for i in range(3, 3 + len(params))],
        [labels[ii] for ii  in range(len(params))],
        loc=1,
    )
    legend2 = plt.legend(
        [lines[i] for i in range(0, 3)],
        ['Best Fit', 'Raw Data', '$\sqrt{PSD}$'],
        loc=2,
    )

    axes[1][1].add_artist(legend1)
    axes[1][1].add_artist(legend2)
    axes[1][1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

