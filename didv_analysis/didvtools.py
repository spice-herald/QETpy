import numpy as np
from numpy import pi
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import didvutils


def stdcomplex(x, axis=0):
    """Function to return complex standard deviation (individually computed for real and imaginary components) for an array of complex values.
    
    Args:
        x: An array of complex values from which we want the complex standard deviation.
        axis: Which axis to take the standard deviation of (should be used if the dimension of the array is greater than 1)
        
    Returns:
        std_complex: The complex standard deviation of the inputted array, along the specified axis.
    """
    rstd = np.std(x.real, axis=axis)
    istd = np.std(x.imag, axis=axis)
    std_complex = rstd+1.0j*istd
    return std_complex
    
def onepoleimpedance(freq, A, tau2):
    """Function to calculate the impedance (dVdI) of a TES with the 1-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), superconducting: A=Rl, normal: A = Rl+Rn
        tau2: The fit parameter tau2 in the complex impedance (in s), superconducting: tau2=L/Rl, normal: tau2=L/(Rl+Rn)
        
    Returns:
        dVdI: The complex impedance of the TES with the 1-pole fit
    
    """
    
    dVdI = (A*(1.0+2.0j*pi*freq*tau2))
    return dVdI

def onepoleadmittance(freq, A, tau2):
    """Function to calculate the admittance (dIdV) of a TES with the 1-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), superconducting: A=Rl, normal: A = Rl+Rn
        tau2: The fit parameter tau2 in the complex impedance (in s), superconducting: tau2=L/Rl, normal: tau2=L/(Rl+Rn)
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 1-pole fit
    
    """
    
    dVdI = onepoleimpedance(freq, A, tau2)
    return (1.0/dVdI)

def twopoleimpedance(freq, A, B, tau1, tau2):
    """Function to calculate the impedance (dVdI) of a TES with the 2-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), A = Rl + R0*(1+beta)
        B: The fit parameter B in the complex impedance (in Ohms), B = R0*l*(2+beta)/(1-l) (where l is Irwin's loop gain)
        tau1: The fit parameter tau1 in the complex impedance (in s), tau1=tau0/(1-l)
        tau2: The fit parameter tau2 in the complex impedance (in s), tau2=L/(Rl+R0*(1+beta))
        
    Returns:
        dVdI: The complex impedance of the TES with the 2-pole fit
    
    """
    
    dVdI = (A*(1.0+2.0j*pi*freq*tau2))+(B/(1.0+2.0j*pi*freq*tau1))
    return dVdI

def twopoleadmittance(freq, A, B, tau1, tau2):
    """Function to calculate the admittance (dIdV) of a TES with the 2-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), A = Rl + R0*(1+beta)
        B: The fit parameter B in the complex impedance (in Ohms), B = R0*l*(2+beta)/(1-l) (where l is Irwin's loop gain)
        tau1: The fit parameter tau1 in the complex impedance (in s), tau1=tau0/(1-l)
        tau2: The fit parameter tau2 in the complex impedance (in s), tau2=L/(Rl+R0*(1+beta))
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 2-pole fit
    
    """
    
    dVdI = twopoleimpedance(freq, A, B, tau1, tau2)
    return (1.0/dVdI)

def threepoleimpedance(freq, A, B, C, tau1, tau2, tau3):
    """Function to calculate the impedance (dVdI) of a TES with the 3-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms)
        B: The fit parameter B in the complex impedance (in Ohms)
        C: The fit parameter C in the complex impedance
        tau1: The fit parameter tau1 in the complex impedance (in s)
        tau2: The fit parameter tau2 in the complex impedance (in s)
        tau3: The fit parameter tau3 in the complex impedance (in s)
        
    Returns:
        dVdI: The complex impedance of the TES with the 3-pole fit
    
    """
    
    dVdI = (A*(1.0+2.0j*pi*freq*tau2))+(B/(1.0+2.0j*pi*freq*tau1-C/(1.0+2.0j*pi*freq*tau3)))
    return dVdI

def threepoleadmittance(freq, A, B, C, tau1, tau2, tau3):
    """Function to calculate the admittance (dIdV) of a TES with the 3-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms)
        B: The fit parameter B in the complex impedance (in Ohms)
        C: The fit parameter C in the complex impedance
        tau1: The fit parameter tau1 in the complex impedance (in s)
        tau2: The fit parameter tau2 in the complex impedance (in s)
        tau3: The fit parameter tau3 in the complex impedance (in s)
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 3-pole fit
    
    """
    
    dVdI = threepoleimpedance(freq, A, B, C, tau1, tau2, tau3)
    return (1.0/dVdI)

def twopoleimpedancepriors(freq, Rl, R0, beta, l, L, tau0):
    """Function to calculate the impedance (dVdI) of a TES with the 2-pole fit from Irwin's TES parameters
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        Rl: The load resistance of the TES (in Ohms)
        R0: The resistance of the TES (in Ohms)
        beta: The current sensitivity of the TES
        l: Irwin's loop gain
        L: The inductance in the TES circuit (in Henrys)
        tau0: The thermal time constant of the TES (in s)
        
    Returns:
        dVdI: The complex impedance of the TES with the 2-pole fit from Irwin's TES parameters
    
    """
    
    dVdI = Rl + R0*(1.0+beta) + 2.0j*pi*freq*L + R0 * l * (2.0+beta)/(1.0-l) * 1.0/(1.0+2.0j*freq*pi*tau0/(1.0-l))
    return dVdI

def twopoleadmittancepriors(freq, Rl, R0, beta, l, L, tau0):
    """Function to calculate the admittance (dIdV) of a TES with the 2-pole fit from Irwin's TES parameters
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        Rl: The load resistance of the TES (in Ohms)
        R0: The resistance of the TES (in Ohms)
        beta: The current sensitivity of the TES, beta=d(log R)/d(log I)
        l: Irwin's loop gain, l = P0*alpha/(G*Tc)
        L: The inductance in the TES circuit (in Henrys)
        tau0: The thermal time constant of the TES (in s), tau0=C/G
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 2-pole fit from Irwin's TES parameters
    
    """
    
    dVdI = twopoleimpedancepriors(freq, Rl, R0, beta, l, L, tau0)
    return (1.0/dVdI)


def convolvedidv(x, A, B, C, tau1, tau2, tau3, sgamp, Rsh, sgfreq, dutycycle):
    """Function to convert the fitted TES parameters for the complex impedance to a TES response to a square wave jitter in time domain.
    
    Args:
        x: Time values for the trace (in s)
        A: The fit parameter A in the complex impedance (in Ohms)
        B: The fit parameter B in the complex impedance (in Ohms)
        C: The fit parameter C in the complex impedance
        tau1: The fit parameter tau1 in the complex impedance (in s)
        tau2: The fit parameter tau2 in the complex impedance (in s)
        tau3: The fit parameter tau3 in the complex impedance (in s)
        sgamp: The peak-to-peak size of the square wave jitter (in Amps)
        Rsh: The shunt resistance of the TES electronics (in Ohms)
        sgfreq: The frequency of the square wave jitter (in Hz)
        dutycycle: The duty cycle of the square wave jitter (between 0 and 1)
        
    Returns:
        np.real(St): The response of a TES to a square wave jitter in time domain with the given fit parameters. The real part is taken in order to ensure that the trace is real
    
    
    """
    
    tracelength = len(x)
    
    # get the frequencies for a DFT, based on the sample rate of the data
    dx = x[1]-x[0]
    freq = fftfreq(len(x), d=dx)
    
    # dIdV of fit in frequency space
    ci = threepoleadmittance(freq, A, B, C, tau1, tau2, tau3)

    # analytic DFT of a duty cycled square wave
    Sf = np.zeros_like(freq)*0.0j
    
    # even frequencies are zero unless the duty cycle is not 0.5
    if (dutycycle==0.5):
        oddInds = ((np.abs(np.mod(np.absolute(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = 1.0j/(pi*freq[oddInds]/sgfreq)*sgamp*Rsh*tracelength
    else:
        oddInds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = -1.0j/(2.0*pi*freq[oddInds]/sgfreq)*sgamp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[oddInds]/sgfreq*dutycycle)-1)
        
        evenInds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        evenInds[0] = False
        Sf[evenInds] = -1.0j/(2.0*pi*freq[evenInds]/sgfreq)*sgamp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[evenInds]/sgfreq*dutycycle)-1)
    
    # convolve the square wave with the fit
    SfTES = Sf*ci
    
    # inverse FFT to convert to time domain
    St = ifft(SfTES)

    return np.real(St)

def squarewaveguessparams(trace, sgamp, Rsh):
    """Function to guess the fit parameters for the 1-pole fit.
    
    Args:
        trace: The trace in time domain (in Amps).
        sgamp: The peak-to-peak size of the square wave jitter (in Amps)
        Rsh: Shunt resistance of the TES electronics (in Ohms)
        
    Returns:
        A0: Guess of the fit parameter A (in Ohms)
        tau20: Guess of the fit parameter tau2 (in s)
    
    """
    
    dIs = max(trace) - min(trace)
    A0 = sgamp*Rsh/dIs
    tau20 = 1.0e-6
    return A0, tau20

def guessdidvparams(trace, traceTopSlope, sgamp, Rsh, L0=1.0e-7):
    """Function to find the fit parameters for either the 1-pole (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit. 
    
    Args:
        trace: The trace in time domain (in Amps)
        traceTopSlope: The flat parts of the trace (in Amps)
        sgamp: The peak-to-peak size of the square wave jitter (in Amps)
        Rsh: Shunt resistance of the TES electronics (in Ohms)
        L0: The guess of the inductance (in Henries)
        
    Returns:
        A0: Guess of the fit parameter A (in Ohms)
        B0: Guess of the fit parameter B (in Ohms)
        tau10: Guess of the fit parameter tau1 (in s)
        tau20: Guess of the fit parameter tau2 (in s)
        isLoopGainSub1: Boolean flag that gives whether the loop gain is greater than one (False) or less than one (True)
        
    """
    
    # get the mean of the trace
    dIsmean = np.mean(trace)
    # mean of the top slope points
    dIsTopSlopemean = np.mean(traceTopSlope)
    #check if loop gain is less than or greater than one (check if we are inverted of not)
    isLoopGainSub1 = dIsTopSlopemean < dIsmean
    
    # the dIdV(0) can be estimate as twice the difference of the top slope points and the mean of the trace
    dIs0 = 2 * np.abs(dIsTopSlopemean-dIsmean)
    dIdV0 = dIs0/(sgamp*Rsh)
    
    # beta can be estimated from the size of the overshoot
    # estimate size of overshoot as maximum of trace minus the dIsTopSlopemean
    dIsTop = np.max(trace)-dIsTopSlopemean
    dIsdVTop = dIsTop/(sgamp*Rsh)
    A0 = 1.0/dIsdVTop
    tau20 = L0/A0
    
    if isLoopGainSub1:
        # loop gain < 1
        B0 = 1.0/dIdV0 - A0
        if B0 > 0.0:
            B0 = -B0 # this should be positive, but since the optimization algorithm checks both cases, we need to make sure it's negative, otherwise the guess will not be within the allowed bounds
        tau10 = -100e-6 # guess a slower tauI
    else:
        # loop gain > 1
        B0 = -1.0/dIdV0 - A0
        tau10 = -100e-7 # guess a faster tauI

    return A0, B0, tau10, tau20, isLoopGainSub1

def fitdidv(freq, dIdV, yerr=None, A0=0.25, B0=-0.6, C0=-0.6, tau10=-1.0/(2*pi*5e2), tau20=1.0/(2*pi*1e5), tau30=0.0, dt=-10.0e-6, poles=2):
    """Function to find the fit parameters for either the 1-pole (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit. 
    
    Args:
        freq: Frequencies corresponding to the dIdV
        dIdV: Complex impedance extracted from the trace in frequency space
        yerr: Error at each frequency of the dIdV. Should be a complex number, e.g. yerr = yerr_real + 1.0j * yerr_imag, where yerr_real is the standard deviation of the real part of the dIdV, and yerr_imag is the standard deviation of the imaginary part of the dIdV
        A0: Guess of the fit parameter A (in Ohms)
        B0: Guess of the fit parameter B (in Ohms)
        C0: Guess of the fit parameter C
        tau10: Guess of the fit parameter tau1 (in s)
        tau20: Guess of the fit parameter tau2 (in s)
        tau30: Guess of the fit parameter tau3 (in s)
        dt: Guess of the time shift (in s)
        poles: The number of poles to use in the fit (should be 1, 2, or 3)
        
    Returns:
        popt: The fitted parameters for the specificed number of poles
        pcov: The corresponding covariance matrix for the fitted parameters
        cost: The cost of the the fit
        
    """
    
    if (poles==1):
        # assume the square wave is not inverted
        p0 = (A0, tau20, dt)
        bounds1 = ((0.0, 0.0, -1.0e-3),(np.inf, np.inf, 1.0e-3))
        # assume the square wave is inverted
        p02 = (-A0, tau20, dt)
        bounds2 = ((-np.inf, 0.0, -1.0e-3),(0.0, np.inf, 1.0e-3))
    elif (poles==2):
        # assume loop gain > 1, where B<0 and tauI<0
        p0 = (A0, B0, tau10, tau20, dt)
        bounds1 = ((0.0, -np.inf, -np.inf, 0.0, -1.0e-3),(np.inf, 0.0, 0.0, np.inf, 1.0e-3))
        # assume loop gain < 1, where B>0 and tauI>0
        p02 = (A0, -B0, -tau10, tau20, dt)
        bounds2 = ((0.0, 0.0, 0.0, 0.0, -1.0e-3),(np.inf, np.inf, np.inf, np.inf, 1.0e-3))
    elif (poles==3):
        # assume loop gain > 1, where B<0 and tauI<0
        p0 = (A0, B0, C0, tau10, tau20, tau30, dt)
        bounds1 = ((0.0, -np.inf, -np.inf, -np.inf, 0.0, 0.0, -1.0e-3),(np.inf, 0.0, 0.0, 0.0, np.inf, np.inf, 1.0e-3))
        # assume loop gain < 1, where B>0 and tauI>0
        p02 = (A0, -B0, -C0, -tau10, tau20, tau30, dt)
        bounds2 = ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0e-3),(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0e-3))
        
    def residual(params):
        # define a residual for the nonlinear least squares algorithm 
        # different functions for different amounts of poles
        if (poles==1):
            A, tau2, dt = params
            ci = onepoleadmittance(freq, A, tau2) * np.exp(-2.0j*pi*freq*dt)
        elif(poles==2):
            A, B, tau1, tau2, dt = params
            ci = twopoleadmittance(freq, A, B, tau1, tau2) * np.exp(-2.0j*pi*freq*dt)
        elif(poles==3):
            A, B, C, tau1, tau2, tau3, dt = params
            ci = threepoleadmittance(freq, A, B, C, tau1, tau2, tau3) * np.exp(-2.0j*pi*freq*dt)
        
        # the difference between the data and the fit
        diff = dIdV-ci
        # get the weights from yerr, these should be 1/(standard deviation) for real and imaginary parts
        if (yerr is None):
            weights = 1.0+1.0j
        else:
            weights = 1.0/yerr.real+1.0j/yerr.imag
        # create the residual vector, splitting up real and imaginary parts of the residual separately
        z1d = np.zeros(freq.size*2, dtype=np.float64)
        z1d[0:z1d.size:2] = diff.real*weights.real
        z1d[1:z1d.size:2] = diff.imag*weights.imag
        return z1d

    # res1 assumes loop gain > 1, where B<0 and tauI<0
    res1 = least_squares(residual, p0, bounds=bounds1, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    # res2 assumes loop gain < 1, where B>0 and tauI>0
    res2 = least_squares(residual, p02, bounds=bounds2, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    
    # check which loop gain casez gave the better fit
    if (res1['cost'] < res2['cost']):
        res = res1
    else:
        res = res2
        
    popt = res['x']
    cost = res['cost']
        
    # check if the fit failed (usually only happens when we reach maximum evaluations, likely when fitting assuming the wrong loop gain)
    if (not res1['success'] and not res2['success']):
        print("Fit failed: "+str(res1['status'])+", "+str(poles)+"-pole Fit")
        
    # take matrix product of transpose of jac and jac, take the inverse to get the analytic covariance matrix
    pcovinv = np.dot(res["jac"].transpose(), res["jac"])
    pcov = np.linalg.inv(pcovinv)
    
    return popt,pcov,cost

def converttotesvalues(popt, pcov, R0, Rl, dR0=0.001, dRl=0.001):
    """Function to convert the fit parameters for either 1-pole (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit to the corresponding TES parameters: 1-pole (Rtot, L, R0, Rl, dt), 2-pole (Rl, R0, beta, l, L, tau0, dt), and 3-pole (no conversion done).
    
    Args:
        popt: The fit parameters for either the 1-pole, 2-pole, or 3-pole fit
        pcov: The corresponding covariance matrix for the fit parameters
        R0: The resistance of the TES (in Ohms)
        Rl: The load resistance of the TES circuit (in Ohms)
        dR0: The error in the R0 value (in Ohms)
        dRl: The error in the Rl value (in Ohms)
        
    Returns:
        popt_out: The TES parameters for the specified fit
        pcov_out: The corresponding covariance matrix for the TES parameters
        
    """
    
    if len(popt)==3:
        ## one pole
        # extract fit parameters
        A = popt[0]
        tau2 = popt[1]
        dt = popt[2]
        
        # convert fit parameters to Rtot=R0+Rl and L
        Rtot = A
        L = A*tau2
        
        popt_out = np.array([Rtot, L, R0, Rl, dt])
        
        # create new covariance matrix (needs to be the correct size)
        pcov_orig = pcov
        pcov_in = np.zeros((5,5))
        row,col = np.indices((2,2))
        
        # populate the new covariance matrix with the uncertainties in R0, Rl, and dt
        pcov_in[row, col] = pcov_orig[row, col]
        vardt = pcov_orig[2,2]
        pcov_in[2,2] = dR0**2
        pcov_in[3,3] = dRl**2
        pcov_in[4,4] = vardt

        # calculate the Jacobian
        jac = np.zeros((5,5))
        jac[0,0] = 1             # dRtotdA
        jac[1,0] = tau2          # dLdA
        jac[1,1] = A             # dLdtau2
        jac[2,2] = 1             # dR0dR0
        jac[3,3] = 1             # dRldRl
        jac[4,4] = 1             # ddtddt
        
        # use the Jacobian to populate the rest of the covariance matrix
        jact = np.transpose(jac)
        pcov_out = np.dot(jac, np.dot(pcov_in, jact))
        
    elif len(popt)==5:
        ## two poles
        # extract fit parameters
        A = popt[0]
        B = popt[1]
        tau1 = popt[2]
        tau2 = popt[3]
        dt = popt[4]
        
        # get covariance matrix for beta, l, L, tau, R0, Rl, dt
        # create new covariance matrix (needs to be the correct size)
        pcov_orig = np.copy(pcov)
        pcov_in = np.zeros((7,7))
        row,col = np.indices((4,4))

        # populate the new covariance matrix with the uncertainties in R0, Rl, and dt
        pcov_in[row, col] = np.copy(pcov_orig[row, col])
        vardt = pcov_orig[4,4]
        pcov_in[4,4] = dRl**2
        pcov_in[5,5] = dR0**2
        pcov_in[6,6] = vardt
        
        # convert A, B tau1, tau2 to beta, l, L, tau
        beta  = (A-Rl)/R0 - 1.0
        l = B/(A+B+R0-Rl)
        L = A*tau2
        tau = tau1 * (A+R0-Rl)/(A+B+R0-Rl)
        popt_out = np.array([Rl,R0,beta,l,L,tau,dt])
        
        # calculate the Jacobian
        jac = np.zeros((7,7))
        jac[0,4] = 1.0                              #dRldRl
        jac[1,5] = 1.0                              #dR0dR0
        jac[2,0] = 1.0/R0                           #dbetadA
        jac[2,4] = -1.0/R0                          #dbetadRl
        jac[2,5] = -(A-Rl)/R0**2.0                  #dbetadR0
        jac[3,0] = -B/(A+B+R0-Rl)**2.0              #dldA (l = Irwin's loop gain = (P0 alpha)/(G T0))
        jac[3,1] = (A+R0-Rl)/(A+B+R0-Rl)**2.0       #dldB
        jac[3,4] = B/(A+B+R0-Rl)**2.0               #dldRl
        jac[3,5] = -B/(A+B+R0-Rl)**2.0              #dldR0
        jac[4,0] = tau2                             #dLdA
        jac[4,3] = A                                #dLdtau2
        jac[5,0] = (tau1*B)/(A+B+R0-Rl)**2.0        #dtaudA
        jac[5,1] = -tau1*(A+R0-Rl)/(A+B+R0-Rl)**2.0 #dtaudB
        jac[5,2] = (A+R0-Rl)/(A+B+R0-Rl)            #dtaudtau1
        jac[5,4] = -B*tau1/(A+B+R0-Rl)**2.0         #dtaudRl
        jac[5,5] = B*tau1/(A+B+R0-Rl)**2.0          #dtaudR0
        jac[6,6] = 1.0                              #ddtddt
        
        # use the Jacobian to populate the rest of the covariance matrix
        jact = np.transpose(jac)
        pcov_out = np.dot(jac, np.dot(pcov_in, jact))
        
    elif len(popt)==7:
        ##three poles (no conversion, since this is just a toy model)
        popt_out = popt
        pcov_out = pcov

    return popt_out, pcov_out

def fitdidvpriors(freq, dIdV, priors, invpriorsCov, yerr=None, Rl=0.35, R0=0.130, beta=0.5, l=10.0, L=500.0e-9, tau0=500.0e-6, dt=-10.0e-6):
    """Function to directly fit Irwin's TES parameters (Rl, R0, beta, l, L, tau0, dt) with the knowledge of prior known values any number of the parameters. In order for the degeneracy of the parameters to be broken, at least 2 fit parameters should have priors knowledge. This is usually Rl and R0, as these can be known from IV data.
    
    Args:
        freq: Frequencies corresponding to the dIdV
        dIdV: Complex impedance extracted from the trace in frequency space
        priors: Prior known values of Irwin's TES parameters for the trace. Should be in the order of (Rl,R0,beta,l,L,tau0,dt)
        invpriorsCov: Inverse of the covariance matrix of the prior known values of Irwin's TES parameters for the trace (any values that are set to zero mean that we have no knowledge of that 
        yerr: Error at each frequency of the dIdV. Should be a complex number, e.g. yerr = yerr_real + 1.0j * yerr_imag, where yerr_real is the standard deviation of the real part of the dIdV, and yerr_imag is the standard deviation of the imaginary part of the dIdV
        Rl: Guess of the load resistance of the TES circuit (in Ohms)
        R0: Guess of the resistance of the TES (in Ohms)
        beta: Guess of the current sensitivity beta
        l: Guess of Irwin's loop gain
        L: Guess of the inductance (in Henrys)
        tau0: Guess of the thermal time constant (in s)
        dt: Guess of the time shift (in s)
        
    Returns:
        popt: The fitted parameters in the order of (Rl, R0, beta, l, L, tau0, dt)
        pcov: The corresponding covariance matrix for the fitted parameters
        cost: The cost of the the fit
        
    """
    
    p0 = (Rl, R0, beta, l, L, tau0, dt)
    bounds=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0e-3),(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0e-3))
    
    def residualPriors(params, priors, invpriorsCov):
        # priors = prior known values of Rl, R0, beta, l, L, tau0 (2-pole)
        # invpriorsCov = inverse of the covariance matrix of the priors
        
        z1dpriors = np.sqrt((priors-params).dot(invpriorsCov).dot(priors-params))
        return z1dpriors
        
    def residual(params):
        # define a residual for the nonlinear least squares algorithm
        # different functions for different amounts of poles
        Rl, R0, beta, l, L, tau0, dt=params
        ci = twopoleadmittancepriors(freq, Rl, R0, beta, l, L, tau0) * np.exp(-2.0j*pi*freq*dt)
        
        # the difference between the data and the fit
        diff = dIdV-ci
        # get the weights from yerr, these should be 1/(standard deviation) for real and imaginary parts
        if(yerr is None):
            weights = 1.0+1.0j
        else:
            weights = 1.0/yerr.real+1.0j/yerr.imag
        
        # create the residual vector, splitting up real and imaginary parts of the residual separately
        z1d = np.zeros(freq.size*2+1, dtype = np.float64)
        z1d[0:z1d.size-1:2] = diff.real*weights.real
        z1d[1:z1d.size-1:2] = diff.imag*weights.imag
        z1d[-1] = residualPriors(params,priors,invpriorsCov)
        return z1d

    def jaca(params):
        # analytically calculate the Jacobian for 2 pole and three pole cases
        popt = params

        # popt = Rl,R0,beta,l,L,tau0,dt
        Rl = popt[0]
        R0 = popt[1]
        beta = popt[2]
        l = popt[3]
        L = popt[4]
        tau0 = popt[5]
        dt = popt[6]
        
        # derivative of 1/x = -1/x**2 (without doing chain rule)
        deriv1 = -1.0/(2.0j*pi*freq*L + Rl + R0*(1.0+beta) + R0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*freq*tau0/(1-l)))**2
        
        dYdRl = np.zeros(freq.size*2, dtype = np.float64)
        dYdRlcomplex = deriv1 * np.exp(-2.0j*pi*freq*dt)
        dYdRl[0:dYdRl.size:2] = np.real(dYdRlcomplex)
        dYdRl[1:dYdRl.size:2] = np.imag(dYdRlcomplex)

        dYdR0 = np.zeros(freq.size*2, dtype = np.float64)
        dYdR0complex = deriv1 * (1.0+beta + l * (2.0+beta)/(1.0 - l +2.0j*pi*freq*tau0))  * np.exp(-2.0j*pi*freq*dt)
        dYdR0[0:dYdR0.size:2] = np.real(dYdR0complex)
        dYdR0[1:dYdR0.size:2] = np.imag(dYdR0complex)

        dYdbeta = np.zeros(freq.size*2, dtype = np.float64)
        dYdbetacomplex = deriv1 * (R0+2.0j*pi*freq*R0*tau0)/(1.0-l + 2.0j*pi*freq*tau0) * np.exp(-2.0j*pi*freq*dt)
        dYdbeta[0:dYdbeta.size:2] = np.real(dYdbetacomplex)
        dYdbeta[1:dYdbeta.size:2] = np.imag(dYdbetacomplex)

        dYdl = np.zeros(freq.size*2, dtype = np.float64)
        dYdlcomplex = deriv1 * R0*(2.0+beta)*(1.0+2.0j*pi*freq*tau0)/(1.0-l+2.0j*pi*freq*tau0)**2 * np.exp(-2.0j*pi*freq*dt)
        dYdl[0:dYdl.size:2] = np.real(dYdlcomplex)
        dYdl[1:dYdl.size:2] = np.imag(dYdlcomplex)

        dYdL = np.zeros(freq.size*2, dtype = np.float64)
        dYdLcomplex = deriv1 * 2.0j*pi*freq * np.exp(-2.0j*pi*freq*dt)
        dYdL[0:dYdL.size:2] = np.real(dYdLcomplex)
        dYdL[1:dYdL.size:2] = np.imag(dYdLcomplex)

        dYdtau0 = np.zeros(freq.size*2, dtype = np.float64)
        dYdtau0complex = deriv1 * -2.0j*pi*freq*l*R0*(2.0+beta)/(1.0-l+2.0j*pi*freq*tau0)**2 * np.exp(-2.0j*pi*freq*dt)
        dYdtau0[0:dYdtau0.size:2] = np.real(dYdtau0complex)
        dYdtau0[1:dYdtau0.size:2] = np.imag(dYdtau0complex)
        
        dYddt = np.zeros(freq.size*2, dtype = np.float64)
        dYddtcomplex = -2.0j*pi*freq/(2.0j*pi*freq*L + Rl + R0*(1.0+beta) + R0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*freq*tau0/(1-l))) * np.exp(-2.0j*pi*freq*dt)
        dYddt[0:dYddt.size:2] = np.real(dYddtcomplex)
        dYddt[1:dYddt.size:2] = np.imag(dYddtcomplex)

        jac = np.column_stack((dYdRl, dYdR0, dYdbeta, dYdl, dYdL, dYdtau0, dYddt))
        return jac

    res = least_squares(residual, p0, bounds=bounds, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    
    popt = res['x']
    cost = res['cost']
    
    # check if the fit failed (usually only happens when we reach maximum evaluations, likely when fitting assuming the wrong loop gain)
    if (not res['success']):
        print('Fit failed: '+str(res['status']))

    # analytically calculate the covariance matrix
    if (yerr is None):
        weights = 1.0+1.0j
    else:
        weights = 1.0/yerr.real+1.0j/yerr.imag
    
    #convert weights to variances (want 1/var, as we are creating the inverse of the covariance matrix)
    weightVals = np.zeros(freq.size*2, dtype = np.float64)
    weightVals[0:weightVals.size:2] = weights.real**2
    weightVals[1:weightVals.size:2] = weights.imag**2
    
    jac = jaca(popt)
    jact = np.transpose(jac)
    wjac = np.zeros_like(jac)
    
    # right multiply inverse of covariance matrix by the jacobian (we do this element by element, to avoid creating a huge covariance matrix)
    for ii in range(0, len(popt)):
        wjac[:,ii] = np.multiply(weightVals, jac[:,ii])
        
    # left multiply by the jacobian and take the inverse to get the analytic covariance matrix
    pcovinv = np.dot(jact, wjac) + invpriorsCov
    pcov = np.linalg.inv(pcovinv)
    
    return popt, pcov, cost

def convertfromtesvalues(popt, pcov):
    """Function to convert from Irwin's TES parameters (Rl, R0, beta, l, L, tau0, dt) to the fit parameters (A, B, tau1, tau2, dt)
    
    Args:
        popt: Irwin's TES parameters in the order of (Rl, R0, beta, l, L, tau0, dt), should be a 1-dimensional np.array of length 7
        pcov: The corresponding covariance matrix for Irwin's TES parameters. Should be a 2-dimensional, 7-by-7 np.array
        
    Returns:
        popt_out: The fit parameters in the order of (A, B, tau1, tau2, dt)
        pcov_out: The corresponding covariance matrix for the fit parameters
        
    """
   
    ## two poles
    # extract fit parameters
    Rl = popt[0]
    R0 = popt[1]
    beta = popt[2]
    l = popt[3]
    L = popt[4]
    tau0 = popt[5]
    dt = popt[6]
    
    # convert A, B tau1, tau2 to beta, l, L, tau
    A = Rl + R0 * (1.0+beta)
    B = R0 * l/(1.0-l) * (2.0+beta)
    tau1 = tau0/(1.0-l)
    tau2 = L/(Rl+R0*(1.0+beta))
    
    popt_out = np.array([A, B, tau1, tau2, dt])

    # calculate the Jacobian
    jac = np.zeros((5,7))
    jac[0,0] = 1.0        #dAdRl
    jac[0,1] = 1.0 + beta #dAdR0
    jac[0,2] = R0         #dAdbeta
    jac[1,1] = l/(1.0-l) * (2.0+beta) #dBdR0
    jac[1,2] = l/(1.0-l) * R0 #dBdbeta
    jac[1,3] = R0 * (2.0+beta)/(1.0-l)  + l/(1.0-l)**2.0 * R0 * (2.0+beta) #dBdl
    jac[2,3] = tau0/(1.0-l)**2.0  #dtau1dl
    jac[2,5] = 1.0/(1.0-l) #dtau1dtau0
    jac[3,0] = -L/(Rl+R0*(1.0+beta))**2.0 #dtau2dRl
    jac[3,1] = -L * (1.0+beta)/(Rl+R0*(1.0+beta))**2 #dtau2dR0
    jac[3,2] = -L*R0/(Rl+R0*(1.0+beta))**2.0 #dtau2dbeta
    jac[3,4] = 1.0/(Rl+R0*(1.0+beta))#dtau2dL
    jac[4,6] = 1.0 #ddtddt
    

    # use the Jacobian to populate the rest of the covariance matrix
    jact = np.transpose(jac)
    pcov_out = np.dot(jac, np.dot(pcov, jact))
        

    return popt_out, pcov_out

def findpolefalltimes(params):
    """Function for taking TES params from a 1-pole, 2-pole, or 3-pole dIdV and calculating the falltimes (i.e. the values of the poles in the complex plane)
    
    Args:
        params: TES parameters for either 1-pole, 2-pole, or 3-pole dIdV. This will be a 1-dimensional np.array of varying length, depending on the fit. 1-pole fit has 3 parameters (A,tau2,dt), 2-pole fit has 5 parameters (A,B,tau1,tau2,dt), and 3-pole fit has 7 parameters (A,B,C,tau1,tau2,tau3,dt). The parameters should be in that order, and any other number of parameters will print a warning and return zero.
        
    Returns:
        np.sort(fallTimes): The falltimes for the dIdV fit, sorted from fastest to slowest.
        
    """
    
    # convert dVdI time constants to fall times of dIdV
    if len(params)==3:
        # one pole fall time for dIdV is same as tau2=L/R
        A, tau2, dt = params
        fallTimes = np.array([tau2])
        
    elif len(params)==5:
        # two pole fall times for dIdV is different than tau1, tau2
        A, B, tau1, tau2, dt = params
        
        def twopoleequations(p):
            taup,taum = p
            eq1 = taup+taum - A/(A+B)*(tau1+tau2)
            eq2 = taup*taum-A/(A+B)*tau1*tau2
            return (eq1, eq2)
        
        taup, taum = fsolve(twopoleequations ,(tau1, tau2))
        fallTimes = np.array([taup, taum])
        
    elif len(params)==7:
        # three pole fall times for dIdV is different than tau1, tau2, tau3
        A, B, C, tau1, tau2, tau3, dt = params
        
        def threepoleequations(p):
            taup, taum, taun = p
            eq1 = taup+taum+taun-(A*tau1+A*(1.0-C)*tau2+(A+B)*tau3)/(A*(1.0-C)+B)
            eq2 = taup*taum+taup*taun+taum*taun - (tau1*tau2+tau1*tau3+tau2*tau3)*A/(A*(1.0-C)+B)
            eq3 = taup*taum*taun - tau1*tau2*tau3*A/(A*(1.0-C)+B)
            return (eq1, eq2, eq3)
        
        taup, taum, taun = fsolve(threepoleequations, (tau1, tau2, tau3))
        fallTimes = np.array([taup, taum, taun])
        
    else:
        print("Wrong number of input parameters, returning zero...")
        fallTimes = np.zeros(1)
    
    # return fall times sorted from shortest to longest
    return np.sort(fallTimes)

def deconvolvedidv(x, trace, Rsh, sgamp, sgfreq, dutycycle):
    """Function for taking a trace with a known square wave jitter and extracting the complex impedance via deconvolution of the square wave and the TES response in frequency space.
    
    Args:
        x: Time values for the trace
        trace: The trace in time domain (in Amps)
        Rsh: Shunt resistance for electronics (in Ohms)
        sgamp: Peak to peak value of square wave jitter (in Amps)
        sgfreq: Frequency of square wave jitter
        dutycycle: duty cycle of square wave jitter
        
    Returns:
        freq: The frequencies that each point of the trace corresponds to
        dIdV: Complex impedance of the trace in frequency space
        zeroInds: Indices of the frequencies where the trace's Fourier Transform is zero. Since we divide by the FT of the trace, we need to know which values should be zero, so that we can ignore these points in the complex impedance.
        
    """
    
    tracelength = len(x)
    
    # get the frequencies for a DFT, based on the sample rate of the data
    dx = x[1]-x[0]
    freq = fftfreq(len(x), d=dx)
    
    # FFT of the trace
    St = fft(trace)
    
    # analytic DFT of a duty cycled square wave
    Sf = np.zeros_like(freq)*0.0j
    
    # even frequencies are zero unless the duty cycle is not 0.5
    if (dutycycle==0.5):
        oddInds = ((np.abs(np.mod(np.absolute(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = 1.0j/(pi*freq[oddInds]/sgfreq)*sgamp*Rsh*tracelength
    else:
        oddInds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = -1.0j/(2.0*pi*freq[oddInds]/sgfreq)*sgamp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[oddInds]/sgfreq*dutycycle)-1)
        
        evenInds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        evenInds[0] = False
        Sf[evenInds] = -1.0j/(2.0*pi*freq[evenInds]/sgfreq)*sgamp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[evenInds]/sgfreq*dutycycle)-1)
    
    # the tracelength/2 value from the FFT is purely real, which can cause errors when taking the standard deviation (get stddev = 0 for real part of dIdV at this frequency, leading to a divide by zero when calculating the residual when fitting)
    Sf[tracelength//2] = 0.0j
    
    # deconvolve the trace from the square wave to get the dIdV in frequency space
    dVdI = (Sf/St)
    
    # set values that are within floating point error of zero to 1.0 + 1.0j (we will give these values virtually infinite error, so the value doesn't matter. Setting to 1.0+1.0j avoids divide by zero if we invert)
    zeroInds = np.abs(dVdI) < 1e-16
    dVdI[zeroInds] = (1.0+1.0j)
    
    # convert to complex admittance
    dIdV = 1.0/dVdI

    return freq, dIdV, zeroInds


class DIDV(object):
    
    def __init__(self, rawtraces, samplerate, sgfreq, sgamp, Rsh, R0=0.3, dR0=0.001, Rl=0.01, dRl=0.001,
                 timeoffset=0, tracegain=1.0, dutycycle=0.5, add180phase=False, priors=None, invpriorscov=None, dt0=10.0e-6):
        
        self.rawtraces = rawtraces
        self.samplerate = samplerate
        self.sgfreq = sgfreq
        self.sgamp = sgamp
        self.R0 = R0
        self.dR0 = dR0
        self.Rl = Rl
        self.dRl = dRl
        self.Rsh = Rsh
        self.timeoffset = timeoffset
        self.tracegain = tracegain
        self.dutycycle = dutycycle
        self.add180phase = add180phase
        self.priors = priors
        self.invpriorscov = invpriorscov
        self.dt0 = dt0
        
        self.freq = None
        self.time = None
        self.ntraces = None
        self.traces = None
        self.flatinds = None
        self.tmean = None
        self.zeroinds = None
        self.didvstd = None
        self.didvmean = None
        self.offset = None
        self.doffset = None
        
        self.fitparams1 = None
        self.fitcov1 = None
        self.fitcost1 = None
        self.irwinparams1 = None
        self.irwincov1 = None
        self.falltimes1 = None
        self.didvfit1_timedomain = None
        self.didvfit1_freqdomain = None
        
        self.fitparams2 = None
        self.fitcov2 = None
        self.fitcost2 = None
        self.irwinparams2 = None
        self.irwincov2 = None
        self.falltimes2 = None
        self.didvfit2_timedomain = None
        self.didvfit2_freqdomain = None
        
        self.fitparams3 = None
        self.fitcov3 = None
        self.fitcost3 = None
        self.irwinparams3 = None
        self.irwincov3 = None
        self.falltimes3 = None
        self.didvfit3_timedomain = None
        self.didvfit3_freqdomain = None
        
        self.fitparams2priors = None
        self.fitcov2priors = None
        self.fitcost2 = None
        self.irwinparams2priors = None
        self.irwincov2priors = None
        self.falltimes2priors = None
        self.didvfit2priors_timedomain = None
        self.didvfit2priors_freqdomain = None
        
    def processtraces(self):
        
        #get number of traces 
        self.ntraces = len(self.rawtraces)
        #converting sampling rate to time step
        dt = (1.0/self.samplerate) 

        #get trace x values (i.e. time) in seconds
        bins = np.arange(0, len(self.rawtraces[0]))

        # add half a period of the square wave frequency if add180phase is True
        if (self.add180phase):
            self.timeoffset = self.timeoffset + 1/(2*self.sgfreq)

        # apply time offset
        self.time = bins*dt-self.timeoffset
        indOffset = int(self.timeoffset*self.samplerate)

        #figure out how many dIdV periods are in the trace, including the time offset
        period = 1.0/self.sgfreq
        nPeriods = np.floor((max(self.time)-self.time[indOffset])/period)

        # find which indices to keep in order to have an integer number of periods, as well as the inputted timeoffset
        indMax = int(nPeriods*self.samplerate/self.sgfreq)
        good_inds = range(indOffset, indMax+indOffset)

        # ignore the tail of the trace after the last period, as this tail just adds artifacts to the FFTs
        self.time = self.time[good_inds]
        self.traces = self.rawtraces[:,good_inds]/(self.tracegain) # convert to Amps

        #need these x-values to be properly scaled for maximum likelihood slope fitting
        period_unscaled = self.samplerate/self.sgfreq

        #save the  "top slope" points in the trace, which are the points just before the overshoot in the dI/dV
        flatindsTemp = list()
        for i in range(0, int(nPeriods)):
            # get index ranges for flat parts of trace
            flatIndLow = int((float(i)+0.25)*period_unscaled)
            flatIndHigh = int((float(i)+0.48)*period_unscaled)
            flatindsTemp.append(range(flatIndLow, flatIndHigh))
        self.flatinds = np.array(flatindsTemp).flatten()
        
        #for storing results
        dIdVs=list()

        for trace in self.traces:

            # deconvolve the trace from the square wave to get the dI/dV in frequency domain
            dIdVi = deconvolvedidv(self.time,trace,self.Rsh,self.sgamp,self.sgfreq,self.dutycycle)[1]
            dIdVs.append(dIdVi)

        #convert to numpy structure
        dIdVs=np.array(dIdVs)

        means=np.mean(self.traces,axis=1)

        #store results
        self.tmean = np.mean(self.traces,axis=0)
        self.freq,self.zeroinds = deconvolvedidv(self.time,self.tmean,self.Rsh,self.sgamp,self.sgfreq,self.dutycycle)[::2]

        # divide by sqrt(N) for standard deviation of mean
        self.didvstd = stdcomplex(dIdVs)/np.sqrt(self.ntraces)
        self.didvstd[self.zeroinds] = (1.0+1.0j)*1.0e20
        self.didvmean = np.mean(dIdVs, axis=0)

        self.offset = np.mean(means)
        self.doffset = np.std(means)/np.sqrt(self.ntraces)
    
    def dofit(self,poles):
        if self.tmean is None:
            self.processtraces()

        if poles==1:
            # guess the 1 pole square wave parameters
            A0_1pole, tau20_1pole = squarewaveguessparams(self.tmean, self.sgamp, self.Rsh)
            
            # 1 pole fitting
            self.fitparams1, self.fitcov1, self.fitcost1 = fitdidv(self.freq, self.didvmean, yerr=self.didvstd, A0=A0_1pole, tau20=tau20_1pole, dt=self.dt0, poles=poles)
            
            # Convert parameters from 1-pole fit to the Irwin parameters
            self.irwinparams1, self.irwincov1 = converttotesvalues(self.fitparams1, self.fitcov1, self.R0, self.Rl, dR0=self.dR0, dRl=self.dRl)
            
            # Convert to dIdV falltimes
            self.falltimes1 = findpolefalltimes(self.fitparams1)
            
            self.didvfit1_timedomain = convolvedidv(self.time, self.fitparams1[0], 0.0, 0.0, 0.0, self.fitparams1[1], 0.0, self.sgamp, self.Rsh, self.sgfreq, self.dutycycle)+self.offset
            
            ## save the fits in frequency domain as variables for saving/plotting
            self.didvfit1_freqdomain = onepoleadmittance(self.freq, self.fitparams1[0], self.fitparams1[1]) * np.exp(-2.0j*pi*self.freq*self.fitparams1[2])
        
        elif poles==2:
            
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isLoopGainSub1 = guessdidvparams(self.tmean, self.tmean[self.flatinds], self.sgamp, self.Rsh, L0=1.0e-7)
            
            # 2 pole fitting
            self.fitparams2, self.fitcov2, self.fitcost2 = fitdidv(self.freq, self.didvmean, yerr=self.didvstd, A0=A0, B0=B0, tau10=tau10, tau20=tau20, dt=self.dt0, poles=poles)
            
            # Convert parameters from 2-pole fit to the Irwin parameters
            self.irwinparams2, self.irwincov2 = converttotesvalues(self.fitparams2, self.fitcov2, self.R0, self.Rl, dR0=self.dR0, dRl=self.dRl)
            
            # Convert to dIdV falltimes
            self.falltimes2 = findpolefalltimes(self.fitparams2)
            
            self.didvfit2_timedomain = convolvedidv(self.time, self.fitparams2[0], self.fitparams2[1], 0.0, self.fitparams2[2], self.fitparams2[3], 0.0, self.sgamp, self.Rsh, self.sgfreq, self.dutycycle)+self.offset
            
            ## save the fits in frequency domain as variables for saving/plotting
            self.didvfit2_freqdomain = twopoleadmittance(self.freq, self.fitparams2[0], self.fitparams2[1], self.fitparams2[2], self.fitparams2[3]) * np.exp(-2.0j*pi*self.freq*self.fitparams2[4])
        
        elif poles==3:
            
            if self.fitparams2 is None:
                # Guess the 3-pole fit starting parameters from 2-pole fit guess
                A0, B0, tau10, tau20, isLoopGainSub1 = guessdidvparams(self.tmean, self.tmean[self.flatinds], self.sgamp, self.Rsh, L0=1.0e-7)
                B0 = -abs(B0)
                C0 = -0.01 
                tau10 = -abs(tau10)
                tau30 = 1.0e-4 
                dt0 = self.dt0
            else:
                A0 = self.fitparams2[0] 
                B0 = -abs(self.fitparams2[1]) 
                C0 = -0.01 
                tau10 = -abs(self.fitparams2[2]) 
                tau20 = self.fitparams2[3] 
                tau30 = 1.0e-4 
                dt0 = self.fitparams2[4]
            
            # 3 pole fitting
            self.fitparams3, self.fitcov3, self.fitcost3 = fitdidv(self.freq, self.didvmean, yerr=self.didvstd, A0=A0, B0=B0, C0=C0, tau10=tau10, tau20=tau20, tau30=tau30, dt=dt0, poles=3)
        
            # Convert to dIdV falltimes
            self.falltimes3 = findpolefalltimes(self.fitparams3)
        
            self.didvfit3_timedomain = convolvedidv(self.time, self.fitparams3[0], self.fitparams3[1], self.fitparams3[2], self.fitparams3[3], self.fitparams3[4], self.fitparams3[5], self.sgamp, self.Rsh, self.sgfreq, self.dutycycle)+self.offset
            
            ## save the fits in frequency domain as variables for saving/plotting
            self.didvfit3_freqdomain = threepoleadmittance(self.freq, self.fitparams3[0], self.fitparams3[1], self.fitparams3[2], self.fitparams3[3], self.fitparams3[4], self.fitparams3[5]) * np.exp(-2.0j*pi*self.freq*self.fitparams3[6])
        
        else:
            raise ValueError("The number of poles should be 1, 2, or 3.")
        
    def dopriorsfit(self):
        if self.tmean is None:
            self.processtraces()
        
        if self.irwinparams2 is None:
            
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isLoopGainSub1 = guessdidvparams(self.tmean, self.tmean[self.flatinds], self.sgamp, self.Rsh, L0=1.0e-7)
            v2guess = np.array([A0, B0, tau10, tau20, self.dt0])
            priorsguess = converttotesvalues(v2guess, np.eye(5), self.R0, self.Rl)[0] # 2 pole params (beta, l, L, tau0, R0, Rl, dt)
            
            # guesses for the 2 pole priors fit (these guesses must be positive)
            beta0 = abs(priorsguess[2])
            l0 = abs(priorsguess[3])
            L0 = abs(priorsguess[4])
            tau0 = abs(priorsguess[5])
            dt0 = self.dt0
        else:
            # guesses for the 2 pole priors fit (these guesses must be positive), using the values from the non-priors 2-pole fit
            beta0 = abs(self.irwinparams2[2])
            l0 = abs(self.irwinparams2[3])
            L0 = abs(self.irwinparams2[4])
            tau0 = abs(self.irwinparams2[5])
            dt0 = self.irwinparams2[6]

        # 2 pole fitting
        self.irwinparams2priors, self.irwincov2priors, self.irwincost2priors = fitdidvpriors(self.freq, self.didvmean, self.priors, self.invpriorscov, yerr=self.didvstd, R0=abs(self.R0), Rl=abs(self.Rl), beta=beta0, l=l0, L=L0, tau0=tau0, dt=dt0)

        # convert answer back to A, B, tauI, tauEL basis for plotting
        fitparams2priors = convertfromtesvalues(self.irwinparams2priors, self.irwincov2priors)[0]

        # Find the dIdV falltimes
        self.falltimes2priors = findpolefalltimes(fitparams2priors)

        # save the fits with priors in time and frequency domain
        self.didvfit2priors_timedomain = convolvedidv(self.time, fitparams2priors[0], fitparams2priors[1], 0.0, fitparams2priors[2], fitparams2priors[3], 0.0, self.sgamp, self.Rsh, self.sgfreq, self.dutycycle)+self.offset
        
        self.didvfit2priors_freqdomain = twopoleadmittancepriors(self.freq, self.irwinparams2priors[0], self.irwinparams2priors[1], self.irwinparams2priors[2], self.irwinparams2priors[3], self.irwinparams2priors[4], self.irwinparams2priors[5]) * np.exp(-2.0j*pi*self.freq*self.irwinparams2priors[6])
        
    def doallfits(self):
        self.dofit(1)
        self.dofit(2)
        self.dofit(3)
        self.dopriorsfit()
    
    def getparams(self, poles, lgcirwin, lgcpriors):
        if lgcpriors == False:
            if lgcirwin == True:
                params = getattr(self, "irwinparams"+str(poles))
                cov = getattr(self, "irwincov"+str(poles))
            else:
                params = getattr(self, "fitparams" + str(poles))
                cov = getattr(self, "fitcov"+str(poles))
        else:
            if poles ==2:
                if lgcirwin == True:
                    params = getattr(self, "irwinparams" + str(poles)+"priors")
                    cov = getattr(self, "irwincov" + str(poles)+"priors")
                else:
                    params = getattr(self, "fitparams" + str(poles) + "priors")
                    cov = getattr(self, "fitcov" + str(poles) + "priors")
            else:
                params = None
                cov = None
        
        return params, cov
    
    def plot_full_trace(self, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        didvutils.plot_full_trace(self, poles = poles, plotpriors = plotpriors, lgcsave = lgcsave, savepath = savepath)
    
    def plot_single_period_of_trace(self, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        didvutils.plot_single_period_of_trace(self, poles = poles, plotpriors = plotpriors, lgcsave = lgcsave, savepath = savepath)
    
    def plot_zoomed_in_trace(self, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        didvutils.plot_zoomed_in_trace(self, poles = poles, plotpriors = plotpriors, lgcsave = lgcsave, savepath = savepath)
        
    def plot_didv_flipped(self, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        didvutils.plot_didv_flipped(self, poles = poles, plotpriors = plotpriors, lgcsave = lgcsave, savepath = savepath)
        
    def plot_re_im_didv(self, poles = 2, plotpriors = True, lgcsave = False, savepath = ""):
        didvutils.plot_re_im_didv(self, poles = poles, plotpriors = plotpriors, lgcsave = lgcsave, savepath = savepath)
    
