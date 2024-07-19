import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import qetpy.plotting as utils
import warnings
warnings.simplefilter('default')

__all__ = ["get_biasparams_i0", 
           "get_biasparams_ilg", 
           "get_biasparams_offsets",
           "get_biasparams_muon",
           "get_biasparams_normal_iv"
]


def _get_i0_offset(offset, offset_err, offset_dict, output_offset=None,
                   closed_loop_norm=None, output_gain=1,
                   lgc_invert_offset=False,
                   lgc_calibration_on=False, calibration_dict=None, 
                   lgc_diagnostics=False):
    """
    Gets and returns the current and uncertainty in the current
    through the TES using the offsets method
    
    Parameters
    ----------
    offset: float
        The current offset of the dIdV as fit (i.e. as measured, without
        any corrections)
        
    offset_err: float
        The uncertainty in the current offset of the dIdV fit
        
    offset_dict: dict
        Dictionary of offsets gotten from the IV sweep.
        
    output_offset: float, volts
        The output offset gotten from the event metadata. In units of volts,
        we correct for volts to amps conversion with the closed loop norm.
        
    closed_loop_norm: float, volts/amp=ohms
        The constant from the metadata used to translate the voltage measured by
        the DAQ into a current coming into the input coil of the SQUIDs. In units of
        volts/amp = ohms.
        
    output_gain: float, dimensionless
        The dimensionless gain for the front end electronics. Used to translate the
        output_offset in units of volts to the equivilant value read in the DAQ in
        units of volts.
        
    lgc_calibration_on : bool, optional
        By default False (i.e. not using the calibration). If True, uses the calibration_dict
        to more closely approximate how changing the output_offset changes the current
        measured.
        
    calibration_dict : dict, optional
        A dictonary of data used to more closely model the relationship between the
        output_offset and the change in the measured current in the device. 
        
    lgc_diagnostics : bool, optional
        Used if you want to see the raw currents and offsets and how they're
        added together. Prints these out
        
    Returns
    -------
    i0 : float
        The calculated absolute current through the TES
    i0_err : float
        The uncertainty in the absolute current through the TES
        
    """

    # SQUID controller variable offset
    delta_i_variable = 0
    if output_offset is not None:

        # check if "closed_loop_norm" available
        if closed_loop_norm is None:
            raise ValueError(
                'ERROR: "closed_loop_norm" parameter is '
                'required!'
            )
        
        # IV sweep "i0_variable_offset" (check old names for back compatibility)
        i0_variable_offset_sweep = None
        if 'i0_variable_offset' in offset_dict.keys():
            i0_variable_offset_sweep = offset_dict['i0_variable_offset']
        elif ('i0_changable_offset_cal' in offset_dict.keys()) and (lgc_calibration_on is True):
            i0_variable_offset_sweep  =  offset_dict['i0_changable_offset_cal']
        elif ('i0_changable_offset_uncal' in offset_dict.keys()) and (lgc_calibration_on is False):
            i0_variable_offset_sweep  =  offset_dict['i0_changable_offset_uncal']
        else:
            raise ValueError('ERROR: i0 variable offset not found in '
                             '"ivsweep_result" dictionary!')

        # current IV variable offset
        if lgc_calibration_on is False:
            i0_variable_offset = output_offset * output_gain/closed_loop_norm
        else:
            if calibration_dict is None:
                raise ValueError('ERROR: must include calibration_dict if '
                                 'lgc_calibration_on is True (i.e. being used)!'
                                 'Check offset dicts')
                                 
            elif (output_gain != 50):
                raise ValueError('ERROR: calibration only done for a gain of 50')
                
            elif (calibration_dict['model'] == 'twopartlinear'):
                m1, m2, b1, b2 = calibration_dict['params']
                if output_offset > 0.0:
                    i0_variable_offset = output_offset * m1 + b1
                else:
                    i0_variable_offset = output_offset * m2 + b2
                    
            elif (calibration_dict['model'] == 'lookup_extrapolated'):
                offsets_arr = calibration_dict['params']['lookup_x']
                currents_arr = calibration_dict['params']['lookup_y']
                params_low = calibration_dict['params']['params_low']
                params_high = calibration_dict['params']['params_high']
                
                def linear(x, m, b):
                    return x*m + b
                
                if output_offset in offsets_arr:
                    i0_variable_offset = currents_arr[list(offsets_arr).index(offset)]
                elif output_offset > 0.0:
                    i0_variable_offset = linear(output_offset, *params_high)
                else:
                    i0_variable_offset = linear(output_offset, *params_low)
                    
            else:
                raise ValueError('ERROR: unknown calibration_dict model')
            
       
        # delta variable offset
        delta_i_variable = i0_variable_offset - i0_variable_offset_sweep
       
        #alerts user when the current offset changed
        if np.abs(delta_i_variable) > 1e-9:
            print(" ")
            print("----------------------------------------------------------------")
            print("ALERT: Variable output voltage offset has changed since the IV sweep.")
            print("Thus, it needs to be taken into account when calculating true I0!")
            print("IV sweep variable offset [muAmps]: " + str(i0_variable_offset_sweep*1e6))
            print("This dIdV dataset variable offset [muAmps]: " + str(i0_variable_offset*1e6))
            print("RESULTS MAY NOT BE ACCURATE IF GAIN/OFFSET NON LINEARITIES!")
            print("----------------------------------------------------------------")
            print(" ")

    # calculate new i0
    current_didv = offset - delta_i_variable 
    if lgc_invert_offset:
        current_didv = -current_didv
    current_err_didv = offset_err
    i0 = current_didv - offset_dict['i0_off']
    
    if lgc_diagnostics:
        print("Current as measured from dIdV: " + str(offset) + " amps")
        print("Variable current as meaured from metadata for this dIdV: " + str(i0_variable_offset) + " amps")
        print(" ")
        print("Variable current as measured during IV when offsets were measured: " + 
              str(i0_variable_offset_sweep) + " amps")
        print("Current offset as measured from IV: " + str(offset_dict['i0_off']) + " amps")
        print(" ")
        print("Delta variable current: " + str(i0_variable_offset) + " - " + 
             str(i0_variable_offset_sweep) + " = \n " + str(delta_i_variable) + " amps")
        print(" ")
        print("True current through TES: " + str(offset) + " - " + str(delta_i_variable) + " - \n" + 
             str(offset_dict['i0_off']) + " = " + str(i0) + " amps")
    
    i0_err = np.sqrt((current_err_didv)**2.0 + (offset_dict['i0_off_err'])**2.0)
    
    if lgc_diagnostics:
        print(" ")
        print(" --------- ")
        print(" ")
        print("Current uncertainty from dIdV: " + str(current_err_didv) + " amps")
        print("Current offset uncertainty from IV: " + str(offset_dict['i0_off_err']) + " amps")
        print(" ")
        print("Total current uncetainty: (("  + str(current_err_didv) + ")**2.0 + (" +
              str(offset_dict['i0_off_err']) + ")**2.0 )**0.5 = \n" + 
              str(i0_err) + " amps")
    
    return i0, i0_err

def _get_ibias_offset(tes_bias, offset_dict, lgc_diagnostics=False):
    """
    Gets and returns the current and uncertainty in the current
    used to bias the TES from a metadata list
    
    Parameters
    ----------
        
    tes_bias: float
        The ibias gotten from the event metadata, i.e. without correcting for
        the ibias offset calculated from the IV curve
        
    offset_dict: dict
        Dictionary of offsets gotten from the IV sweep.
        
    lgc_diagnostics : bool, optional
        Used if you want to see the raw currents and offsets and how they're
        added together. Prints these out
        
    Returns
    -------
    ibias : float
        The calculated absolute current used to bias the TES
    ibias_err : float
        The uncertainty in the absolute current used to bias the TES
        
    """
    
    ibias = tes_bias - offset_dict['ibias_off']
    ibias_err = offset_dict['ibias_off_err']
    
    if lgc_diagnostics:
        print("Bias current from metadata: " + str(tes_bias) + " amps")
        print("Bias current offset from IV: " + str(offset_dict['ibias_off']) + " amps")
        print("True bias current: " + str(tes_bias) + " - " + str(offset_dict['ibias_off']) + 
             " = \n" + str(ibias) + " amps")
        print(" ")
        
        print("Bias current uncertainty from IV: " + str(offset_dict['ibias_off_err']) + " amps")
        
    return np.absolute(ibias), ibias_err


def _get_v0(i0, i0_err, ibias, ibias_err, rsh, rp):
    """
    Gets and returns voltage and uncertainty in voltage across the
    TES at a given bias point.
    
    Parameters
    ----------
        
    i0 : float
        The current through the TES at a given bias point
        
    i0_err: float
        The uncertainty in the current through the TES at a given
        bias point
        
    ibias: float
        The bias current applied to the TES/shunt system
        
    ibias_err: float
        The uncertainty in the bias current applied to the TES/shunt
        system
        
    rsh: float
        The shunt resistance in ohms
        
    rp: float
        The parasitic resistance in ohms
        
    Returns
    -------
    v0 : float
        The calculated voltage across the TES in volts
        
    v0_err : float
        The calculated uncertainty in the voltage across the TES
        in volts
        
    """
    
    ibias = np.absolute(ibias)
    i0 = np.absolute(i0)
    vb = rsh * (ibias - i0) #voltage across the shunt resistor
    vb_err = rsh * np.sqrt(i0_err**2 + ibias_err**2)
    
    v0 = np.absolute(vb - rp * i0) #the voltage across the shunt resistor minus
                      #the voltage across the parasitic resistance
    v0_err = np.sqrt(vb_err**2 + rp**2 * i0_err**2)
    
    return v0, v0_err

def _get_r0(i0, i0_err, v0, v0_err):
    """
    Gets and returns the resistance and uncertainty in resistance
    of a TES at a given bias point
    
    Parameters
    ----------
        
    i0 : float
        The current through the TES at a given bias point
        
    i0_err: float
        The uncertainty in the current through the TES at a given
        bias point
        
    v0: float
        The voltage across the TES at the bias point
        
    v0_err: float
        The uncertainty in the voltage across the TES at the bias
        point
        
    Returns
    -------
    r0 : float
        The calculated resistance of the TES at the bias point
        in ohms
        
    r0_err : float
        The calculated uncertainty in the resistance of the TES at
        the bias point in ohms
        
    """
    
    r0 = np.absolute(v0/i0)
    r0_err = np.sqrt(v0_err**2 * i0**-2 + v0**2 * i0_err**2 * i0**-4)
    
    return r0, r0_err

def _get_p0(i0, i0_err, v0, v0_err):
    """
    Gets and returns the bias power and uncertainty in bias power
    of a TES at a given bias point
    
    Parameters
    ----------
        
    i0 : float
        The current through the TES at a given bias point
        
    i0_err: float
        The uncertainty in the current through the TES at a given
        bias point
        
    v0: float
        The voltage across the TES at the bias point
        
    v0_err: float
        The uncertainty in the voltage across the TES at the bias
        point
        
    Returns
    -------
    p0 : float
        The calculated bias power of the TES at the bias point
        in watts
        
    p0_err : float
        The calculated uncertainty in the bias power of the TES at
        the bias point in watts
        
    """
    
    p0 = np.absolute(v0*i0)
    p0_err = np.sqrt(v0_err**2 * i0**2 + v0**2 * i0_err**2)
    
    return p0, p0_err

def get_biasparams_i0(i0, i0_err, ibias, ibias_err, rsh, rp,
                    rn=None):
    """
    Gets and returns a dictonary of i0, v0, r0, and p0 with uncertainties
    from the measured TES bias current and i0
    
    Parameters
    ----------
        
    i0 : float
        The current through the TES at a given bias point
        
    i0_err: float
        The uncertainty in the current through the TES at a given
        bias point
        
    ibias: float
        The bias current applied to the TES/shunt system
        
    ibias_err: float
        The uncertainty in the bias current applied to the TES/shunt
        system
        
    rsh: float
        The shunt resistance in ohms
        
    rp: float
        The parasitic resistance in ohms

    rn: float (optional) 
        The normal resistance in ohms
        
    Returns
    -------
    bias_parameters_dict : dict
        A dictonary of bias parameters and uncertainties, including i0, r0,
        v0, and p0
        
    """
    
    i0 = np.absolute(i0)
    
    rl = rp + rsh
    
    v0, v0_err = _get_v0(i0, i0_err, ibias, ibias_err, rsh, rp)
    r0, r0_err = _get_r0(i0, i0_err, v0, v0_err)
    p0, p0_err = _get_p0(i0, i0_err, v0, v0_err)

    if rn is None:
        rn = np.nan

    
    bias_parameter_dict = {
        'i0': i0,
        'i0_err': i0_err,
        'v0': v0,
        'v0_err': v0_err,
        'r0': r0,
        'r0_err': r0_err,
        'p0': p0,
        'p0_err': p0_err,
        'rp': rp,
        'rsh': rsh,
        'rshunt': rsh,
        'rl': rl,
        'ibias': ibias,
        'rn': rn,
    }
    
    return bias_parameter_dict
    
def get_biasparams_ilg(params, cov,
                       ibias, ibias_err,
                       rsh, rp, rn=None):
    """
    Gets and returns a dictonary of i0, v0, r0, and p0 with uncertainties
    using the infinte loop gain approximation
    
    Parameters
    ----------
  
    params : dict
        The parameters (A, B, tau_1, etc.) of the previous dIdV fit.
               
    cov : matrix
        The covariance matrix for the params, starting with the A, B
        components.
        
    ibias: float
        The bias current applied to the TES/shunt system
        
    ibias_err: float
        The uncertainty in the bias current applied to the TES/shunt
        system
        
    rsh: float
        The shunt resistance in ohms
        
    rp: float
        The parasitic resistance in ohms
        
    Returns
    -------
    bias_parameters_dict : dict
        A dictonary of bias parameters and uncertainties, including i0, r0,
        v0, and p0
        
    """

    # check dimension
    num_params = cov.shape[0]
    if len(params.keys()) != num_params:
        raise ValueError('ERROR: inconsistent number of '
                         'parameters with covariance '
                         'matrix shape')
    
    # Rload
    rl = rp + rsh

    # r0
    dvdi0 = None
    r0_jac = None
    if num_params == 5:
        dvdi0 =  params['A'] +  params['B']
        r0_jac = np.asarray([1, 1, 0, 0, 0])
    else:
        dvdi0 =  params['A'] + params['B']/(1-params['C'])
        r0_jac = np.asarray([1, 1, 1, 0, 0, 0, 0])
        
    r0 = abs(dvdi0) + rl
    r0_err = np.matmul(r0_jac, np.matmul(cov, np.transpose(r0_jac)))
    
    i0 = ibias * rsh / (r0 + rl)
    i0_err = ((ibias_err * rsh / (r0 + rl))**2 + (r0_err * ibias * rsh * (rl + r0)**-2)**2)**0.5
    
    
    v0, v0_err = _get_v0(i0, i0_err, ibias, ibias_err, rsh, rp)
    p0, p0_err = _get_p0(i0, i0_err, v0, v0_err)
    
    if rn is None:
        rn = np.nan
 
    bias_parameter_dict = {
        'i0': i0,
        'i0_err': i0_err,
        'v0': v0,
        'v0_err': v0_err,
        'r0': r0,
        'r0_err': r0_err,
        'p0': p0,
        'p0_err': p0_err,
        'rp': rp,
        'rsh': rsh,
        'rshunt': rsh,
        'rl': rl,
        'ibias': ibias,
        'rn': rn,
    }
    
    return bias_parameter_dict





def get_biasparams_offsets(offset_data, offset_data_err, 
                           ibias_data, offset_dict, 
                           output_offset, closed_loop_norm, output_gain, rsh=5e-3, 
                           lgc_calibrated_offset_on=False):
    """
    Calculates the biasparams dict given a baseline value of
    meaured current through the TES, information about the 
    offsets applied to the current through the device, and
    measured offsets during the IV sweep as captured in
    an offsets dictionary. 
    
    Parameters
    ----------
    offset_data: float
        The average baseline value of measured current through the
        TES in units of amps.
        
    offset_err_data: float
        The uncertainty in the average baseline value of measured
        current through the TES in units of amps.
        
    ibias_data: float
        The nominal TES bias current (i.e. from the event metadata)
        
    offset_dict: dict
        A dictionary containing information about the offset between
        the measured and true current through the device and bias current.
        
    output_offset: float
        The output offset (e.g. as set on the FEB) applied to move the
        measured current through the device up and down. In units of volts.
    
    closed_loop_norm: float
        The normalization constant used to turn measured DAQ voltages into
        TES currents.
        
    output_gain: float
        The output gain, as e.g. given in the event metadata.
        
    rsh: float, optional
        The shunt resistance in Ohms. Defaults to 5 mohms.
        
    lgc_calibrated_offset_on: bool, optional
        Defaults to False. If True, uses the calibrated offsets approach.
        
    Returns
    -------
    biasparams : dict
        A dictionary of the TES bias parameters and uncertainties:
        i0, r0, v0, p0
        
    """
    
    if offset_dict['calibration_dict'] is not None:
        calibration_dict_ = offset_dict['calibration_dict']
    else:
        calibration_dict_ = None
    
    i0_offset, i0_offset_err = _get_i0_offset(offset_data, offset_data_err,
                                              offset_dict, output_offset=None,
                                              closed_loop_norm=None, output_gain=1,
                                              lgc_calibration_on=lgc_calibrated_offset_on,
                                              calibration_dict=calibration_dict_)
                                              
    ibias, ibias_err = _get_ibias_offset(ibias_data, offset_dict)
    
    rp = offset_dict['rp']
    
    biasparams = get_biasparams_i0(i0_offset, i0_offset_err, ibias, ibias_err,
                                   rsh, rp, rn=None)
    biasparams['biasparams_type'] = 'offset'       
    return biasparams
    

def get_biasparams_muon(baseline_av, baseline_err,
                        norm_av, norm_err,
                        rn, rn_err, ibias,
                        rl, rl_err, rsh=5e-3):
    """
    Calculates the biasparams dict given a baseline value of
    meaured current through the TES, a value of the measured
    current through the TES at the bias point when driven
    normal by e.g. a muon dumping tons of power into the detector,
    the bias current, and the normal and parasitic resistances.
    
    Parameters
    ----------
    baseline_av: float
        The average (as measured) current through the TES when in a
        baseline (i.e. in transition) state. In units of amps.
        
    baseline_err: float
        The uncertainty in the current through the TES when in a
        baseline (i.e. in transition) state. In units of amps.
        
    norm_av: float
        The average (as measured) current through the TES when normal.
        Note that this is still at the same bias point, the TES needs
        to have been driven normal by heating from e.g. a muon passing
        through the detetor. In units of amps.
        
    norm_err: float
        The uncertainty in the current through the TES when normal, 
        in units of amps.
        
    rn: float
        The device normal resistance in units of ohms.
        
    rn_err: float
        The uncertainty in the device normal resistance in ohms.
        
    ibias: float
        The nominal current applied to the TES/shunt system, in amps.
        
    rl: float
        The load (parasitic + shunt) resistance in units of ohms.
        
    rl_err: float
        The uncertainty in the load resistance in units of ohms.
        
    rsh: float, optional
        The shunt resistance, in ohms. Defaults to 5 mohms.
        
        
    Returns
    -------
    biasparams : dict
        A dictionary of the TES bias parameters and uncertainties:
        i0, r0, v0, p0
        
    """
    
    i_n = ibias * rsh/(rl + rn)
    i_n_err_sq = (ibias * rsh*(rl + rn)**-2)**2 * (rl_err**2 + rn_err**2)
           
    i0 = np.abs(i_n) + np.abs(baseline_av - norm_av)
    i0_err_sq = i_n_err_sq + baseline_err**2 + norm_err**2
    i0_err = np.sqrt(i0_err_sq)
    
    rp = rl - rsh
    
    biasparams = get_biasparams_i0(i0, i0_err, ibias, 0.0,
                                   rsh, rp, rn=None)
    biasparams['biasparams_type'] = 'muon'
    
    return biasparams
    
                        
def get_biasparams_normal_iv(normal_avs, normal_errs,
                             baseline_av, baseline_err,
                             ibias_norms, ibias_baseline,
                             rl, rl_err, rsh=5e-3,
                             lgc_diagnostics=False):
    """
    Calculates the biasparams dict given a normal IV set of points,
    a baseline (in transition) current, an array of biases
    
    Parameters
    ----------
    normal_avs: array
        The average (as measured) currents through the TES when during
        the normal IV sweep, in units of amps.
        
    normal_errs: array
        The uncertainty in the currents through the TES when during
        the normal IV sweep, in units of amps.
        
    baseline_av: float
        The average (as measured) current through the TES when in a
        baseline (i.e. in transition) state. In units of amps.
        
    baseline_err: float
        The uncertainty in the current through the TES when in a
        baseline (i.e. in transition) state. In units of amps.
        
    ibias_norms: array
        The nominal currents applied to the TES/shunt system
        during the normal IV sweep, in amps.
        
    ibias_baseline: float
        The nominal current applied to the TES/shunt system
        when in transition, in amps.
        
    rl: float
        The load (parasitic + shunt) resistance in units of ohms.
        
    rl_err: float
        The uncertainty in the load resistance in units of ohms.
        
    rsh: float, optional
        The shunt resistance, in ohms. Defaults to 5 mohms.
        
    lgc_diagnostics : bool, optional
        If True, prints out diagnostic messages.
        
        
    Returns
    -------
    biasparams : dict
        A dictionary of the TES bias parameters and uncertainties:
        i0, r0, v0, p0
        
    """
    
    def linear(x, m, b):
        return x*m + b
    
    popt, pcov = sp.optimize.curve_fit(linear, ibias_norms, normal_avs, sigma=normal_errs)
    
    m = popt[0]
    m_err = pcov[0][0]**0.5
    
    calc_rn = rsh/m - rl
    calc_rn_err_sq = m_err**2 * rsh**2 * m**-4 + rl_err**2
    calc_rn_err = np.sqrt(calc_rn_err_sq)
                              
    if lgc_diagnostics:
        print("Calculated Rn: " + str(calc_rn*1e3) + " +/- " + str(calc_rn_err*1e3) + " mOhms")
        
    i0 = np.abs(baseline_av - popt[1])
    i0_err_sq = baseline_err**2 + pcov[1][1]
    i0_err = np.sqrt(i0_err_sq)
    
    biasparams = get_biasparams_i0(i0, i0_err, ibias_baseline, 0.0,
                                   rsh, rl-rsh, rn=None)

    biasparams['biasparams_type'] = 'normal_iv'
    return biasparams
    
