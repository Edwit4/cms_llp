import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics

# Exponential distribution
def dist_exponential(x, lam, const):
    f = np.exp(-lam*(x+const))
    return f

# Exponential + Voigt distribution
def dist_exp_voigt(x, lam, const, sigma, gamma,
        mass, sig_frac):
    exp = dist_exponential(x, lam, const)
    voigt = voigt_profile(x - mass, sigma, gamma)
    voigt_normalization = integrate.simps(voigt, x)
    voigt_normed = voigt / voigt_normalization
    exp_normalization = integrate.simps(exp, x)
    exp_normed = exp / exp_normalization
    return ((1-sig_frac)*exp_normed + sig_frac*voigt_normed)

def llp_roc_curve(outputs, Ms, num_thresh=1000, seed=123,
        constant_denom=True, output_thresh=True):
    """
    Estimate AUC by creating a ROC curve using invariant mass fits.

    Parameters
    ----------
    outputs: Network outputs
    Ms: Invariant masses
    num_thresh: Number of thresholds to use. Exact if output_thresh is
        False, approximate if True. If -1 use the number of unique
        outputs.
    seed: Numpy seed
    constant_denom: Calculate FPR, TPR denominators once from overall
        invariant mass fit if True, otherwise recalculate from using fits
        on each side of the threshold at each threshold.
    output_thresh: Use unique outputs as thresholds if True taking
        approximately num_thresh outputs as thresholds, otherwise
        uniformly space thresholds using exactly num_thresh different
        thresholds.

    Returns
    -------
    ROC_dict: Dict containing thresholds, counts / masses on each side
        for each threshold, fit information for thresholds, FPRs, TPRs,
        and AUC.
    """

    def get_num_bins(Ms):
        """
        Choose number of bins given list of invariant masses.

        Parameters
        ----------
        Ms: Invariant masses

        Returns
        -------
        num_bins: Number of bins for mass histogram
        """

        if len(Ms) == 1:
            num_bins = 1
        else:
            # Use Freedman-Diaconis rule for optimal binning. Seemed
            # reasonable for choosing number of bins, possible that some
            # better method exists, though binning method shouldn't
            # change things much.
            q75, q25 = np.percentile(Ms, [75 ,25])
            iqr = q75 - q25
            bin_width = 2*iqr*len(Ms)**(-1/3)
            num_bins = int((np.max(Ms) - np.min(Ms)) / bin_width)

        return num_bins

    def fit_Ms(Ms):
        """
        Fit voigt and exponential functions to invariant mass
        distribution.

        Parameters
        ----------
        Ms: Invariant masses

        Returns
        -------
        fit_dict: Dict containing fit parameters
        """

        if len(Ms) > 0:
            # Calc number of bins
            num_bins = get_num_bins(Ms)
            if (num_bins < 2) and (len(Ms) < 2):
                num_bins = 2
            elif (num_bins < 2) and (len(Ms) >= 2):
                num_bins = len(Ms)

            # Make histogram
            M_hist, bin_edges = np.histogram(Ms, bins=num_bins, density=True)

            # Get values at centers of bins
            window_centered = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Perform fit
            # params in order: e_lam, e_const, v_sig, v_gam, v_mass, sig_frac
            # CMS
            initial_guess = [0.04, 0, 1.5, 1.15, 91, 0.9]
            bounds_lo = (0, -100, 0, 0, 90, 0)
            bounds_hi = (5, 100, 100, 5, 92, 1)

            popt, pcov = curve_fit(dist_exp_voigt,
                    window_centered, M_hist,
                    p0=initial_guess, bounds=(bounds_lo, bounds_hi),
                    maxfev=5000)
        else:
            # If no Ms then fill with nans
            popt = np.full(6,np.nan)
            pcov = np.diag(np.full(6,np.nan))

        perr = np.sqrt(np.diag(pcov))

        # Organize parameters into dict
        fit_dict = {'params':
                {'lambda': popt[0],
                'const': popt[1],
                'sigma': popt[2],
                'gamma': popt[3],
                'mass': popt[4],
                'frac': popt[5]},

                'sigmas':
                {'lambda': perr[0],
                'const': perr[1],
                'sigma': perr[2],
                'gamma': perr[3],
                'mass': perr[4],
                'frac': perr[5]}}

        return fit_dict

    def calc_thresh_values(outputs, threshold, Ms):
        """
        Calculate fit parameters and counts for a given threshold.

        Parameters
        ----------
        outputs: Networks outputs
        threshold: Threshold for which to calculate values above and below
        Ms: Invariant masses

        Returns
        -------
        roc_dict: Dict containing counts, fit information, and masses
            with respect to the given threshold
        """

        # Get masses corresponding to outputs on either side of the
        # threshold
        upper_outs = (outputs >= threshold)
        lower_outs = (outputs < threshold)
        upper_Ms = Ms[upper_outs]
        lower_Ms = Ms[lower_outs]

        # Perform fit
        upper_fit_dict = fit_Ms(upper_Ms)
        lower_fit_dict = fit_Ms(lower_Ms)

        # Get signal fractions above and below the threshold
        upper_frac = upper_fit_dict['params']['frac']
        lower_frac = lower_fit_dict['params']['frac']

        # Get number of samples above and below threshold
        upper_count = np.sum(upper_outs)
        lower_count = np.sum(lower_outs)

        # Use fraction to calculate number of signal samples on either
        # side of the threshold (nan returned if no masses are on that
        # side of the threshold)
        if np.isnan(upper_frac):
            upper_sig_count = 0
        else:
            upper_sig_count = np.round(upper_frac * upper_count)
        if np.isnan(lower_frac):
            lower_sig_count = 0
        else:
            lower_sig_count = np.round(lower_frac * lower_count)

        # Organize dicts
        count_dict = {'upper sig count': upper_sig_count,
                'lower sig count': lower_sig_count,
                'upper count': upper_count,
                'lower count': lower_count}

        roc_dict = {'count dict': count_dict, 'lower fit dict':
                lower_fit_dict, 'upper fit dict': upper_fit_dict, 
                'lower Ms': lower_Ms, 'upper Ms': upper_Ms}

        return roc_dict

    def calc_fpr_tpr(count_dict, thresholds, pos_count, neg_count):
        """
        Calculate FPR and TPR, using isotonic regression to ensure
        monotonicity.

        Parameters
        ----------
        count_dict: Dict containing information for counts with respect
            to thresholds.
        thresholds: Thresholds for ROC curve.
        pos_count: Total positive sample count from overall fit.
        neg_count: Total negative sample count from overall fit.

        Returns
        -------
        fpr: FPR values from thresholds, made monotonic with isotonic
            regression.
        tpr: TPR values from thresholds, made monotonic with isotonic
            regression.
        """

        # Get signal counts above and below thresholds
        upper_sig_counts = np.array(count_dict['upper sig count'])
        lower_sig_counts = np.array(count_dict['lower sig count'])

        # Get total sample counts above and below thresholds
        upper_counts = np.array(count_dict['upper count'])
        lower_counts = np.array(count_dict['lower count'])

        # Prep isotonic regression
        fpr_ir = IsotonicRegression(increasing=False,
                out_of_bounds='clip')
        tpr_ir = IsotonicRegression(increasing=False,
                out_of_bounds='clip')

        # Calculate FPR, TPR with denominators based on parameter setting
        if constant_denom:
            fpr = (upper_counts - upper_sig_counts) / neg_count
            tpr = upper_sig_counts / pos_count
        else:
            fpr = (upper_counts - upper_sig_counts) / (upper_counts -
                upper_sig_counts + lower_counts - lower_sig_counts)
            tpr = upper_sig_counts / (upper_sig_counts + lower_sig_counts)

        # Perform isotonic regression
        fpr = fpr_ir.fit_transform(thresholds, fpr)
        tpr = tpr_ir.fit_transform(thresholds, tpr)

        return fpr, tpr

    def calc_auc(outputs, Ms, num_thresh, pos_count, neg_count):
        """
        Estimate AUC from network outputs and invariant masses.

        Parameters
        ----------
        outputs: Network outputs
        Ms: Invariant masses
        num_thresh: Number (or approximate number) of thresholds to use.
            If -1 use number of unique outputs.
        pos_count: Total number of positive samples from overall fit.
        neg_count: Total number of negative samples from overall fit.

        Returns
        -------
        ROC_dict: Dict containing thresholds, counts / masses on each side
            for each threshold, fit information for thresholds, FPRs, TPRs,
            and AUC.
        """

        # Get unique outputs, if num_thresh is -1 use all unique outputs
        # as thresholds
        unique_thresh = np.unique(outputs)
        if num_thresh == -1:
            num_thresh = len(unique_thresh)

        # Choose thresholds based on parameter setting
        if output_thresh:
            # Choose approximately num_thresh number of values from
            # unique outputs to use as thresholds.
            np.random.seed(seed)
            rands = np.array([np.random.uniform() for t in unique_thresh])
            rand_bools = (rands <= (num_thresh / len(unique_thresh)))
            thresholds = np.concatenate([unique_thresh[rand_bools],
                [np.max(unique_thresh[rand_bools])+1]])
        else:
            # Uniformly space thresohlds from minimum to maximum outputs
            thresholds = np.linspace(
                    np.min(outputs),np.max(outputs),num_thresh)

        # Initialize empty dictionary
        roc_dict = {'count dict': {'upper sig count': [],
            'lower sig count': [],
            'upper count': [],
            'lower count': []}, 
                'lower fit dict': {'params':
                    {'lambda': [],
                    'const': [],
                    'sigma': [],
                    'gamma': [],
                    'mass': [],
                    'frac': []},

                    'sigmas':
                    {'lambda': [],
                    'const': [],
                    'sigma': [],
                    'gamma': [],
                    'mass': [],
                    'frac': []}},
                'upper fit dict': {'params':
                    {'lambda': [],
                    'const': [],
                    'sigma': [],
                    'gamma': [],
                    'mass': [],
                    'frac': []},

                    'sigmas':
                    {'lambda': [],
                    'const': [],
                    'sigma': [],
                    'gamma': [],
                    'mass': [],
                    'frac': []}},
                'lower Ms': [], 'upper Ms': []}

        # Loop over thresholds
        print('Calculating AUC')
        for i,t in enumerate(thresholds):
            print(f'threshold {i}/{len(thresholds)}\r',end='')

            # Calculate values for current threshold  
            t_dict = calc_thresh_values(outputs, t, Ms)

            # Put values in appropriate lists in dict
            for key in ['lower Ms', 'upper Ms']:
                roc_dict[key].append(t_dict[key])
            for parsig_key in ['params', 'sigmas']:
                for key in roc_dict['lower fit dict'][parsig_key]:
                    roc_dict['lower fit dict'][parsig_key][key].append(
                            t_dict['lower fit dict'][parsig_key][key])
            for parsig_key in ['params', 'sigmas']:
                for key in roc_dict['upper fit dict'][parsig_key]:
                    roc_dict['upper fit dict'][parsig_key][key].append(
                            t_dict['upper fit dict'][parsig_key][key])
            for key in roc_dict['count dict']:
                roc_dict['count dict'][key].append(
                        t_dict['count dict'][key])

        # Ensure invariant mass lists are numpy arrays
        for key in ['lower Ms', 'upper Ms']:
            roc_dict[key] = np.array(roc_dict[key])

        # Calculate FPRs + TPRs
        fpr, tpr = calc_fpr_tpr(roc_dict['count dict'], thresholds,
                pos_count, neg_count)
        
        # Calculate AUC
        auc = metrics.auc(fpr, tpr)

        # Store more values in dict
        roc_dict['fpr'] = fpr
        roc_dict['tpr'] = tpr
        roc_dict['auc'] = auc
        roc_dict['threshold'] = thresholds
        roc_dict['all out'] = outputs
        roc_dict['all Ms'] = Ms

        return roc_dict

    # Get positive + negative sample counts from overall invariant mass
    # fit
    fit_dict = fit_Ms(Ms)
    pos_count = np.round(len(Ms) * fit_dict['params']['frac'])
    neg_count = len(Ms) - pos_count

    # Calculate AUC + related values
    roc_dict = calc_auc(outputs, Ms, num_thresh, pos_count, neg_count)
    print(f'LLP AUC: {roc_dict["auc"]}')

    # Invert outputs and calculate again if AUC < 0.5 (there's probably
    # an issue if this is getting called)
    if roc_dict['auc'] < 0.5:
        print('AUC below 0.5, inverting output and trying again')
        roc_dict = calc_auc(1 - outputs, Ms, num_thresh, pos_count,
                neg_count)
        print(f'LLP AUC: {roc_dict["auc"]}')

    return roc_dict


