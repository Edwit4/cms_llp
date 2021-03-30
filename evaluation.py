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

def llp_roc_curve(outputs, Ms, num_thresh=100):
    outputs = np.sort(outputs)

    def calc_fpr_tpr(outputs, threshold, M_lo, M_hi):

        # Use Freedman-Diaconis rule for optimal binning
        q75, q25 = np.percentile(Ms, [75 ,25])
        iqr = q75 - q25
        bin_width = 2*iqr*len(Ms)**(-1/3)
        num_bins = int((np.max(Ms) - np.min(Ms)) / bin_width)

        # Make histogram
        upper_outs = (outputs >= threshold)
        lower_outs = (outputs < threshold)
        upper_Ms = Ms[upper_outs]
        lower_Ms = Ms[lower_outs]
        bin_edges = np.linspace(M_lo,M_hi,num_bins+1)
        upper_M_hist, _ = np.histogram(upper_Ms,
                bins=bin_edges, density=True)
        lower_M_hist, _ = np.histogram(lower_Ms,
                bins=bin_edges, density=True)

        # Get values at centers of bins
        window_centered = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Perform fit
        # params in order: e_lam, e_const, v_sig, v_gam, v_mass, sig_frac
        # CMS
        initial_guess = [0.04, 30, 1.5, 1.15, 91, 0.9]
        bounds_lo = (0, 0, 0, 0, 90, 0)
        bounds_hi = (5, 100, 5, 5, 92, 1)
        #bounds_hi = (1, 100, 10, 1.5, 92, 1)
        # Sim
        #initial_guess = [0.15, -21, 5, 5, 24, 0.4]
        #bounds_lo = (0, -22, 0, 0, 23, 0)
        #bounds_hi = (.3, -20, 10, 10, 25, 1)
        if len(upper_Ms) > 0:
            upper_popt, upper_pcov = curve_fit(dist_exp_voigt, window_centered,
                    upper_M_hist, p0=initial_guess, bounds=(bounds_lo,
                        bounds_hi))
        else:
            upper_popt = np.full(6,np.nan)
            upper_pcov = np.diag(np.full(6,np.nan))

        if len(lower_Ms) > 0:
            lower_popt, lower_pcov = curve_fit(dist_exp_voigt, window_centered,
                    lower_M_hist, p0=initial_guess, bounds=(bounds_lo,
                        bounds_hi))
        else:
            lower_popt = np.full(6,np.nan)
            lower_pcov = np.diag(np.full(6,np.nan))

        upper_lam = upper_popt[0]
        upper_const = upper_popt[1]
        upper_sigma = upper_popt[2]
        upper_gamma = upper_popt[3]
        upper_mass = upper_popt[4]
        upper_frac  = upper_popt[5]

        lower_lam = lower_popt[0]
        lower_const = lower_popt[1]
        lower_sigma = lower_popt[2]
        lower_gamma = lower_popt[3]
        lower_mass = lower_popt[4]
        lower_frac  = lower_popt[5]

        #upper_exp_normalization = integrate.simps(
        #        dist_exponential(window_centered, upper_lam,
        #            upper_const), window_centered)
        #upper_voigt_normalization = integrate.simps(voigt_profile(
        #    window_centered-upper_mass, upper_sigma, upper_gamma),
        #    window_centered)
        #upper_exp_fit = (1-upper_frac)*dist_exponential(window_centered,
        #        upper_lam, upper_const) / upper_exp_normalization
        #upper_voigt_fit = upper_frac*voigt_profile(
        #        window_centered-upper_mass, upper_sigma,
        #        upper_gamma) / upper_voigt_normalization

        #lower_exp_normalization = integrate.simps(
        #        dist_exponential(window_centered, lower_lam,
        #            lower_const), window_centered)
        #lower_voigt_normalization = integrate.simps(voigt_profile(
        #    window_centered-lower_mass, lower_sigma, lower_gamma),
        #    window_centered)
        #lower_exp_fit = (1-lower_frac)*dist_exponential(window_centered,
        #        lower_lam, lower_const) / lower_exp_normalization
        #lower_voigt_fit = lower_frac*voigt_profile(window_centered-lower_mass, 
        #        lower_sigma, lower_gamma) / lower_voigt_normalization

        upper_perr = np.sqrt(np.diag(upper_pcov))
        lower_perr = np.sqrt(np.diag(lower_pcov))
        upper_count = np.sum(upper_outs)
        lower_count = np.sum(lower_outs)
        if np.isnan(upper_frac):
            upper_sig_count = 0
        else:
            upper_sig_count = np.floor(upper_frac * upper_count)
        if np.isnan(lower_frac):
            lower_sig_count = 0
        else:
            lower_sig_count = np.floor(lower_frac * lower_count)
        fpr = (upper_count - upper_sig_count) / (upper_count -
            upper_sig_count + lower_count - lower_sig_count)
        tpr = upper_sig_count / (upper_sig_count + lower_sig_count)

        roc_dict = {'fpr': fpr, 'tpr': tpr, 'lower params': lower_popt,
                'upper params': upper_popt, 'lower sigmas':
                lower_perr, 'upper sigmas': upper_perr,
                'lower Ms': lower_Ms, 'upper Ms': upper_Ms}

        return roc_dict

    def calc_auc(outputs, Ms, num_thresh):

        M_lo = np.min(Ms)
        M_hi = np.max(Ms)
        thresholds = np.linspace(
                np.min(outputs),np.max(outputs),num_thresh)
        #thresholds = np.concatenate([np.unique(outputs),
        #    [np.max(outputs)+1]])

        roc_dict = {'fpr':[], 'tpr':[], 'lower params':[], 
            'upper params':[], 'lower sigmas':[], 'upper sigmas':[], 
            'lower Ms':[], 'upper Ms':[]}
        for i,t in enumerate(thresholds):
            print(f'threshold {i}/{len(thresholds)}\r',end='')
            t_dict = calc_fpr_tpr(outputs, t, M_lo, M_hi)
            for key in t_dict:
                roc_dict[key].append(t_dict[key])
            #if t in thresholds[-1:-6:-1]:
            #    # Use Freedman-Diaconis rule for optimal binning
            #    ms = Ms[outputs>=t]
            #    q75, q25 = np.percentile(ms, [75 ,25])
            #    iqr = q75 - q25
            #    bin_width = 2*iqr*len(ms)**(-1/3)
            #    num_bins = int((np.max(ms) - np.min(ms)) / bin_width)
            #    plt.cla()
            #    plt.hist(ms,bins=num_bins)
            #    plt.title(f'Invariant Mass for Output >= {t}')
            #    plt.xlabel('M (GeV)')
            #    plt.ylabel('Count')
            #    plt.savefig(f'mass_dist_{t}.png')

        for key in roc_dict:
            roc_dict[key] = np.array(roc_dict[key])
        
        sort = np.argsort(roc_dict['fpr'])
        for key in roc_dict.keys():
            roc_dict[key] = roc_dict[key][sort]

        auc = metrics.auc(roc_dict['fpr'], roc_dict['tpr'])
        #plt.cla()
        #plt.plot(roc_dict['fpr'],roc_dict['tpr'])
        #plt.show()

        roc_dict['auc'] = auc
        roc_dict['threshold'] = thresholds[sort]
        roc_dict['all out'] = outputs
        roc_dict['all Ms'] = Ms

        return roc_dict

    roc_dict = calc_auc(outputs, Ms, num_thresh)
    print(roc_dict['auc'])
    if roc_dict['auc'] < 0.5:
        roc_dict = calc_auc(1 - outputs, Ms, num_thresh)
        print(roc_dict['auc'])

    return roc_dict

def llp_roc_curve_test(outputs, Ms, approx_num_thresh=1000, seed=123):

    def get_num_bins(Ms):

        if len(Ms) == 1:
            num_bins = 1
        else:
            # Use Freedman-Diaconis rule for optimal binning
            q75, q25 = np.percentile(Ms, [75 ,25])
            iqr = q75 - q25
            bin_width = 2*iqr*len(Ms)**(-1/3)
            num_bins = int((np.max(Ms) - np.min(Ms)) / bin_width)

        return num_bins

    def fit_Ms(Ms):

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
            # Sim
            #initial_guess = [0.15, -21, 5, 5, 24, 0.4]
            #bounds_lo = (0, -22, 0, 0, 23, 0)
            #bounds_hi = (.3, -20, 10, 10, 25, 1)

            popt, pcov = curve_fit(dist_exp_voigt,
                    window_centered, M_hist,
                    p0=initial_guess, bounds=(bounds_lo, bounds_hi),
                    maxfev=5000)
        else:
            popt = np.full(6,np.nan)
            pcov = np.diag(np.full(6,np.nan))

        perr = np.sqrt(np.diag(pcov))

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

    def calc_upper_sig(outputs, threshold, Ms):

        upper_outs = (outputs >= threshold)
        lower_outs = (outputs < threshold)
        upper_Ms = Ms[upper_outs]
        lower_Ms = Ms[lower_outs]

        upper_fit_dict = fit_Ms(upper_Ms)
        lower_fit_dict = fit_Ms(lower_Ms)

        upper_frac = upper_fit_dict['params']['frac']
        lower_frac = lower_fit_dict['params']['frac']

        upper_count = np.sum(upper_outs)
        lower_count = np.sum(lower_outs)

        if np.isnan(upper_frac):
            upper_sig_count = 0
        else:
            upper_sig_count = np.round(upper_frac * upper_count)
        if np.isnan(lower_frac):
            lower_sig_count = 0
        else:
            lower_sig_count = np.round(lower_frac * lower_count)

        count_dict = {'upper sig count': upper_sig_count,
                'lower sig count': lower_sig_count,
                'upper count': upper_count,
                'lower count': lower_count}

        roc_dict = {'count dict': count_dict, 'lower fit dict':
                lower_fit_dict, 'upper fit dict': upper_fit_dict, 
                'lower Ms': lower_Ms, 'upper Ms': upper_Ms}

        return roc_dict

    def calc_fpr_tpr(count_dict, thresholds, pos_count, neg_count):

        upper_sig_counts = np.array(count_dict['upper sig count'])
        lower_sig_counts = np.array(count_dict['lower sig count'])

        #counts_ir = IsotonicRegression(increasing=False,out_of_bounds='clip')
        #diffs_ir = IsotonicRegression(increasing=False,out_of_bounds='clip')

        #sig_counts_mono = counts_ir.fit_transform(thresholds, upper_sig_counts)
        #diffs_mono = diffs_ir.fit_transform(thresholds,
        #        upper_counts-sig_counts_mono)

        #fpr = diffs_mono / neg_count
        #tpr = sig_counts_mono / pos_count

        #fpr = (upper_count - upper_sig_count) / (upper_count -
        #    upper_sig_count + lower_count - lower_sig_count)
        #tpr = upper_sig_count / (upper_sig_count + lower_sig_count)
        upper_counts = np.array(count_dict['upper count'])
        lower_counts = np.array(count_dict['lower count'])
        fpr_ir = IsotonicRegression(increasing=False,
                out_of_bounds='clip')
        tpr_ir = IsotonicRegression(increasing=False,
                out_of_bounds='clip')
        fpr = (upper_counts - upper_sig_counts) / (upper_counts -
            upper_sig_counts + lower_counts - lower_sig_counts)
        tpr = upper_sig_counts / (upper_sig_counts + lower_sig_counts)
        fpr = fpr_ir.fit_transform(thresholds, fpr)
        tpr = tpr_ir.fit_transform(thresholds, tpr)

        return fpr, tpr

    def calc_auc(outputs, Ms, approx_num_thresh, pos_count, neg_count):

        M_lo = np.min(Ms)
        M_hi = np.max(Ms)
        unique_thresh = np.unique(outputs)
        np.random.seed(seed)
        rands = np.array([np.random.uniform() for t in unique_thresh])
        rand_bools = (rands <= (approx_num_thresh / len(unique_thresh)))
        thresholds = np.concatenate([unique_thresh[rand_bools],
            [np.max(unique_thresh[rand_bools])+1]])
        #thresholds = np.linspace(
        #        np.min(outputs),np.max(outputs),approx_num_thresh)
        #thresholds = np.concatenate([np.unique(outputs),
        #    [np.max(outputs)+1]])

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
        print('Calculating AUC')
        for i,t in enumerate(thresholds):
            print(f'threshold {i}/{len(thresholds)}\r',end='')
            t_dict = calc_upper_sig(outputs, t, Ms)
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
            #if t in thresholds[-1:-6:-1]:
            #    # Use Freedman-Diaconis rule for optimal binning
            #    ms = Ms[outputs>=t]
            #    q75, q25 = np.percentile(ms, [75 ,25])
            #    iqr = q75 - q25
            #    bin_width = 2*iqr*len(ms)**(-1/3)
            #    num_bins = int((np.max(ms) - np.min(ms)) / bin_width)
            #    plt.cla()
            #    plt.hist(ms,bins=num_bins)
            #    plt.title(f'Invariant Mass for Output >= {t}')
            #    plt.xlabel('M (GeV)')
            #    plt.ylabel('Count')
            #    plt.savefig(f'mass_dist_{t}.png')

        for key in ['lower Ms', 'upper Ms']:
            roc_dict[key] = np.array(roc_dict[key])

        fpr, tpr = calc_fpr_tpr(roc_dict['count dict'], thresholds,
                pos_count, neg_count)
        
        #print(np.any(np.diff(roc_dict['fpr'])>0))
        #print(np.any(np.diff(roc_dict['tpr'])>0))
        #sort = np.argsort(roc_dict['fpr'])
        #for key in roc_dict.keys():
        #    roc_dict[key] = roc_dict[key][sort]
        #plt.cla()
        #plt.plot(fpr,thresholds,label='fpr')
        #plt.plot(tpr,thresholds,label='tpr')
        #plt.xlabel('Threshold')
        #plt.ylabel('Rate')
        #plt.title('Change In Rates With Threshold')
        #plt.legend()
        #plt.savefig('fprtpr_vs_thresh.png')
        #plt.show()

        auc = metrics.auc(fpr, tpr)
        #plt.cla()
        #plt.plot(fpr, tpr)
        #plt.show()

        roc_dict['fpr'] = fpr
        roc_dict['tpr'] = tpr
        roc_dict['auc'] = auc
        roc_dict['threshold'] = thresholds
        roc_dict['all out'] = outputs
        roc_dict['all Ms'] = Ms

        return roc_dict

    fit_dict = fit_Ms(Ms)
    pos_count = np.round(len(Ms) * fit_dict['params']['frac'])
    neg_count = len(Ms) - pos_count

    roc_dict = calc_auc(outputs, Ms, approx_num_thresh, pos_count, neg_count)
    print(f'LLP AUC: {roc_dict["auc"]}')
    if roc_dict['auc'] < 0.5:
        print('AUC below 0.5, inverting output and trying again')
        roc_dict = calc_auc(1 - outputs, Ms, approx_num_thresh, pos_count,
                neg_count)
        print(f'LLP AUC: {roc_dict["auc"]}')

    return roc_dict

def plot_M_fit(Ms, params, filename, num_bins=75):
    M_lo = np.min(Ms)
    M_hi = np.max(Ms)
    bin_edges = np.linspace(M_lo,M_hi,num_bins+1)
    M_hist, _ = np.histogram(Ms, bins=bin_edges, density=True)
    window_centered = (bin_edges[:-1] + bin_edges[1:]) / 2

    lam = params[0]
    const = params[1]
    sigma = params[2]
    gamma = params[3]
    mass = params[4]
    frac  = params[5]

    exp_normalization = integrate.simps(dist_exponential(window_centered,
        lam, const), window_centered)
    voigt_normalization = integrate.simps(voigt_profile(
        window_centered-mass, sigma, gamma),
        window_centered)

    plt.cla()
    plt.plot(window_centered, (1-frac)*dist_exponential(window_centered,
        lam, const) / exp_normalization, color='cyan',
        label='exponential')
    plt.plot(window_centered,
            frac*voigt_profile(window_centered-mass,
        sigma, gamma) / voigt_normalization, color='magenta',
        label='voigt')
    plt.hist(Ms,density=True,bins=num_bins)
    plt.xlabel('M')
    plt.ylabel('density')
    plt.title(f'invariant mass')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_llp_ROC_curve(fprs, tprs, filename):
    auc = metrics.auc(fprs,tprs)
    plt.cla()
    plt.plot(fprs,tprs)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'ROC Curve, AUC = {auc:.3f}')
    plt.savefig(filename)
    plt.close()

def plot_sig_frac(thresholds, sig_fracs, filename):
    plt.cla()
    plt.plot(thresholds, sig_fracs)
    plt.xlabel('Threshold')
    plt.ylabel('Signal Fraction')
    plt.title('Thresholded Signal Fraction')
    plt.savefig(filename)
    plt.close()

def plot_param_sigmas(thresholds, sigmas, filename):
    plt.cla()
    plt.plot(thresholds,sigmas)
    plt.xlabel('Threshold')
    plt.ylabel('Sigma')
    plt.title('Signal Fraction Error')
    plt.savefig(filename)
    plt.close()

def plot_ROC_curve(labels, outputs, filename):
    fpr, tpr, thresh = metrics.roc_curve(labels, outputs)
    area = metrics.auc(fpr, tpr)
    plt.cla()
    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve, AUC = {area:.3f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(filename)
    plt.close()

    return fpr, tpr, thresh, area

def plot_AUC_history(auc_history, auc_filename):
    plt.cla()
    plt.plot(np.arange(len(auc_history))+1, auc_history)
    plt.title('Validation AUC During Training')
    plt.xlabel('epoch')
    plt.ylabel('AUC')
    plt.savefig(auc_filename)
    plt.close()

def plot_loss_histories(train_loss_history, valid_loss_history, filename,
        early_stopping=None):
    plt.cla()
    plt.plot(np.arange(len(train_loss_history))+1,
            train_loss_history, label='training loss')
    plt.plot(np.arange(len(valid_loss_history))+1,
            valid_loss_history, label='validation loss')
    if early_stopping is not None:
        plt.axvline(x=early_stopping, label='early stopping',
                color='r', linestyle='dashed')
    plt.title('Loss Histories')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_pred_distribution(labels, outputs, filename):
    num_bins = 100
    pred_on_signal = outputs[labels==1]
    pred_on_bg = outputs[labels==0]
    _, prefix, suffix = filename.split('.')
    log_prefix = prefix + '_log'
    log_filename = log_prefix + '.' + suffix
    plt.cla()
    plt.hist(pred_on_signal, bins=num_bins, label='signal', alpha=0.5,
            range=(0,1))
    plt.hist(pred_on_bg, bins=num_bins, label='bg', alpha=0.5,
            range=(0,1))
    plt.title('Output Distribution')
    plt.xlabel('output')
    plt.ylabel('count')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    plt.cla()
    plt.hist(pred_on_signal, bins=num_bins, label='signal', alpha=0.5,
            range=(0,1))
    plt.hist(pred_on_bg, bins=num_bins, label='bg', alpha=0.5,
            range=(0,1))
    plt.yscale('log',nonpositive='clip')
    plt.title('Output Distribution')
    plt.xlabel('output')
    plt.ylabel('count')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_bag_histories(bag_histories, filename):
    plt.cla()
    for key in bag_histories.keys():
        plt.plot(np.arange(len(bag_histories[key]))+1,
                bag_histories[key], label= 'f = '+key[:5])
    plt.title('Bag Loss Histories')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def make_dist(data,num_bins,label,title,filename,log=False,weights=None):
    if weights is None:
        weights = np.ones(len(data))
    fig,ax = plt.subplots()
    ax.hist(data,bins=num_bins,weights=weights)#,density=True)
    if log:
        ax.set_yscale('log',nonpositive='clip')
    ax.set_title(title)
    ax.set_xlabel(label)
    #ax.set_ylabel('density')
    ax.set_ylabel('Count')
    textstr = '\n'.join((r'$\mu=%.2f$' % (np.mean(data), ),
        r'$\mathrm{median}=%.2f$' % (np.median(data), ),
        r'$\sigma=%.2f$' % (np.std(data), ),
        r'$\mathrm{count}=%.0f$' % (len(data), ) ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right',
            bbox=props)
    #ax.figure.savefig(filename)
    #plt.close(fig)
    plt.show()
