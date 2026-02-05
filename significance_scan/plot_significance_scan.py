import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, approx_fprime
from scipy.stats import t

def poly(x, x0, A, B):
    return A * (x - x0)**2 + B

def confidence_band(model, popt, pcov, x_eval, xdata, ydata):
    n_data = len(xdata)
    n_params = len(popt)
    
    residuals = ydata - model(xdata, *popt)
    mse = np.sum(residuals**2) / (n_data - n_params)

    def model_p(p, x):
        return model(x, *p)

    jac = np.array([approx_fprime(popt, model_p, 1e-6, xi) for xi in x_eval])
    pred_variance = np.einsum("ij,jk,ik->i", jac, pcov, jac)

    t_score = t.ppf(0.5 + 0.6827 / 2.0, n_data - n_params)
    delta = t_score * np.sqrt(mse * pred_variance)

    y_pred = model(x_eval, *popt)
    return y_pred, y_pred - delta, y_pred + delta

def _apply_range_mask(data, key, range_tuple):
    if not range_tuple:
        return data
    
    mask = (data[key] >= range_tuple[0]) & (data[key] <= range_tuple[1])
    return {k: v[mask] for k, v in data.items()}

def _perform_fit(scores, sigs, sig_errs):
    finite_mask = np.isfinite(scores) & np.isfinite(sigs) & np.isfinite(sig_errs)
    
    if np.sum(finite_mask) < 3:
        return None

    _scores, _sigs, _sig_errs = scores[finite_mask], sigs[finite_mask], sig_errs[finite_mask]
    
    p0 = [np.mean(_scores), -1.0, np.mean(_sigs)]
    try:
        popt, pcov = curve_fit(poly, _scores, _sigs, sigma=_sig_errs, p0=p0, maxfev=20000)
        return popt, pcov, (_scores, _sigs)
    except RuntimeError:
        print("Fit failed, falling back to max from data points")
        return None

def significance_plotter(data, path, add_fitline=False, add_maxline=True, datarange=None, fitrange=None, show=False):
    plot_data = _apply_range_mask(data, 'score', datarange)
    scores, sigs, sig_errs = plot_data['score'], plot_data['significance'], plot_data['significance_err']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(scores, sigs, yerr=sig_errs, label='Scan Working Points', ls='none', marker='o')

    fit_results = None
    if add_fitline:
        fit_data = _apply_range_mask(plot_data, 'score', fitrange)
        fit_results = _perform_fit(fit_data['score'], fit_data['significance'], fit_data['significance_err'])
        
        if fit_results:
            popt, pcov, (fit_x, fit_y) = fit_results
            x_fit = np.linspace(fit_x.min(), fit_x.max(), 500)
            y_fit, y_low, y_up = confidence_band(poly, popt, pcov, x_fit, fit_x, fit_y)
            ax.plot(x_fit, y_fit, color='red', label='Parabolic Fit')
            ax.fill_between(x_fit, y_low, y_up, color='red', alpha=0.3, label=r'$\pm 1\sigma$ band')

    if add_maxline:
        if fit_results:
            popt, _, _ = fit_results
            x0_fit, y0_fit = popt[0], poly(popt[0], *popt)
            ax.axvline(x0_fit, color='red', linestyle='--', label=f'Optimized BDT Cut ({x0_fit:.3f})')
            ax.axhline(y0_fit, color='red', linestyle='--', label=f'Optimized Significance ({y0_fit:.2f})')
        elif len(sigs) > 0:
            max_idx = np.nanargmax(sigs)
            label_text = f'Optimized Significance ({sigs[max_idx]:.2f} \u00B1 {sig_errs[max_idx]:.2f})'
            ax.axvline(scores[max_idx], color='red', linestyle='--', label=f'Optimized BDT Cut ({scores[max_idx]:.3f})')
            ax.axhline(sigs[max_idx], color='red', linestyle='--', label=label_text)

    ax.set_xlabel('BDT Score', loc='right', fontsize=18)
    ax.set_ylabel(r'Signal Significance $(S / \sqrt{S+B})$', loc='top', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='lower right', fontsize=14)
    
    fig.savefig(path, bbox_inches='tight')
    if show:
        plt.show()

def main(args):
    if args.input_file:
        data_file = Path(args.input_file)
        assert data_file.is_file(), 'Cannot find data file'
    else:
        data_file = Path('.') / 'significance_scan_data.pkl'
        assert data_file.is_file(), 'Cannot find data file'

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path('.') / 'significance_scan.pdf'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    with open(data_file, 'rb') as f:
        score_data = pickle.load(f)

    significance_plotter(
        score_data,
        output_file,
        add_fitline=args.add_fitline,
        datarange=args.datarange,
        fitrange=args.fitrange
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='input_file', type=str, help='pickle data file')
    parser.add_argument('-o', '--output', dest='output', type=str, help='output file path')
    parser.add_argument('-l', '--label', dest='label', type=str, help='output file label')
    parser.add_argument('-fit', '--add-fitline', action='store_true', help='Fit a parabola to the significance scan')
    parser.add_argument('-dr', '--datarange', nargs=2, dest='datarange', default=None, type=float, help='range for data')
    parser.add_argument('-fr', '--fitrange',nargs=2, dest='fitrange', default=None, type=float, help='range for fitting function')
    args = parser.parse_args()

    main(args)
