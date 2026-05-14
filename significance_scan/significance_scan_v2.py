import os
import sys
import yaml
import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from uncertainties import ufloat

sys.path.insert(1, str(Path('..').resolve()))
from fit_models import FitModel
from do_mc_yields_and_effs import get_eff
from do_fit import do_lowq2_signal_region_fit, do_jpsi_control_region_fit

# Branching ratios from PDG
BR_BKEE = 4.5e-7
BR_BJPSI = 1.02e-3
BR_JPSIEE = 5.97e-2


def loop_wrapper(iterable, args, unit='working point', title=None):
    if title:
        print(title)
    return tqdm(iterable, unit=unit, disable=args.verbose)


def remove_intermediate_plots(path_str):
    plot_path = Path(path_str)
    if plot_path.is_dir():
        [p.unlink() for p in plot_path.iterdir() if p.is_file() and not p.match('*_final_*.pdf')]


def estimate_lowq2_signal(n_jpsik, eff_eek, eff_jpsik, n_jpsik_err=None, eff_eek_err=None, eff_jpsik_err=None):
    errors_provided = all(e is not None for e in [n_jpsik_err, eff_eek_err, eff_jpsik_err])
    
    if not errors_provided:
        try:
            n_signal = n_jpsik * BR_BKEE * eff_eek / (BR_BJPSI * BR_JPSIEE * eff_jpsik)
            return n_signal, 0
        except ZeroDivisionError:
            return 0, 0

    try:
        _n_jpsik = ufloat(n_jpsik, n_jpsik_err)
        _eff_eek = ufloat(eff_eek, eff_eek_err)
        _eff_jpsik = ufloat(eff_jpsik, eff_jpsik_err)

        n_signal_ufloat = _n_jpsik * BR_BKEE * _eff_eek / (BR_BJPSI * BR_JPSIEE * _eff_jpsik)
        return n_signal_ufloat.nominal_value, n_signal_ufloat.std_dev
    except ZeroDivisionError:
        return 0, 0

def estimate_significance(n_sig, n_bkg, n_sig_err=None, n_bkg_err=None):
    try:
        significance = n_sig / np.sqrt(n_sig + n_bkg)
    except (ZeroDivisionError, ValueError):
        significance = 0

    if n_sig_err is None or n_bkg_err is None:
        return significance, 0

    try:
        s_plus_b = n_sig + n_bkg
        numerator = n_sig**2 * n_bkg_err**2 + n_sig_err**2 * (2 * n_bkg + n_sig)**2
        significance_err = 0.5 * np.sqrt(numerator / s_plus_b**3)
    except (ZeroDivisionError, ValueError):
        significance_err = 0
        
    return significance, significance_err


def significance_scan(dataset_params, output_params, fit_params, args):
    output_keys = [
        'score', 
        'significance', 'significance_err',
        'n_eek_bkg', 'n_eek_bkg_err', 
        'n_eek_sig', 'n_eek_sig_err',
        'n_jpsik_sig', 'n_jpsik_sig_err', 
        'eff_eek', 'eff_eek_err',
        'eff_jpsik', 'eff_jpsik_err'
    ]
    outputs = {key: [] for key in output_keys}

    # scan_range = [5, 6]
    scan_range = np.arange(0.99, 0.999, 0.001)

    for bdt_cut in loop_wrapper(scan_range, args, title='Calculating Significances'):
        print("bdt score: ", bdt_cut)
        fit_params.bdt_score_cut = bdt_cut

        # Calculate lowq2 MC eff for bdt cut in mass window
        args.mode = 'lowq2'
        output_params.output_dir = './scan_fits/lowq2' 
        cut_string_lowq2 = f'(Mll > 1.05 && Mll < 2.45) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score > {bdt_cut}'
        eff_eek, eff_eek_err = get_eff(
            dataset_params.rare_file, dataset_params, output_params, 
            fit_params, args, cut_string=cut_string_lowq2)

        # Fit bkg-only lowq2
        lowq2_yields = do_lowq2_signal_region_fit(
            dataset_params, output_params, fit_params, args, 
            get_yields=True, 
            file_label=f'bdt>{str(bdt_cut).replace(".", "p")}',
            write=False, 
            toy_fit=False)
        remove_intermediate_plots(output_params.output_dir)

        # Calculate jpsi MC eff for bdt cut in mass window
        args.mode = 'jpsi'
        output_params.output_dir = './scan_fits/jpsi' 
        cut_string_jpsi = f'(Mll > 2.95 && Mll < 3.2) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score > {bdt_cut}'
        eff_jpsik, eff_jpsik_err = get_eff(dataset_params.jpsi_file, dataset_params, output_params, fit_params, args, cut_string=cut_string_jpsi)

        # Fit jpsi signal
        jpsi_sig_window = {'yield_sig': [5.1, 5.4]}
        jpsi_yields = do_jpsi_control_region_fit(
            dataset_params, output_params, fit_params, args,
            get_yields=True, 
            custom_yield_ranges=jpsi_sig_window,
            file_label=f'bdt>{str(bdt_cut).replace(".", "p")}',
            write=False, splot=False)
        remove_intermediate_plots(output_params.output_dir)

        _n_lowq2_bkg = (
            ufloat(*lowq2_yields['yield_comb_bkg']) +
            ufloat(*lowq2_yields['yield_part_bkg']) +
            ufloat(*lowq2_yields['yield_jpsi_bkg'])
        )
        n_lowq2_bkg, n_lowq2_bkg_err = _n_lowq2_bkg.n, _n_lowq2_bkg.s
        n_jpsi_sig, n_jpsi_sig_err = jpsi_yields['yield_sig']

        n_lowq2_sig, n_lowq2_sig_err = estimate_lowq2_signal(
            n_jpsi_sig, eff_eek, eff_jpsik,
            n_jpsik_err=n_jpsi_sig_err, eff_eek_err=eff_eek_err, eff_jpsik_err=eff_jpsik_err
        )
        significance, significance_err = estimate_significance(
            n_lowq2_sig, n_lowq2_bkg, n_sig_err=n_lowq2_sig_err, n_bkg_err=n_lowq2_bkg_err
        )

        results = {
            'score': bdt_cut, 
            'significance': significance, 'significance_err': significance_err,
            'n_eek_bkg': n_lowq2_bkg, 'n_eek_bkg_err': n_lowq2_bkg_err,
            'n_eek_sig': n_lowq2_sig, 'n_eek_sig_err': n_lowq2_sig_err,
            'n_jpsik_sig': n_jpsi_sig, 'n_jpsik_sig_err': n_jpsi_sig_err,
            'eff_eek': eff_eek, 'eff_eek_err': eff_eek_err,
            'eff_jpsik': eff_jpsik, 'eff_jpsik_err': eff_jpsik_err
        }
        for key, value in results.items():
            outputs[key].append(value)

    for key, value_list in outputs.items():
        outputs[key] = np.array(value_list)

    path = Path('./significance_scan_data.pkl')
    with open(path, 'wb') as pkl_file:
        pickle.dump(outputs, pkl_file)


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])


    significance_scan(dataset_params, output_params, fit_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config/fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-v', '--verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    parser.add_argument('-minos', '--minos', action='store_true', help='use MINOS minimizer')
    args = parser.parse_args()

    main(args)
