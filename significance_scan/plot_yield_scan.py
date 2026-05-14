import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from uncertainties import unumpy


def yield_plotter(data, path, show=False):
    score = data['score']
    n_eek_sig = data['n_eek_sig']
    n_eek_sig_err = data['n_eek_sig_err']
    n_eek_bkg = data['n_eek_bkg']
    n_eek_bkg_err = data['n_eek_bkg_err']
    n_jpsik_sig = data['n_jpsik_sig']
    n_jpsik_sig_err = data['n_jpsik_sig_err']
    
    for i in range(len(score)):
        print(score[i], n_jpsik_sig[i], n_jpsik_sig_err[i])

    fig, ax = plt.subplots(figsize=(8,8))
    ax.errorbar(score, n_eek_sig, yerr=n_eek_sig_err, label=r'$N^{sig}_{B \rightarrow eeK}$ (extrapolated)', ls='none', marker='.')
    ax.errorbar(score, n_eek_bkg, yerr=n_eek_bkg_err, label=r'$N^{bkg}_{B \rightarrow eeK}$', ls='none', marker='.')
    ax.errorbar(score, n_jpsik_sig, yerr=n_jpsik_sig_err, label=r'$N^{sig}_{B \rightarrow J/\psi(ee) K}$', ls='none', marker='.')
    
    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'$N_{events}$',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='lower right',fontsize=16)
    ax.set_yscale('log')

    
    if show:
        fig.show()

    fig.savefig(path, bbox_inches='tight')

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
        output_file = Path('.') / 'yield_scan.pdf'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    with open(data_file, 'rb') as f:
        score_data = pickle.load(f)

    yield_plotter(score_data, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='input_file', 
        type=str, help='pickle data file')
    parser.add_argument('-o', '--output', dest='output', 
        type=str, help='output file path')
    parser.add_argument('-l', '--label', dest='label', 
        type=str, help='output file label')
    args, _ = parser.parse_known_args()

    main(args)
