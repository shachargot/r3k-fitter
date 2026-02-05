from pathlib import Path
import yaml
import tempfile
import numpy as np
import ROOT
import os
import uuid
from tqdm import tqdm


def makedirs(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def set_verbosity(args):
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kInfo if args.verbose > 1 else ROOT.kWarning
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.INFO if args.verbose > 1 else ROOT.RooFit.ERROR)
    printlevel = ROOT.RooFit.PrintLevel(1 if args.verbose else -1)

    return printlevel


def set_mode(dataset_params, output_params, fit_params, args):
    mode = args.mode

    file_mode = 'rare' if ('q2' in mode) else mode
    valid_file_key = [i for i in vars(dataset_params).keys() if file_mode+'_file' in i]
    assert len(valid_file_key) == 1, f'No valid file given for mode "{mode}"'
    dataset_params.mc_sig_file = getattr(dataset_params, valid_file_key[0])

    valid_fit_key = [i for i in fit_params.regions.keys() if mode in i]
    assert len(valid_fit_key) == 1, f'No valid fit region given for mode "{mode}"'
    fit_params.region = fit_params.regions[valid_fit_key[0]]
    fit_params.channel_label = '_'+valid_fit_key[0]+'_region'
    fit_params.fit_range = fit_params.region['fit_range']
    fit_params.ll_mass_range = fit_params.region['ll_mass_range']
    fit_params.fit_defaults = fit_params.region['defaults']
    fit_params.blinded = fit_params.region.get('blinded')
    fit_params.toy_signal_yield = fit_params.region.get('toy_signal_yield')


def save_params(params, template_filename, fit_params, args, update_dict=None, get_params=False, just_print=False, lock_file=False):
    template = {}
    for param in params:
        for key in fit_params.fit_defaults.keys():
            if key in param.GetName():
                template[key] = param.getVal()

    if args.verbose or just_print:
        print('Fitted Template Parameters:')
        for k, v in template.items():
            print('\t'+k+' = '+str(round(v, 2)))

        if just_print:
            return

    if Path(template_filename).is_file():
        with open(template_filename, 'r') as f:
            old_template = yaml.safe_load(f)
    else:
        old_template = None

    if old_template:
        old_template.update(template)
        template = old_template

    if update_dict:
        update_dict.update(template)
        template = update_dict

    if not lock_file:
        with open(template_filename, 'w') as f:
            yaml.dump(template, f)

    return template


def prepare_inputs(dataset_params, fit_params, b_mass_branch=None, isData=True, set_file=None, set_tree=None, score_cut=None, binned=False, unblind=False, weight_branch_name=None, weight_norm=None, weight_sf=None):
    # Read data from config file or manually set input
    if set_file is None:
        f_in = ROOT.TFile(dataset_params.data_file if isData else dataset_params.mc_sig_file, 'READ')
    else:
        f_in = ROOT.TFile(set_file, 'READ')
    # Read branches
    tree = f_in.Get(dataset_params.tree_name if set_tree is None else set_tree)
    # Decide whether to use BDT filtering/export
    # Prefer explicit argument 'score_cut' when provided, otherwise fall back to fit_params.bdt_score_cut
    bdt_cut_value = score_cut if score_cut is not None else getattr(fit_params, 'bdt_score_cut', None)
    use_bdt = bdt_cut_value is not None
    # Create RooRealVars (BDT only if in use)
    if use_bdt:
        bdt_branch = ROOT.RooRealVar(dataset_params.score_branch, 'BDT Score', -100., 100.)
    else:
        bdt_branch = None
    ll_mass_branch = ROOT.RooRealVar(dataset_params.ll_mass_branch, 'Di-Lepton Mass [GeV]', -100., 100.)

    # Take region cuts from cfg file
    if use_bdt:
        cutstring = '{}>{}&&{}>{}&&{}<{}'.format(
            dataset_params.score_branch,
            bdt_cut_value,
            dataset_params.ll_mass_branch,
            fit_params.region['ll_mass_range'][0],
            dataset_params.ll_mass_branch,
            fit_params.region['ll_mass_range'][1],
        )
        cutvar = ROOT.RooFormulaVar('cutvar', 'cutvar', cutstring, ROOT.RooArgList(bdt_branch, ll_mass_branch))
    else:
        cutstring = '{}>{}&&{}<{}'.format(
            dataset_params.ll_mass_branch,
            fit_params.region['ll_mass_range'][0],
            dataset_params.ll_mass_branch,
            fit_params.region['ll_mass_range'][1],
        )
        cutvar = ROOT.RooFormulaVar('cutvar', 'cutvar', cutstring, ROOT.RooArgList(ll_mass_branch))
    # Set fit ranges
    blindDataset = (isData and (fit_params.blinded)) and not unblind

    # Generate dataset and scale if specified
    if isData:
        # Build RooArgSet columns according to whether BDT is used
        if use_bdt:
            variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch)
            if binned:
                tmp_dataset = ROOT.RooDataHist('tmp_dataset_data'+fit_params.channel_label, 'Dataset', variables, ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cutvar))
            else:
                tmp_dataset = ROOT.RooDataSet('tmp_dataset_data'+fit_params.channel_label, 'Dataset', variables, ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cutvar))
        else:
            variables = ROOT.RooArgSet(b_mass_branch, ll_mass_branch)
            if binned:
                tmp_dataset = ROOT.RooDataHist('tmp_dataset_data'+fit_params.channel_label, 'Dataset', variables, ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cutvar))
            else:
                tmp_dataset = ROOT.RooDataSet('tmp_dataset_data'+fit_params.channel_label, 'Dataset', variables, ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cutvar))

        dataset = tmp_dataset.Clone(('dataset_data' if isData else 'dataset_mc')+fit_params.channel_label)
    else:
        # Optimized Logic for MC using RDataFrame
        rdf = ROOT.RDataFrame(tree)
        weight_branch_name = dataset_params.mc_weight_branch if weight_branch_name is None else weight_branch_name
        # Calculate Sum for Normalization Logic (Pre-cut application for correct norm)
        # Note: We filter by cutstring first to get the yield in the specific region
        rdf_temp = rdf.Filter(cutstring)
        weight_sum = rdf_temp.Sum("total_weight").GetValue()
        # Determine Scale Factor
        final_sf = 1.0
        if weight_norm:
            if weight_sum == 0:
                print("Warning: Weight sum is zero, cannot normalize.")
                final_sf = 0
            else:
                final_sf = weight_norm / weight_sum
        elif weight_sf:
            final_sf = weight_sf

        # Define the Effective Weight Column
        if final_sf != 1.0:
            actual_weight_name = "scaled_weight"
            # Define new weight: existing_weight * scale_factor
            rdf = rdf.Define(actual_weight_name, f"{weight_branch_name} * {final_sf}")
        else:
            actual_weight_name = weight_branch_name

        # 4. Apply Cuts
        rdf_cut = rdf.Filter(cutstring)

        # Using a temporary file avoids the slow Python loop and memory overhead
        with tempfile.NamedTemporaryFile(suffix='.root', delete=True) as tmp_f:
            snapshot_opts = ROOT.RDF.RSnapshotOptions()
            snapshot_opts.fMode = "RECREATE"

            # Only export the BDT score column when BDT is in use
            if use_bdt:
                columns = [dataset_params.b_mass_branch, dataset_params.score_branch,
                           dataset_params.ll_mass_branch, actual_weight_name]
            else:
                columns = [dataset_params.b_mass_branch,
                           dataset_params.ll_mass_branch, actual_weight_name]

            rdf_cut.Snapshot(dataset_params.tree_name, tmp_f.name, columns, snapshot_opts)

            # Load back into RooFit from the temp file
            f_tmp = ROOT.TFile.Open(tmp_f.name)
            tree_tmp = f_tmp.Get(dataset_params.tree_name)

            # Define the weight variable for RooFit
            weight_var = ROOT.RooRealVar(actual_weight_name, 'Weight', -1e9, 1e9)

            # Build RooArgSet for the RooDataSet import matching exported columns
            if use_bdt:
                variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch, weight_var)
            else:
                variables = ROOT.RooArgSet(b_mass_branch, ll_mass_branch, weight_var)

            if binned:
                tmp_dataset = ROOT.RooDataHist(
                    f'tmp_dataset_mc_{fit_params.channel_label}',
                    'Dataset',
                    variables,
                    ROOT.RooFit.Import(tree_tmp),
                    ROOT.RooFit.WeightVar(weight_var.GetName())
                )
            else:
                tmp_dataset = ROOT.RooDataSet(
                    f'tmp_dataset_mc_{fit_params.channel_label}',
                    'Dataset',
                    variables,
                    ROOT.RooFit.Import(tree_tmp),
                    ROOT.RooFit.WeightVar(weight_var.GetName())
                )

            # Clone to detach from temp file so we can close it
            dataset = tmp_dataset.Clone(('dataset_data' if isData else 'dataset_mc')+fit_params.channel_label)
            del tmp_dataset
            del tree_tmp
            f_tmp.Close()
            del f_tmp 
            del rdf
            del rdf_cut
            ROOT.gDirectory.Clear()
    
    if blindDataset:
        dataset = dataset.reduce(ROOT.RooFit.CutRange('sb1,sb2'))

    f_in.Close()
 
    return b_mass_branch, dataset


def format_params(params, let_float=False):
    params = {k: (v if isinstance(v, list) else (v,)) for k, v in params.items()}
    params = {k: (v if let_float else (v[0],)) for k, v in params.items()}

    return params


def write_workspace(output_params, args, model, extra_objs=[]):
    f_out = ROOT.TFile(str(Path(output_params.output_dir) / f'workspace_{args.mode}.root'), 'RECREATE')
    workspace = ROOT.RooWorkspace('workspace', 'workspace')
    getattr(workspace, 'import')(model.dataset)
    getattr(workspace, 'import')(model.fit_model)

    for obj in extra_objs:
        getattr(workspace, 'import')(obj)

    if args.verbose:
        workspace.Print()

    workspace.Write()
    f_out.Close()


def integrate(var, model, integral_range, fit_result, coeffs=None):
    var.setRange('int_range', *integral_range)
    rangeset = ROOT.RooFit.Range('int_range')
    xset = ROOT.RooArgSet(var)
    nset = ROOT.RooFit.NormSet(xset)
    integral_unscaled = model.createIntegral(xset, nset, rangeset)

    if coeffs is None:
        final_yield = model.expectedEvents(xset)
        final_yield_err = 0
    else:
        if isinstance(coeffs, list):
            final_yield = sum(c.getVal() for c in coeffs)
            final_yield_err = np.sqrt(np.sum([c.getError()**2 for c in coeffs]))
        else:
            final_yield = coeffs.getVal()
            if isinstance(coeffs, ROOT.RooRealVar):
                final_yield_err = coeffs.getError()
            else:
                final_yield_err = coeffs.getPropagatedError(fit_result)

    integral = integral_unscaled.getVal() * final_yield
    integral_err = final_yield_err * integral_unscaled.getVal()

    return integral, integral_err


def calculate_yields(b_mass_branch, component_map, fit_range, fit_result, custom_yield_ranges=None):
    if custom_yield_ranges is None:
        custom_yield_ranges = {}

    yields = {}
    for name, comp_info in component_map.items():
        pdf, coeff = comp_info[0], comp_info[1]

        current_range = custom_yield_ranges.get(name, fit_range)

        val, err = integrate(b_mass_branch, pdf, current_range, fit_result, coeffs=coeff)

        if len(comp_info) == 3:
            fraction = comp_info[2]
            val *= fraction
            err *= fraction

        yields[name] = (round(val, 2), round(err, 2))

    return yields


def loop_wrapper(iterable, args, unit='working point', title=None):
    if title:
        print(title)
    if isinstance(iterable, zip):
        unzipped = list(iterable)
        return iterable if args.verbose else tqdm(unzipped, total=len(unzipped), unit=unit)
    else:
        return iterable if args.verbose else tqdm(iterable, total=len(iterable), unit=unit)


def load_template_from_file(output_params, args):
    with open(Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', 'r') as file:
        template = yaml.safe_load(file)

    return template


def get_component_frac(key, component_yields, mc_yield_tot):
    return component_yields.get(key, 0) / mc_yield_tot if mc_yield_tot > 0 else 0
