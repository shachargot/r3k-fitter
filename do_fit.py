import ROOT
import yaml
import argparse
from pathlib import Path
import numpy as np
from pprint import pprint

from fit_models import FitModel
from physics_constants import (
    get_mc_scale_factor,
    SAMPLES,
    BR_B_PLUS_PSI2S_KSTAR,
    BR_B_ZERO_PSI2S_KSTAR,
    BR_KSTAR_PLUS_KPI0,
    BR_K0STAR_KPI,
    BR_B_PLUS_KSTAR_EE,
    BR_B_ZERO_KSTAR_EE
)
from utils import (
    set_verbosity,
    makedirs,
    set_mode,
    prepare_inputs,
    save_params,
    integrate,
    calculate_yields,
    write_workspace,
    load_template_from_file,
    get_component_frac
)

ROOT.gErrorIgnoreLevel = ROOT.kError
ALLOWED_MODES = ['jpsi', 'psi2s', 'lowq2']


# Nominal signal region fit to low-q2 non-res (NOTE: currently only blinded bkg and toy signal fits allowed)
def do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None,  toy_fit=True, unblinded=False, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)
    b_mass_branch.setRange('semilow', 4.65, 5.7)
    b_mass_branch.setRange('sb1', fit_params.fit_range[0], fit_params.blinded[0])
    b_mass_branch.setRange('sb2', fit_params.blinded[1], fit_params.fit_range[1])

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        sf = get_mc_scale_factor('rare_signal')
        _, dataset_rare = prepare_inputs(
            dataset_params,
            fit_params,
            b_mass_branch=b_mass_branch,
            isData=False,
            weight_branch_name=dataset_params.mc_weight_branch,
            weight_sf=sf
        )
        total_expected_signal_yield = float(dataset_rare.sumEntries())

        # Build Roofit model for signal
        model_sig_template = FitModel({'name': 'lowq2_signal', 'branch': b_mass_branch, 'dataset': dataset_rare, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.build_model()

        # Fit model to data
        model_sig_template.fit(dataset_rare, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_rare,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_samesign_data = prepare_inputs(
            dataset_params,
            fit_params,
            b_mass_branch=b_mass_branch,
            isData=True,
            set_file=dataset_params.samesign_data_file_lowq2,
            unblind=True
        )

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'name': 'lowq2_comb_bkg', 'branch': b_mass_branch, 'dataset': dataset_samesign_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.build_model()

        # Fit model to data
        model_comb_template.fit(dataset_samesign_data, use_minos=True if args.minos else False, printlevel=printlevel, fit_range='semilow', fit_norm_range='semilow')
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_samesign_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            bins=30,
            file_label=file_label,
            fit_range='semilow',
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit jpsi leakage in low-q2 region from MC
    if args.verbose:
        print('\nStarting Fit 3 - J/Psi Leakage Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    sf = get_mc_scale_factor('jpsi_resonant')
    _, dataset_jpsi = prepare_inputs(
        dataset_params,
        fit_params,
        b_mass_branch=b_mass_branch,
        isData=False,
        set_file=dataset_params.jpsi_lowq2_file,
        weight_branch_name=dataset_params.mc_weight_branch,
        weight_sf=sf,
    )
    total_expected_jpsi_bkg_yield = float(dataset_jpsi.sumEntries())

    if args.verbose:
        print(f'expected jpsi bkg: {total_expected_jpsi_bkg_yield}')

    # Build Roofit model for exponential background
    model_jpsi_template = FitModel({'name': 'lowq2_jpsi_leakage_bkg', 'branch': b_mass_branch, 'dataset': dataset_jpsi, 'channel_label': fit_params.channel_label})
    model_jpsi_template.add_background_model('jpsi_bkg_pdf', 'gauss', fit_params.fit_defaults, let_float=True)
    model_jpsi_template.build_model()

    # Fit model to data
    model_jpsi_template.fit(dataset_jpsi, use_minos=True if args.minos else False, fit_range='low', fit_norm_range='low', printlevel=printlevel)
    params = model_jpsi_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsi_template.plot_fit(
        b_mass_branch,
        dataset_jpsi,
        Path(output_params.output_dir) / f'fit_{args.mode}_jpsi_template.pdf',
        file_label=file_label,
        fit_range='low',
        fit_result=model_jpsi_template.fit_result,
        bins=35,
    )

    # Save fit shape parameters
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 4 - Partial Background Template\n{}'.format(50*'~'))

    partial_components = [
        'kstar_kaon',
        'kstar_pion',
        'k0star_kaon',
        'k0star_pion',
    ]

    model_part_template = FitModel({
        'name': 'lowq2_part_bkg',
        'branch': b_mass_branch,
        'dataset': dataset_samesign_data,
        'channel_label': fit_params.channel_label
    })

    # Proxy scaling: Use K*0 shape for missing K*+ mode; scale factor adds K*+ yield
    # estimated via ratio of total branching fractions: 1 + (BR_total(K*+) / BR_total(K*0))
    # partial_scalings = {
    #    'k0star_kaon': 1.0 + (BR_B_PLUS_KSTAR_EE * BR_KSTAR_PLUS_KPI0) / (BR_B_ZERO_KSTAR_EE * BR_K0STAR_KPI)
    # }

    total_expected_partial_yield, component_yields, dataset_merged = model_part_template.add_composite_kde_model(
        model_name='part_bkg_pdf',
        components=partial_components,
        dataset_params=dataset_params,
        fit_params=fit_params,
        samples_config=SAMPLES,
        scale_factor_func=get_mc_scale_factor,
        prepare_inputs_func=prepare_inputs,
        # yield_modifiers=partial_scalings,
        verbose=args.verbose
    )

    if args.verbose:
        print(f'expected partial bkg: {total_expected_partial_yield}')

    components_to_plot = {
        SAMPLES[name]['label']: getattr(model_part_template, f"pdf_{name}")
        for name in partial_components
    }

    model_part_template.plot_fit(
        b_mass_branch,
        dataset_merged,
        Path(output_params.output_dir) / f'fit_{args.mode}_partial_template.pdf',
        fit_components=components_to_plot,
        legend='ur',
        file_label=file_label,
        bins=30,
    )

    mc_yield_tot = sum(component_yields.values())
    kstar_kaon_yield_frac = get_component_frac('kstar_kaon', component_yields, mc_yield_tot)
    kstar_pion_yield_frac = get_component_frac('kstar_pion', component_yields, mc_yield_tot)
    k0star_kaon_yield_frac = get_component_frac('k0star_kaon', component_yields, mc_yield_tot)
    k0star_pion_yield_frac = get_component_frac('k0star_pion', component_yields, mc_yield_tot)

    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Add template for final fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))
    if args.cache:
        template = load_template_from_file(output_params, args)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(
        dataset_params,
        fit_params,
        b_mass_branch=b_mass_branch,
        set_file=dataset_params.data_file_lowq2,
        isData=True
    )

    # Use toys to produce expected signal
    if toy_fit:
        # Fit background-only model to data sidebands
        bkg_only_model = FitModel({'name': 'lowq2_bkg_only', 'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        bkg_only_model.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        bkg_only_model.add_background_model('jpsi_bkg_pdf', 'gauss', template, let_float=False)
        bkg_only_model.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])

        bkg_only_model.set_yield('comb_bkg_pdf', 2000, 0, 2*dataset_data.numEntries())
        bkg_only_model.set_yield('part_bkg_pdf', total_expected_partial_yield, 0, 2*dataset_data.numEntries())
        bkg_only_model.set_yield('jpsi_bkg_pdf', total_expected_jpsi_bkg_yield, 0, 2*dataset_data.numEntries())
        bkg_only_model.build_model()

        comb_bkg_only = bkg_only_model.background_models['comb_bkg_pdf']
        part_bkg_only = bkg_only_model.background_models['part_bkg_pdf']
        jpsi_bkg_only = bkg_only_model.background_models['jpsi_bkg_pdf']

        comb_bkg_only.coeff.setConstant(False)
        # jpsi_bkg_only.coeff.setConstant(False)
        part_bkg_only.coeff.setConstant(False)
        comb_bkg_only.exp_slope.setConstant(False)

        bkg_only_model.constraints.update({
            'part_bkg_coeff_constraint': ROOT.RooGaussian('part_bkg_coeff_constraint', 'part_bkg_coeff_constraint', part_bkg_only.coeff, ROOT.RooFit.RooConst(part_bkg_only.coeff.getVal()), ROOT.RooFit.RooConst(part_bkg_only.coeff.getVal()*.2)),
        })

        bkg_only_model.fit(dataset_data, use_minos=True if args.minos else False, fit_range='sb1,sb2', fit_norm_range='sb1,sb2', printlevel=printlevel)
        params = bkg_only_model.fit_result.floatParsFinal()

        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

        bkg_only_model.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_bkg_only.pdf',
            file_label=file_label,
            fit_components={
                'Combinatorial Bkg.': comb_bkg_only.model,
                'Part.-Reco. Bkg.': part_bkg_only.model,
                'B #rightarrow J/#psi K Bkg.': jpsi_bkg_only.model,
            },
            fit_range='full',
            fit_norm_range='sb1,sb2',
            fit_result=bkg_only_model.fit_result,
            bins=30,
            legend='ur',
            yrange=[0, 300],
        )

        # Generate expected background from sideband fit
        expected_bkg, _ = integrate(
            b_mass_branch,
            bkg_only_model.fit_model,
            [4.5, 5.7],
            bkg_only_model.fit_result,
            coeffs=[comb_bkg_only.coeff, jpsi_bkg_only.coeff, part_bkg_only.coeff],
        )
        toy_background = bkg_only_model.fit_model.generate(ROOT.RooArgSet(b_mass_branch), expected_bkg)

        # Generate expected signal from MC shape and jpsi-extrapolated yield
        if args.cache:
            sf = get_mc_scale_factor('rare_signal')
            _, dataset_rare = prepare_inputs(
                dataset_params,
                fit_params,
                b_mass_branch=b_mass_branch,
                isData=False,
                weight_branch_name=dataset_params.mc_weight_branch,
                weight_sf=sf
            )
            total_expected_signal_yield = float(dataset_rare.sumEntries())

            model_sig_template = FitModel({'name': 'lowq2_signal', 'branch': b_mass_branch, 'dataset': dataset_rare, 'channel_label': fit_params.channel_label})
            model_sig_template.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
            model_sig_template.build_model()

        toy_signal_yield = total_expected_signal_yield if fit_params.toy_signal_yield is None else fit_params.toy_signal_yield
        toy_signal = model_sig_template.fit_model.generate(ROOT.RooArgSet(b_mass_branch), toy_signal_yield)

        # Create toy dataset for final fit
        toy_dataset = dataset_data.emptyClone('dataset_data'+fit_params.channel_label, 'Toy Dataset (S+B)')
        toy_dataset.append(toy_background)
        toy_dataset.append(toy_signal)

        # Toy dataset plot
        tmp_frame = b_mass_branch.frame(ROOT.RooFit.Title(' '), ROOT.RooFit.Range('full'))
        dataset_data.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack), ROOT.RooFit.Name('ds'))
        toy_background.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue), ROOT.RooFit.Name('tb'))
        toy_signal.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed), ROOT.RooFit.Name('ts'))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.MarkerColor(ROOT.kCyan), ROOT.RooFit.Name('td'))
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kMagenta), ROOT.RooFit.Name('f'))
        comp = ROOT.RooArgSet(bkg_only_model.comb_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.Name('f_comb'))
        comp = ROOT.RooArgSet(bkg_only_model.jpsi_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Name('f_jpsi'))
        comp = ROOT.RooArgSet(bkg_only_model.part_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kViolet), ROOT.RooFit.Name('f_part'))

        legend = ROOT.TLegend(0.6, 0.6, 0.9, 0.9)
        legend.AddEntry(tmp_frame.findObject('tb'), 'Toy Background', 'LPE')
        legend.AddEntry(tmp_frame.findObject('ts'), 'Toy Signal', 'LPE')
        legend.AddEntry(tmp_frame.findObject('td'), 'Toy Dataset (S+B)', 'LPE')
        legend.AddEntry(tmp_frame.findObject('ds'), 'Blinded Data', 'LPE')
        legend.AddEntry(tmp_frame.findObject('f'), 'Bkg.-Only Fit', 'L')
        legend.AddEntry(tmp_frame.findObject('f_comb'), 'Bkg.-Only Fit (Comb.)', 'L')
        legend.AddEntry(tmp_frame.findObject('f_jpsi'), 'Bkg.-Only Fit (Jpsi)', 'L')
        legend.AddEntry(tmp_frame.findObject('f_part'), 'Bkg.-Only Fit (Part.-Reco.)', 'L')

        tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
        tmp_frame.Draw()
        tmp_frame.GetYaxis().SetRangeUser(0, 300)
        legend.Draw()
        tmp_c.SaveAs(str(Path(output_params.output_dir) / f'fit_{args.mode}_toy_dataset.pdf'))
        tmp_c.Close()

        dataset_data = toy_dataset

    # Build final Roofit model
    model_final = FitModel({'name': 'lowq2_final', 'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})

    if toy_fit:
        model_final.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=False)
    model_final.add_background_model('jpsi_bkg_pdf', 'gauss', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])

    if toy_fit:
        model_final.set_yield('sig_pdf', total_expected_signal_yield, 0, dataset_data.numEntries())
    model_final.set_yield('comb_bkg_pdf', 100, 0, 5*dataset_data.numEntries())
    model_final.set_yield('part_bkg_pdf', total_expected_partial_yield, 10, 5*dataset_data.numEntries())
    model_final.set_yield('jpsi_bkg_pdf', total_expected_jpsi_bkg_yield, 10, 5*dataset_data.numEntries())
    model_final.build_model()

    # Define handles for model components
    if toy_fit:
        sig = model_final.signal_models['sig_pdf']
    comb = model_final.background_models['comb_bkg_pdf']
    part = model_final.background_models['part_bkg_pdf']
    jpsi = model_final.background_models['jpsi_bkg_pdf']

    # Set constraint on jpsipi yield w.r.t. signal, let some params float
    if toy_fit:
        sig.coeff.setConstant(False)
        # sig.dcb1_mean.setConstant(False)
        # sig.dcb1_sigma.setConstant(False)
        # sig.dcb_coeff_ratio.setConstant(False)
    part.coeff.setConstant(False)
    comb.coeff.setConstant(False)
    jpsi.coeff.setConstant(False)
    comb.exp_slope.setConstant(False)

    # Make sure RooFit doesn't garbage collect
    if not hasattr(model_final, 'memory_store'):
        model_final.memory_store = []
    # <-- Define constraints here if needed
    model_final.memory_store.extend([])  # <-- And set them here

    # Add Gaussian constraints to fit
    model_final.add_constraints({
        # <-- Add constraints here if needed
    })

    # Fit model to data
    fit_range = 'full' if toy_fit else 'sb1,sb2'
    fit_norm_range = 'full' if toy_fit else 'sb1,sb2'
    model_final.fit(dataset_data, use_minos=True if args.minos else False, fit_range=fit_range, fit_norm_range=fit_norm_range, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Define the component map
    component_map = {
        **({'yield_sig':              (sig.model,    sig.coeff)} if toy_fit else {}),
        'yield_comb_bkg':             (comb.model,   comb.coeff),
        'yield_part_bkg':             (part.model,   part.coeff),
        'yield_jpsi_bkg':             (jpsi.model, jpsi.coeff),
        'yield_part_bkg_kstar_kaon':  (part.model,   part.coeff, kstar_kaon_yield_frac),
        'yield_part_bkg_kstar_pion':  (part.model,   part.coeff, kstar_pion_yield_frac),
        'yield_part_bkg_k0star_kaon': (part.model,   part.coeff, k0star_kaon_yield_frac),
        'yield_part_bkg_k0star_pion': (part.model,   part.coeff, k0star_pion_yield_frac),
    }

    # Call the generic calculator from utils.py
    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    # Use the results to create plot text and then plot the model
    if toy_fit:
        signal_yield = yields['yield_sig']
        sig_range = (custom_yield_ranges or {}).get('yield_sig')
        rounded_yield = [round(y) for y in signal_yield]
        plot_text = f'N_{{B #rightarrow eeK}} = {signal_yield[0]} #pm {signal_yield[1]}'
        if sig_range:
            plot_text = f'N_{{B #rightarrow eeK}} [{sig_range[0]}-{sig_range[1]} GeV] = {rounded_yield[0]} #pm {rounded_yield[1]}'
    else:
        # If not a toy fit, calculate total background in the blinded region for the plot label
        bkg_yields_in_blinded_region = calculate_yields(
            b_mass_branch, component_map, fit_params.blinded, model_final.fit_result
        )
        total_bkg_val = sum(val for val, err in bkg_yields_in_blinded_region.values())
        total_bkg_err = np.sqrt(sum(err**2 for val, err in bkg_yields_in_blinded_region.values()))
        plot_text = f'N_{{Bkg}} [{fit_params.blinded[0]}-{fit_params.blinded[1]} GeV] = {round(total_bkg_val)} #pm {round(total_bkg_err, 2)}'

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode}_final.pdf',
        # bins=30,
        file_label=file_label,
        fit_components={
            **({'Signal':                    sig.model} if toy_fit else {}),
            'Combinatorial Bkg.':            comb.model,
            'Part.-Reco. Bkg.':              part.model,
            'B #rightarrow J/#psi K Bkg.': jpsi.model,
        },
        fit_result=model_final.fit_result,
        legend='ul',
        extra_text=plot_text,
        stat_text_pos='middle',
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb.coeff.getVal(), 0, 5*dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part.coeff.getVal(), 0, dataset_data.numEntries())
    jpsi_bkg_pdf_norm = ROOT.RooRealVar('jpsi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', jpsi.coeff.getVal(), 0, dataset_data.numEntries())    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsi_bkg_pdf_norm])

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsi_bkg_pdf_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    if get_yields:
        return yields
    else:
        pprint(yields)


# Nominal control region fit to jpsi resonance
def do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        sf = get_mc_scale_factor('jpsi_resonant')
        _, dataset_mc = prepare_inputs(
            dataset_params,
            fit_params,
            b_mass_branch=b_mass_branch,
            isData=False,
            weight_branch_name=dataset_params.mc_weight_branch,
            weight_sf=sf
        )
        total_expected_signal_yield = float(dataset_mc.sumEntries())

        # Build Roofit model for signal
        model_sig_template = FitModel({'name': 'jpsi_signal', 'branch': b_mass_branch, 'dataset': dataset_mc, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_components=[
                model_sig_template.signal_models['sig_pdf'].dcb1_pdf,
                model_sig_template.signal_models['sig_pdf'].dcb2_pdf,
            ],
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        # Use loose bdt cut for same-sign data to get more statistics for the fit
        _, dataset_data = prepare_inputs(
            dataset_params,
            fit_params,
            isData=True,
            b_mass_branch=b_mass_branch,
            set_file=dataset_params.samesign_data_file_jpsi,
        )

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'name': 'jpsi_comb_bkg', 'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
            bins=30,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - Partial Background Template \n{}'.format(50*'~'))

    # Define Components from config/physics_constants.py
    partial_components = [
        'kstar_jpsi_kaon',
        'kstar_jpsi_pion',
        'k0star_jpsi_kaon',
        'k0star_jpsi_pion',
        'chic1_jpsi_kaon'
    ]

    # Build template
    model_part_template = FitModel({
        'name': 'jpsi_part_bkg',
        'branch': b_mass_branch,
        'dataset': dataset_data,  # Tmp input, will be replaced by individual component datasets in add_composite_kde_model
        'channel_label': fit_params.channel_label
    })

    total_expected_partial_yield, component_yields, dataset_merged = model_part_template.add_composite_kde_model(
        model_name='part_bkg_pdf',
        components=partial_components,
        dataset_params=dataset_params,
        fit_params=fit_params,
        samples_config=SAMPLES,
        scale_factor_func=get_mc_scale_factor,
        prepare_inputs_func=prepare_inputs,
        verbose=args.verbose
    )

    # Plotting with Decomposition
    components_to_plot = {
        SAMPLES[name]['label']: getattr(model_part_template, f"pdf_{name}")
        for name in partial_components
    }

    model_part_template.plot_fit(
        b_mass_branch,
        dataset_merged,
        Path(output_params.output_dir) / f'fit_{args.mode}_partial_template.pdf',
        fit_components=components_to_plot,
        legend='ur',
        file_label=file_label,
        bins=30,
    )

    # Calculate individual component fractions
    mc_yield_tot = sum(component_yields.values())
    kstar_kaon_yield_frac = get_component_frac('kstar_jpsi_kaon', component_yields, mc_yield_tot)
    kstar_pion_yield_frac = get_component_frac('kstar_jpsi_pion', component_yields, mc_yield_tot)
    k0star_kaon_yield_frac = get_component_frac('k0star_jpsi_kaon', component_yields, mc_yield_tot)
    k0star_pion_yield_frac = get_component_frac('k0star_jpsi_pion', component_yields, mc_yield_tot)
    chic1_kaon_yield_frac = get_component_frac('chic1_jpsi_kaon', component_yields, mc_yield_tot)
    kstar_yield_frac = (kstar_kaon_yield_frac + kstar_pion_yield_frac + k0star_kaon_yield_frac + k0star_pion_yield_frac)

    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to jpsipi MC
    if args.verbose:
        print('\nStarting Fit 4 - JpsiPi Partial Template \n{}'.format(50*'~'))

    sf = get_mc_scale_factor('jpsipi_jpsi_pion')
    _, dataset_jpsipi_pion = prepare_inputs(
        dataset_params,
        fit_params,
        isData=False,
        b_mass_branch=b_mass_branch,
        set_file=dataset_params.jpsipi_jpsi_pion_file,
        weight_branch_name=dataset_params.mc_weight_branch,
        weight_sf=sf
    )
    total_expected_jpsipi_yield = float(dataset_jpsipi_pion.sumEntries())

    model_jpsipi_pion_template = FitModel({'name': 'jpsi_jpsipi_bkg', 'branch': b_mass_branch, 'dataset': dataset_jpsipi_pion, 'channel_label': fit_params.channel_label})
    model_jpsipi_pion_template.add_background_model('jpsipi_bkg_pdf', 'dcb', fit_params.fit_defaults, let_float=True)
    model_jpsipi_pion_template.build_model()

    # Fit model to data
    model_jpsipi_pion_template.fit(dataset_jpsipi_pion, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_jpsipi_pion_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsipi_pion_template.plot_fit(
        b_mass_branch,
        dataset_jpsipi_pion,
        Path(output_params.output_dir) / f'fit_{args.mode}_jpsipi_template.pdf',
        file_label=file_label,
        fit_result=model_jpsipi_pion_template.fit_result,
    )

    # Save fit shape parameters
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Final Composite Fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        template = load_template_from_file(output_params, args)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(
        dataset_params,
        fit_params,
        isData=True,
        set_file=dataset_params.data_file_jpsi, 
        b_mass_branch=b_mass_branch
    )

    # Build final Roofit model
    model_final = FitModel({'name': 'jpsi_final', 'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])
    model_final.add_background_model('jpsipi_bkg_pdf', 'dcb', template, let_float=False)

    # Initialize pre-calculated coefficients
    model_final.set_yield('sig_pdf', total_expected_signal_yield, 0, dataset_data.numEntries())
    model_final.set_yield('comb_bkg_pdf', 2000, 0, dataset_data.numEntries())
    model_final.set_yield('part_bkg_pdf', total_expected_partial_yield, 0, dataset_data.numEntries())
    model_final.set_yield('jpsipi_bkg_pdf', total_expected_jpsipi_yield, 0, dataset_data.numEntries())
    model_final.build_model()

    # Define handles for model components
    sig = model_final.signal_models['sig_pdf']
    comb = model_final.background_models['comb_bkg_pdf']
    part = model_final.background_models['part_bkg_pdf']
    jpsipi = model_final.background_models['jpsipi_bkg_pdf']

    # Set constraint on jpsipi yield w.r.t. signal, let some params float
    sig.coeff.setConstant(False)
    part.coeff.setConstant(False)
    jpsipi.coeff.setConstant(False)
    comb.coeff.setConstant(False)
    sig.dcb1_mean.setConstant(False)
    sig.dcb1_sigma.setConstant(False)
    sig.dcb_coeff_ratio.setConstant(False)
    comb.exp_slope.setConstant(False)

    # Make sure RooFit doesn't garbage collect
    if not hasattr(model_final, 'memory_store'):
        model_final.memory_store = []
       
    if total_expected_signal_yield == 0:
        target_jpsipi_ratio = 0
    else: 
        target_jpsipi_ratio = total_expected_jpsipi_yield / total_expected_signal_yield
    if args.verbose:
        print(f'Target JPsiPi ratio: {target_jpsipi_ratio}')
    jpsipi_ratio = ROOT.RooFormulaVar(f'jpsipi_ratio{fit_params.channel_label}',  'JpsiPi Bkg / Signal',  '@0/@1', ROOT.RooArgList(jpsipi.coeff, sig.coeff))
    model_final.memory_store.extend([jpsipi_ratio])

    # Add Gaussian constraints to fit
    model_final.add_constraints({
        'jpsipi_ratio_constraint': ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(target_jpsipi_ratio), ROOT.RooFit.RooConst(target_jpsipi_ratio*0.05)),
    })

    # Fit model to data
    model_final.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Define the component map
    component_map = {
        'yield_sig':                  (sig.model,    sig.coeff),
        'yield_comb_bkg':             (comb.model,   comb.coeff),
        'yield_part_bkg':             (part.model,   part.coeff),
        'yield_jpsipi_bkg':           (jpsipi.model, jpsipi.coeff),
        'yield_part_bkg_kstar':       (part.model,   part.coeff, kstar_yield_frac),
        'yield_part_bkg_kstar_kaon':  (part.model,   part.coeff, kstar_kaon_yield_frac),
        'yield_part_bkg_kstar_pion':  (part.model,   part.coeff, kstar_pion_yield_frac),
        'yield_part_bkg_k0star_kaon': (part.model,   part.coeff, k0star_kaon_yield_frac),
        'yield_part_bkg_k0star_pion': (part.model,   part.coeff, k0star_pion_yield_frac),
        'yield_part_bkg_chic1_kaon':  (part.model,   part.coeff, chic1_kaon_yield_frac),
    }

    # Call the generic calculator from utils.py
    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    # Use the results to create plot text and then plot the model
    signal_yield = yields['yield_sig']
    sig_range = (custom_yield_ranges or {}).get('yield_sig')
    rounded_yield = [round(y) for y in signal_yield]
    plot_text = f'N_{{J/#psi}} = {rounded_yield[0]} #pm {rounded_yield[1]}'
    if sig_range:
        plot_text = f'N_{{J/#psi}} [{sig_range[0]}-{sig_range[1]} GeV] = {rounded_yield[0]} #pm {rounded_yield[1]}'

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode}_final.pdf',
        file_label=file_label,
        fit_components={
            'Signal':                        sig.model,
            'Combinatorial Bkg.':            comb.model,
            'Part.-Reco. Bkg.':              part.model,
            'B #rightarrow J/#psi #pi Bkg.': jpsipi.model,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text=plot_text,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb.coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part.coeff.getVal(), 0, dataset_data.numEntries())
    jpsipi_bkg_pdf_norm = ROOT.RooRealVar('jpsipi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', jpsipi.coeff.getVal(), 0, dataset_data.numEntries())    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsipi_bkg_pdf_norm])

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsipi_bkg_pdf_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    if get_yields:
        return yields
    else:
        pprint(yields)


# Constrained parameter control region fit to J/psi resonance for niche cases
def do_constrained_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        sf = get_mc_scale_factor('jpsi_resonant')
        _, dataset_mc = prepare_inputs(
            dataset_params,
            fit_params,
            b_mass_branch=b_mass_branch,
            weight_branch_name=dataset_params.mc_weight_branch,
            weight_sf=sf
        )
        total_expected_signal_yield = float(dataset_mc.sumEntries())

        # Build Roofit model for signal
        model_sig_template = FitModel({'name': 'jpsi_signal', 'branch': b_mass_branch, 'dataset': dataset_mc, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.build_model()

        # Fit model to data
        model_sig_template.fit(dataset_mc, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_components=[
                model_sig_template.signal_models['sig_pdf'].dcb1_pdf,
                model_sig_template.signal_models['sig_pdf'].dcb2_pdf,
            ],
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    if not args.cache:
        # Fit combinatorial background to same-sign electron data
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(
            dataset_params,
            fit_params,
            isData=True,
            b_mass_branch=b_mass_branch,
            set_file=dataset_params.samesign_data_file_jpsi,
        )

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'name': 'jpsi_comb_bkg', 'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.build_model()

        # Fit model to data
        model_comb_template.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape
    if args.verbose:
        print('\nStarting Fit 3 - Partial BackgroundTemplate \n{}'.format(50*'~'))

    # Define Components from config/physics_constants.py
    partial_components = [
        'kstar_jpsi_kaon',
        'kstar_jpsi_pion',
        'k0star_jpsi_kaon',
        'k0star_jpsi_pion',
        'chic1_jpsi_kaon'
    ]

    # Build template
    model_part_template = FitModel({
        'name': 'jpsi_part_bkg',
        'branch': b_mass_branch,
        'dataset': dataset_data,
        'channel_label': fit_params.channel_label
    })

    total_expected_partial_yield, component_yields, dataset_merged = model_part_template.add_composite_kde_model(
        model_name='part_bkg_pdf',
        components=partial_components,
        dataset_params=dataset_params,
        fit_params=fit_params,
        samples_config=SAMPLES,
        scale_factor_func=get_mc_scale_factor,
        prepare_inputs_func=prepare_inputs,
        verbose=args.verbose
    )

    # Plotting with Decomposition
    components_to_plot = {
        SAMPLES[name]['label']: getattr(model_part_template, f"pdf_{name}")
        for name in partial_components
    }

    model_part_template.plot_fit(
        b_mass_branch,
        dataset_merged,
        Path(output_params.output_dir) / f'fit_{args.mode}_partial_template.pdf',
        fit_components=components_to_plot,
        legend='ur',
        file_label=file_label,
        bins=30,
    )

    # Calculate individual component fractions
    mc_yield_tot = sum(component_yields.values())
    kstar_kaon_yield_frac = get_component_frac('kstar_jpsi_kaon', component_yields, mc_yield_tot)
    kstar_pion_yield_frac = get_component_frac('kstar_jpsi_pion', component_yields, mc_yield_tot)
    k0star_kaon_yield_frac = get_component_frac('k0star_jpsi_kaon', component_yields, mc_yield_tot)
    k0star_pion_yield_frac = get_component_frac('k0star_jpsi_pion', component_yields, mc_yield_tot)
    chic1_kaon_yield_frac = get_component_frac('chic1_jpsi_kaon', component_yields, mc_yield_tot)
    kstar_yield_frac = (kstar_kaon_yield_frac + kstar_pion_yield_frac + k0star_kaon_yield_frac + k0star_pion_yield_frac)

    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to jpsipi MC
    if args.verbose:
        print('\nStarting Fit 4 - JpsiPi Partial Template \n{}'.format(50*'~'))

    sf = get_mc_scale_factor('jpsipi_jpsi_pion')
    _, dataset_jpsipi = prepare_inputs(
        dataset_params,
        fit_params,
        isData=False,
        b_mass_branch=b_mass_branch,
        set_file=dataset_params.jpsipi_jpsi_pion_file,
        weight_branch_name=dataset_params.mc_weight_branch,
        weight_sf=sf
    )
    total_expected_jpsipi_yield = float(dataset_jpsipi.sumEntries())

    model_jpsipi_template = FitModel({'name': 'jpsi_jpsipi_bkg', 'branch': b_mass_branch, 'dataset': dataset_jpsipi, 'channel_label': fit_params.channel_label})
    model_jpsipi_template.add_background_model('jpsipi_bkg_pdf', 'dcb', fit_params.fit_defaults, let_float=True)
    model_jpsipi_template.build_model()

    # Fit model to data
    model_jpsipi_template.fit(dataset_jpsipi, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_jpsipi_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsipi_template.plot_fit(
        b_mass_branch,
        dataset_jpsipi,
        Path(output_params.output_dir) / f'fit_{args.mode}_jpsipi_template.pdf',
        file_label=file_label,
        fit_result=model_jpsipi_template.fit_result,
    )

    # Save fit shape parameters
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Final Composite Fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        template = load_template_from_file(output_params, args)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(
        dataset_params,
        fit_params,
        isData=True,
        set_file=dataset_params.data_file_jpsi,
        b_mass_branch=b_mass_branch
    )

    # Build final Roofit model
    model_final = FitModel({'name': 'jpsi_final', 'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])
    model_final.add_background_model('jpsipi_bkg_pdf', 'dcb', template, let_float=False)

    # Initialize pre-calculated coefficients
    model_final.set_yield('sig_pdf', total_expected_signal_yield, 0, dataset_data.numEntries())
    model_final.set_yield('comb_bkg_pdf', 2000, 0, dataset_data.numEntries())
    model_final.set_yield('part_bkg_pdf', total_expected_partial_yield, 0, dataset_data.numEntries())
    model_final.set_yield('jpsipi_bkg_pdf', total_expected_jpsipi_yield, 0, dataset_data.numEntries())
    model_final.build_model()

    # Define handles for model components
    sig = model_final.signal_models['sig_pdf']
    comb = model_final.background_models['comb_bkg_pdf']
    part = model_final.background_models['part_bkg_pdf']
    jpsipi = model_final.background_models['jpsipi_bkg_pdf']

    # Keep constrained to some of the expected yields/shapes
    sig.coeff.setConstant(False)
    part.coeff.setConstant(False)
    jpsipi.coeff.setConstant(False)
    comb.coeff.setConstant(False)
    comb.exp_slope.setConstant(False)

    target_partial_ratio = total_expected_partial_yield / total_expected_signal_yield
    target_jpsipi_ratio = total_expected_jpsipi_yield / total_expected_signal_yield

    # Make sure RooFit doesn't garbage collect
    if not hasattr(model_final, 'memory_store'):
        model_final.memory_store = []

    partial_ratio = ROOT.RooFormulaVar(f'partial_ratio{fit_params.channel_label}', 'Partial Bkg / Signal', '@0/@1', ROOT.RooArgList(part.coeff, sig.coeff))
    jpsipi_ratio = ROOT.RooFormulaVar(f'jpsipi_ratio{fit_params.channel_label}',  'JpsiPi Bkg / Signal',  '@0/@1', ROOT.RooArgList(jpsipi.coeff, sig.coeff))
    model_final.memory_store.extend([partial_ratio, jpsipi_ratio])

    # Add Gaussian constraints to fit
    model_final.add_constraints({
        'partial_ratio_constraint': ROOT.RooGaussian('partial_ratio_constraint', 'partial_ratio_constraint', partial_ratio, ROOT.RooFit.RooConst(target_partial_ratio), ROOT.RooFit.RooConst(target_partial_ratio * 0.05)),
        'jpsipi_ratio_constraint': ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(target_jpsipi_ratio), ROOT.RooFit.RooConst(target_jpsipi_ratio * 0.05)),
    })

    # Fit model to data
    model_final.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Define the component map
    component_map = {
        'yield_sig':                  (sig.model,    sig.coeff),
        'yield_comb_bkg':             (comb.model,   comb.coeff),
        'yield_part_bkg':             (part.model,   part.coeff),
        'yield_jpsipi_bkg':           (jpsipi.model, jpsipi.coeff),
        'yield_part_bkg_kstar':       (part.model,   part.coeff, kstar_yield_frac),
        'yield_part_bkg_kstar_kaon':  (part.model,   part.coeff, kstar_kaon_yield_frac),
        'yield_part_bkg_kstar_pion':  (part.model,   part.coeff, kstar_pion_yield_frac),
        'yield_part_bkg_k0star_kaon': (part.model,   part.coeff, k0star_kaon_yield_frac),
        'yield_part_bkg_k0star_pion': (part.model,   part.coeff, k0star_pion_yield_frac),
        'yield_part_bkg_chic1_kaon':  (part.model,   part.coeff, chic1_kaon_yield_frac),
    }

    # Call the generic calculator from utils.py
    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    # Use the results to create plot text and then plot the model
    signal_yield = yields['yield_sig']
    sig_range = (custom_yield_ranges or {}).get('yield_sig')
    rounded_yield = [round(y) for y in signal_yield]
    plot_text = f'N_{{J/#psi}} = {rounded_yield[0]} #pm {rounded_yield[1]}'
    if sig_range:
        plot_text = f'N_{{J/#psi}} [{sig_range[0]}-{sig_range[1]} GeV] = {rounded_yield[0]} #pm {rounded_yield[1]}'

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode}_final.pdf',
        file_label=file_label,
        fit_components={
            'Signal':                        sig.model,
            'Combinatorial Bkg.':            comb.model,
            'Part.-Reco. Bkg.':              part.model,
            'B #rightarrow J/#psi #pi Bkg.': jpsipi.model,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text=plot_text,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb.coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part.coeff.getVal(), 0, dataset_data.numEntries())
    jpsipi_bkg_pdf_norm = ROOT.RooRealVar('jpsipi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', jpsipi.coeff.getVal(), 0, dataset_data.numEntries())    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsipi_bkg_pdf_norm])

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsipi_bkg_pdf_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    if get_yields:
        return yields
    else:
        pprint(yields)


# Nominal control region fit to psi2s resonance
def do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)

    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        sf = get_mc_scale_factor('psi2s_resonant')
        _, dataset_mc = prepare_inputs(
            dataset_params,
            fit_params,
            b_mass_branch=b_mass_branch,
            isData=False,
            weight_branch_name=dataset_params.mc_weight_branch,
            weight_sf=sf
        )
        total_expected_signal_yield = float(dataset_mc.sumEntries())

        model_sig_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_mc, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.build_model()

        model_sig_template.fit(dataset_mc, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_components=[
                model_sig_template.signal_models['sig_pdf'].dcb1_pdf,
                model_sig_template.signal_models['sig_pdf'].dcb2_pdf,
            ],
            fit_result=model_sig_template.fit_result,
        )

        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        _, dataset_data = prepare_inputs(
            dataset_params,
            fit_params,
            isData=True,
            b_mass_branch=b_mass_branch,
            set_file=dataset_params.samesign_data_file_psi2s,
        )

        model_comb_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.build_model()

        model_comb_template.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
            bins=30,
        )

        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - Partial Background Template\n{}'.format(50*'~'))

    partial_components = [
        'kstar_psi2s_kaon',
        'kstar_psi2s_pion',
        'k0star_psi2s_kaon',
        'k0star_psi2s_pion',
    ]

    model_part_template = FitModel({
        'name': 'psi2s_part_bkg',
        'branch': b_mass_branch,
        'dataset': dataset_data,
        'channel_label': fit_params.channel_label
    })

    # Proxy scaling: Use K*0 shape for missing K*+ mode; scale factor adds K*+ yield
    # estimated via ratio of total branching fractions: 1 + (BR_total(K*+) / BR_total(K*0))
    # partial_scalings = {
    #    'k0star_psi2s_kaon': 1.0 + (BR_B_PLUS_PSI2S_KSTAR * BR_KSTAR_PLUS_KPI0) / (BR_B_ZERO_PSI2S_KSTAR * BR_K0STAR_KPI)
    # }

    total_expected_partial_yield, component_yields, dataset_merged = model_part_template.add_composite_kde_model(
        model_name='part_bkg_pdf',
        components=partial_components,
        dataset_params=dataset_params,
        fit_params=fit_params,
        samples_config=SAMPLES,
        scale_factor_func=get_mc_scale_factor,
        prepare_inputs_func=prepare_inputs,
      # yield_modifiers=partial_scalings,
        verbose=args.verbose
    )

    components_to_plot = {
        SAMPLES[name]['label']: getattr(model_part_template, f"pdf_{name}")
        for name in partial_components
    }

    model_part_template.plot_fit(
        b_mass_branch,
        dataset_merged,
        Path(output_params.output_dir) / f'fit_{args.mode}_partial_template.pdf',
        fit_components=components_to_plot,
        legend='ur',
        file_label=file_label,
        bins=30,
    )

    mc_yield_tot = sum(component_yields.values())
    kstar_kaon_yield_frac = get_component_frac('kstar_psi2s_kaon', component_yields, mc_yield_tot)
    kstar_pion_yield_frac = get_component_frac('kstar_psi2s_pion', component_yields, mc_yield_tot)
    k0star_kaon_yield_frac = get_component_frac('k0star_psi2s_kaon', component_yields, mc_yield_tot)
    k0star_pion_yield_frac = get_component_frac('k0star_psi2s_pion', component_yields, mc_yield_tot)

    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    if args.verbose:
        print('\nStarting Fit 4 - Psi2sPi Partial Template \n{}'.format(50*'~'))

    sf = get_mc_scale_factor('psi2spi_psi2s_pion')
    _, dataset_psi2spi = prepare_inputs(
        dataset_params,
        fit_params,
        isData=False,
        b_mass_branch=b_mass_branch,
        set_file=dataset_params.psi2spi_psi2s_pion_file,
        weight_branch_name=dataset_params.mc_weight_branch,
        weight_sf=sf
    )
    total_expected_psi2spi_yield = float(dataset_psi2spi.sumEntries())

    model_psi2spi_template = FitModel({'name': 'psi2s_psi2spi_bkg', 'branch': b_mass_branch, 'dataset': dataset_psi2spi, 'channel_label': fit_params.channel_label})
    model_psi2spi_template.add_background_model('psi2spi_bkg_pdf', 'dcb', fit_params.fit_defaults, let_float=True)
    model_psi2spi_template.build_model()

    # Fit model to data
    model_psi2spi_template.fit(dataset_psi2spi, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_psi2spi_template.fit_result.floatParsFinal()

    # Plot fit result
    model_psi2spi_template.plot_fit(
        b_mass_branch,
        dataset_psi2spi,
        Path(output_params.output_dir) / f'fit_{args.mode}_psi2spi_template.pdf',
        file_label=file_label,
        fit_result=model_psi2spi_template.fit_result,
    )

    # Save fit shape parameters
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        template = load_template_from_file(output_params, args)

    _, dataset_data = prepare_inputs(
        dataset_params,
        fit_params,
        isData=True,
        set_file=dataset_params.data_file_psi2s,
        b_mass_branch=b_mass_branch
    )

    model_final = FitModel({'name': 'psi2s_final', 'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])
    model_final.add_background_model('psi2spi_bkg_pdf', 'dcb', template, let_float=False)

    model_final.set_yield('sig_pdf', total_expected_signal_yield, 0, dataset_data.numEntries())
    model_final.set_yield('comb_bkg_pdf', 2000, 0, dataset_data.numEntries())
    model_final.set_yield('part_bkg_pdf', total_expected_partial_yield, 0, dataset_data.numEntries())
    model_final.set_yield('psi2spi_bkg_pdf', total_expected_psi2spi_yield, 0, dataset_data.numEntries())
    model_final.build_model()

    sig = model_final.signal_models['sig_pdf']
    comb = model_final.background_models['comb_bkg_pdf']
    part = model_final.background_models['part_bkg_pdf']
    psi2spi = model_final.background_models['psi2spi_bkg_pdf']

    sig.coeff.setConstant(False)
    part.coeff.setConstant(False)
    comb.coeff.setConstant(False)
    psi2spi.coeff.setConstant(False)
    comb.exp_slope.setConstant(False)

    if not hasattr(model_final, 'memory_store'):
        model_final.memory_store = []
    # <-- Define constraints here if needed
    target_psi2spi_ratio = total_expected_psi2spi_yield / total_expected_signal_yield
    psi2spi_ratio = ROOT.RooFormulaVar(f'psi2spi_ratio{fit_params.channel_label}',  'JpsiPi Bkg / Signal',  '@0/@1', ROOT.RooArgList(psi2spi.coeff, sig.coeff))
    model_final.memory_store.extend([psi2spi_ratio])

    if args.verbose:
        print(f'Target Psi2sPi ratio: {target_psi2spi_ratio}')

    # Add Gaussian constraints to fit
    model_final.add_constraints({
       'psi2spi_ratio_constraint': ROOT.RooGaussian('psi2spi_ratio_constraint', 'psi2spi_ratio_constraint', psi2spi_ratio, ROOT.RooFit.RooConst(target_psi2spi_ratio), ROOT.RooFit.RooConst(target_psi2spi_ratio*0.05)),
    })

    model_final.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    component_map = {
        'yield_sig':                  (sig.model,    sig.coeff),
        'yield_comb_bkg':             (comb.model,   comb.coeff),
        'yield_part_bkg':             (part.model,   part.coeff),
        'yield_part_bkg_kstar_kaon':  (part.model,   part.coeff, kstar_kaon_yield_frac),
        'yield_part_bkg_kstar_pion':  (part.model,   part.coeff, kstar_pion_yield_frac),
        'yield_part_bkg_k0star_kaon': (part.model,   part.coeff, k0star_kaon_yield_frac),
        'yield_part_bkg_k0star_pion': (part.model,   part.coeff, k0star_pion_yield_frac),
        'yield_psi2spi_bkg':          (psi2spi.model,   psi2spi.coeff),
    }

    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    signal_yield = yields['yield_sig']
    sig_range = (custom_yield_ranges or {}).get('yield_sig')
    rounded_yield = [round(y) for y in signal_yield]
    plot_text = f'N_{{#psi(2S)}} = {rounded_yield[0]} #pm {rounded_yield[1]}'
    if sig_range:
        plot_text = f'N_{{#psi(2S)}} [{sig_range[0]}-{sig_range[1]} GeV] = {rounded_yield[0]} #pm {rounded_yield[1]}'

    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode}_final.pdf',
        file_label=file_label,
        fit_components={
            'Signal':                        sig.model,
            'Combinatorial Bkg.':            comb.model,
            'Part.-Reco. Bkg.':              part.model,
            'B #rightarrow #psi(2S) #pi Bkg.': psi2spi.model,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text=plot_text,
    )

    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb.coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part.coeff.getVal(), 0, dataset_data.numEntries())
    psi2spi_bkg_pdf_norm = ROOT.RooRealVar('psi2spi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', psi2spi.coeff.getVal(), 0, dataset_data.numEntries())
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm, psi2spi_bkg_pdf_norm])

    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, psi2spi_bkg_pdf_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    if get_yields:
        return yields
    else:
        pprint(yields)


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    if args.mode == 'all':
        # Sequential execution for 'all' mode
        modes_to_run = [
            ('lowq2', do_lowq2_signal_region_fit, {"toy_fit": args.toy_fit}),
            ('jpsi', do_jpsi_control_region_fit, {}),
            ('psi2s', do_psi2s_control_region_fit, {})
        ]

        for mode_name, fit_function, extra_kwargs in modes_to_run:
            args.mode = mode_name
            if args.verbose:
                print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
            fit_function(dataset_params, output_params, fit_params, args, **extra_kwargs)

    elif args.mode == 'lowq2':
        if args.constrained_fit:
            raise NotImplementedError('No full constrained fit for lowq2')
        else:
            do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, toy_fit=args.toy_fit)

    elif args.mode == 'jpsi':
        if args.constrained_fit:
            do_constrained_jpsi_control_region_fit(dataset_params, output_params, fit_params, args)
        else:
            do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args)

    elif args.mode == 'psi2s':
        if args.constrained_fit:
            raise NotImplementedError('No full constrained fit for psi2s')
        else:
            do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which fit to perform')
    parser.add_argument('-v', '--verbose', nargs='?', const=1, default=0, type=int, help='Set verbosity level. Default is 0. If flag is used without value, sets to 1.')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    parser.add_argument('-t', '--toy_fit', dest='toy_fit', action='store_true', help='fit toy data in low-q2')
    parser.add_argument('-cf', '--constrained_fit', dest='constrained_fit', action='store_true', help='Fit with norm-constrained templates')
    parser.add_argument('-minos', '--minos', dest='minos', action='store_true', help='use MINOS minimizer')
    args = parser.parse_args()

    main(args)
