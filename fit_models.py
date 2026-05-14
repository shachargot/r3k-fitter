from pathlib import Path
import ROOT
from utils import set_verbosity, format_params


class PDFDictWrapper:
    def __init__(self, name, model):
        self.name = name
        self.model = model


class PDFDict():
    def __init__(self, name, shape, xvar, parameters, dataset=None, let_float=False, channel=None):
        allowed_shapes = ['dcb', 'cb+gauss', 'dcb+dcb', 'cb+cb', 'gauss', 'exp', 'poly', 'generic', 'kde']
        assert shape in allowed_shapes, "Choose a PDF shape from list:{}".format(allowed_shapes)

        self.name = name
        self.parameters = parameters
        self.channel = channel
        self.channel_label = channel if channel else ''
        self.build_model(shape, xvar, self.parameters, let_float, dataset, label=self.name)

    def build_model(self, shape, xvar, parameters, let_float, dataset, label=''):
        if shape == 'gauss':
            shape_dict = {
                'gauss_mean':   'Gaussian: location parameter of the Gaussian',
                'gauss_sigma':  'Gaussian: width parameter of the Gaussian',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.model = ROOT.RooGaussian(
                self.name+self.channel_label,
                'Gaussian pdf',
                xvar, self.gauss_mean, self.gauss_sigma)

        if shape == 'dcb':
            shape_dict = {
                'dcb_mean':   'DS-CB: location parameter of the Gaussian component',
                'dcb_sigma':  'DS-CB: width parameter of the Gaussian component',
                'dcb_alpha1': 'DS-CB: location of transition to a power law on the left, in std devs away from mean',
                'dcb_n1':     'DS-CB: exponent of power-law tail on the left',
                'dcb_alpha2': 'DS-CB: location of transition to a power law on the right, in std devs away from mean',
                'dcb_n2':     'DS-CB: exponent of power-law tail on the right',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.model = ROOT.RooCrystalBall(
                self.name+self.channel_label,
                'Double-sided crystal-ball pdf',
                xvar, self.dcb_mean, self.dcb_sigma, self.dcb_alpha1, self.dcb_n1, self.dcb_alpha2, self.dcb_n2)

        if shape == 'cb+gauss':
            shape_dict = {
                'cb_gauss_coeff_ratio': 'CB+Gauss: Ratio of CB & gaussian components',
                'gauss_mean':           'CB+Gauss: Mean of gaussian component',
                'gauss_sigma':          'CB+Gauss: Width of gaussian component',
                'cb_mean':              'CB+Gauss: Mean of CB component',
                'cb_sigma':             'CB+Gauss: Width of CB component',
                'cb_alpha':             'CB+Gauss: Location of transition to a power law of CB component',
                'cb_n':                 'CB+Gauss: Exponent of power-law tail of CB component',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

                if 'cb_gauss_coeff_ratio' in par:
                    name_fmt_comp = par+'_comp_'+label if label else par+'_comp'
                    setattr(self, 'cb_gauss_coeff_ratio_comp', ROOT.RooFormulaVar(
                        name_fmt_comp+self.channel_label,
                        '1-{}'.format(getattr(self, par).GetName()),
                        ROOT.RooArgList(getattr(self, par)))
                    )

            self.gauss_pdf = ROOT.RooGaussian(
                'gauss_pdf'+self.channel_label,
                'CB+Gauss: Gaussian component',
                xvar, self.gauss_mean, self.gauss_sigma)

            self.cb_pdf = ROOT.RooCBShape(
                'cb_pdf'+self.channel_label,
                'CB+Gauss: CB component',
                xvar, self.cb_mean, self.cb_sigma, self.cb_alpha, self.cb_n)

            self.model = ROOT.RooAddPdf(
                self.name+self.channel_label,
                'CB+Gauss',
                ROOT.RooArgList(self.cb_pdf, self.gauss_pdf),
                ROOT.RooArgList(self.cb_gauss_coeff_ratio, self.cb_gauss_coeff_ratio_comp)
            )

        if shape == 'dcb+dcb':
            shape_dict = {
                'dcb_coeff_ratio': 'DCB+DCB: Ratio of DCB components',
                'dcb1_mean':       'DCB+DCB: Mean of DCB1 component',
                'dcb1_sigma':      'DCB+DCB: Width of DCB1 component',
                'dcb1_alpha1':     'DCB+DCB: Location of left transition to a power law of DCB1 component',
                'dcb1_n1':         'DCB+DCB: Exponent of left power-law tail of DCB1 component',
                'dcb1_alpha2':     'DCB+DCB: Location of right transition to a power law of DCB1 component',
                'dcb1_n2':         'DCB+DCB: Exponent of right power-law tail of DCB1 component',
                'dcb2_mean':       'DCB+DCB: Mean of DCB2 component',
                'dcb2_sigma':      'DCB+DCB: Width of DCB2 component',
                'dcb2_alpha1':     'DCB+DCB: Location of left transition to a power law of DCB2 component',
                'dcb2_n1':         'DCB+DCB: Exponent of left power-law tail of DCB2 component',
                'dcb2_alpha2':     'DCB+DCB: Location of right transition to a power law of DCB2 component',
                'dcb2_n2':         'DCB+DCB: Exponent of right power-law tail of DCB2 component',
            }

            for par, desc in shape_dict.items():

                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

                if 'dcb_coeff_ratio' in par:
                    name_fmt_comp = par+'_comp_'+label if label else par+'_comp'
                    setattr(self, 'dcb_coeff_ratio_comp', ROOT.RooFormulaVar(
                        name_fmt_comp+self.channel_label,
                        '1-{}'.format(getattr(self, par).GetName()),
                        ROOT.RooArgList(getattr(self, par)))
                    )

            self.dcb1_pdf = ROOT.RooCrystalBall(
                'dcb1_pdf'+self.channel_label,
                'DCB+DCB: DCB1 component',
                xvar, self.dcb1_mean, self.dcb1_sigma, self.dcb1_alpha1, self.dcb1_n1, self.dcb1_alpha2, self.dcb1_n2)

            self.dcb2_pdf = ROOT.RooCrystalBall(
                'dcb2_pdf'+self.channel_label,
                'DCB+DCB: DCB2 component',
                xvar, self.dcb2_mean, self.dcb2_sigma, self.dcb2_alpha1, self.dcb2_n1, self.dcb2_alpha2, self.dcb2_n2)

            self.model = ROOT.RooAddPdf(
                self.name+self.channel_label,
                'DCB+DCB',
                ROOT.RooArgList(self.dcb1_pdf, self.dcb2_pdf),
                ROOT.RooArgList(self.dcb_coeff_ratio, self.dcb_coeff_ratio_comp)
            )

        if shape == 'cb+cb':
            shape_dict = {
                'cb_coeff_ratio': 'CB+CB: Ratio of CB components',
                'cb1_mean':       'CB+CB: Mean of CB1 component',
                'cb1_sigma':      'CB+CB: Width of CB1 component',
                'cb1_alpha':      'CB+CB: Location of transition to a power law of CB1 component',
                'cb1_n':          'CB+CB: Exponent of power-law tail of CB1 component',
                'cb2_mean':       'CB+CB: Mean of CB2 component',
                'cb2_sigma':      'CB+CB: Width of CB2 component',
                'cb2_alpha':      'CB+CB: Location of transition to a power law of CB2 component',
                'cb2_n':          'CB+CB: Exponent of power-law tail of CB2 component',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

                if 'cb_coeff_ratio' in par:
                    name_fmt_comp = par+'_comp_'+label if label else par+'_comp'
                    setattr(self, 'cb_coeff_ratio_comp', ROOT.RooFormulaVar(
                        name_fmt_comp+self.channel_label,
                        '1-{}'.format(getattr(self, par).GetName()),
                        ROOT.RooArgList(getattr(self, par)))
                    )

            self.cb1_pdf = ROOT.RooCBShape(
                'cb1_pdf'+self.channel_label,
                'CB+CB: CB1 component',
                xvar, self.cb1_mean, self.cb1_sigma, self.cb1_alpha, self.cb1_n)

            self.cb2_pdf = ROOT.RooCBShape(
                'cb2_pdf'+self.channel_label,
                'CB+CB: CB2 component',
                xvar, self.cb2_mean, self.cb2_sigma, self.cb2_alpha, self.cb2_n)

            self.model = ROOT.RooAddPdf(
                self.name+self.channel_label,
                'CB+CB',
                ROOT.RooArgList(self.cb1_pdf, self.cb2_pdf),
                ROOT.RooArgList(self.cb_coeff_ratio, self.cb_coeff_ratio_comp)
            )

        if shape == 'exp':
            shape_dict = {
                'exp_slope': 'Exp: slope of exponential',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.model = ROOT.RooExponential(self.name+self.channel_label, 'Exponential PDF', xvar, self.exp_slope)

        if shape == 'poly':
            n_polypars = sum('poly_a' in s for s in parameters.keys())
            shape_dict = {'poly_a{}'.format(i): 'Poly: {}th coeff.'.format(i) for i in range(n_polypars)}
            shape_dict.update({'poly_offset': 'Poly: x-axis offset'})

            model_pars = []
            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                roovar = ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt]
                )
                setattr(self, par, roovar)
                model_pars.append(roovar)

            self.model = ROOT.RooPolynomial(self.name+self.channel_label, 'Exponential PDF', xvar, ROOT.RooArgList(*model_pars))

        if shape == 'generic':
            shape_dict = {
                'exp_slope':  'Generic (exp*erfc): slope of exponential',
                'erfc_mean':  'Generic (exp*erfc): mean of error function',
                'erfc_sigma': 'Generic (exp*erfc): width of error function',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            function = 'TMath::Exp(TMath::Abs({})*({}-{}))*TMath::Erfc(({}-{})/{})'.format(self.exp_slope.GetName(), xvar.GetName(), self.erfc_mean.GetName(), xvar.GetName(), self.erfc_mean.GetName(), self.erfc_sigma.GetName())
            self.model = ROOT.RooGenericPdf(
                self.name+self.channel_label,
                'Generic PDF (exp*erfc)',
                function, ROOT.RooArgSet(xvar, self.erfc_mean, self.erfc_sigma, self.exp_slope)
            )

        if shape == 'kde':
            assert dataset is not None, 'Dataset required for KDE initilization'
            name_fmt = 'kde_mirror_'+label if label else 'kde_mirror'
            kde_mirror = getattr(ROOT.RooKeysPdf, *parameters[name_fmt])
            name_fmt = 'kde_rho_'+label if label else 'kde_rho'
            self.model = ROOT.RooKeysPdf(self.name+self.channel_label, 'Kernel Density Estimate PDF', xvar, dataset, kde_mirror, *parameters[name_fmt])


class FitModel:
    def __init__(self, dictionary={}):
        self.name = None
        self.branch = None
        self.dataset = None
        self.channel_label = None
        self.signal_models = {}
        self.background_models = {}
        self.constraints = {}
        self.fit_model = None
        self.fit_result = None

        for k, v in dictionary.items():
            setattr(self, k, v)

        assert self.branch is not None, "Must define variable for fit"

    def __repr__(self):
        newline = '\n'
        items = [f'    {k}={v!r}' for k, v in self.__dict__.items()]
        return f"{self.__class__.__name__}(\n{f',{newline}'.join(items)}\n)"

    def add_signal_model(self, *args, **kwds):
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], (PDFDict, PDFDictWrapper)):
            self.add_signal_model_from_object(*args, **kwds)
        else:
            self.add_signal_model_from_scratch(*args, **kwds)

    def add_signal_model_from_scratch(self, name, shape, parameters, let_float=True, dataset=None):
        fit_params = format_params(parameters, let_float)
        ds_to_use = dataset if dataset is not None else self.dataset
        sig_model = PDFDict(name, shape, self.branch, fit_params, ds_to_use, let_float, self.channel_label)
        self.signal_models[name] = sig_model
        setattr(self, sig_model.name, sig_model.model)

    def add_signal_model_from_object(self, name, model_dict):
        self.signal_models[name] = model_dict
        setattr(self, name, model_dict.model)

    def add_background_model(self, *args, **kwds):
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], (PDFDict, PDFDictWrapper)):
            self.add_background_model_from_object(*args, **kwds)
        else:
            self.add_background_model_from_scratch(*args, **kwds)

    def add_background_model_from_scratch(self, name, shape, parameters, let_float=True, dataset=None):
        fit_params = format_params(parameters, let_float)
        ds_to_use = dataset if dataset is not None else self.dataset
        bkg_model = PDFDict(name, shape, self.branch, fit_params, ds_to_use, let_float, self.channel_label)
        self.background_models[name] = bkg_model
        setattr(self, bkg_model.name, bkg_model.model)

    def add_background_model_from_object(self, name, model_dict):
        self.background_models[name] = model_dict
        setattr(self, name, model_dict.model)

    def add_composite_kde_model(self, model_name, components, dataset_params, fit_params,
                                samples_config, scale_factor_func, prepare_inputs_func,
                                yield_modifiers=None, verbose=False):

        pdf_list = ROOT.RooArgList()
        coeff_list = ROOT.RooArgList()
        component_yields = {}
        total_yield = 0

        # Default to empty dict if None
        yield_modifiers = yield_modifiers if yield_modifiers is not None else {}

        if not hasattr(self, 'memory_store'):
            self.memory_store = []

        dataset_merged = None

        for name in components:
            sample_cfg = samples_config[name]
            sf = scale_factor_func(name)

            # Load component dataset
            file_path = getattr(dataset_params, sample_cfg['file_key'])
            _, ds_comp = prepare_inputs_func(
                dataset_params, fit_params, isData=False,
                b_mass_branch=self.branch,
                set_file=file_path,
                weight_branch_name=dataset_params.mc_weight_branch,
                weight_sf=sf
            )
 
            # Merge for plotting
            if dataset_merged is None:
                dataset_merged = ds_comp.Clone(f'dataset_merged_{model_name}')
            else:
                dataset_merged.append(ds_comp)

            # Name component pdf_{name}
            pdf_sub_name = f"pdf_{name}"

            # Configure parameters
            comp_params = fit_params.fit_defaults.copy()
            if f'kde_mirror_{model_name}' in comp_params:
                comp_params[f'kde_mirror_{pdf_sub_name}'] = comp_params[f'kde_mirror_{model_name}']
            if f'kde_rho_{model_name}' in comp_params:
                comp_params[f'kde_rho_{pdf_sub_name}'] = comp_params[f'kde_rho_{model_name}']

            # Create KDE PDF
            self.add_background_model_from_scratch(
                pdf_sub_name, 'kde', comp_params, dataset=ds_comp
            )

            pdf_obj = getattr(self, pdf_sub_name)
            pdf_list.add(pdf_obj)

            # Calculate Yield
            y_exp = ds_comp.sumEntries()

            # Apply Yield Modifiers
            if name in yield_modifiers:
                scale = yield_modifiers[name]
                if verbose:
                    print(f"    * Up-weighting '{name}' by factor {scale:.3f} (Original N={y_exp:.2f} -> New N={y_exp*scale:.2f})")
                y_exp *= scale

            total_yield += y_exp
            component_yields[name] = y_exp

            # Create fixed coefficient
            coeff = ROOT.RooConstVar(f"coef_{name}", f"Expected Yield {name}", y_exp)
            coeff_list.add(coeff)
            self.memory_store.append(coeff)

            if verbose:
                print(f"  > Component {name:<20}: N_exp = {y_exp:.2f}")

        # Construct Sum PDF
        sum_pdf = ROOT.RooAddPdf(model_name, f'Combined {model_name}', pdf_list, coeff_list)

        # Inject into wrapper
        wrapper = PDFDictWrapper(model_name, sum_pdf)
        self.add_background_model_from_object(model_name, wrapper)
        self.fit_model = sum_pdf

        return total_yield, component_yields, dataset_merged

    def set_yield(self, model_name, val, min_val, max_val):
        if model_name in self.signal_models:
            wrapper = self.signal_models[model_name]
        elif model_name in self.background_models:
            wrapper = self.background_models[model_name]
        else:
            raise KeyError(f"Model '{model_name}' not found in FitModel. Did you add it first?")

        coeff_name = f"{model_name}_coeff{self.channel_label}"
        wrapper.coeff = ROOT.RooRealVar(coeff_name, f"Yield for {model_name}", val, min_val, max_val)

        return wrapper.coeff

    def add_constraints(self, constraint_dict):
        self.constraints.update(constraint_dict)

    def build_model(self, name='pdf_sum_final', title=None):
        pdf_list = ROOT.RooArgList()
        coeff_list = ROOT.RooArgList()

        def add_components(model_dict):
            for key, wrapper in model_dict.items():
                # Only include models that have a yield attached
                if hasattr(wrapper, 'coeff'):
                    pdf_list.add(wrapper.model)
                    coeff_list.add(wrapper.coeff)

        add_components(self.signal_models)
        add_components(self.background_models)

        if pdf_list.getSize() == 0:
            total_components = len(self.signal_models) + len(self.background_models)
            if total_components == 1:
                if len(self.signal_models) == 1:
                    wrapper = next(iter(self.signal_models.values()))
                else:
                    wrapper = next(iter(self.background_models.values()))

                self.fit_model = wrapper.model
                return self.fit_model

            raise RuntimeError("build_total_pdf failed: No components found with set_yield(). Cannot build empty PDF.")

        if pdf_list.getSize() == 1:
            single_pdf = pdf_list.at(0)
            self.fit_model = single_pdf
            return self.fit_model

        self.fit_model = ROOT.RooAddPdf(self.name if name is None else name, self.name if title is None else title, pdf_list, coeff_list)

        return self.fit_model

    def get_parameter(self, name):
        if not hasattr(self, 'fit_model') or self.fit_model is None:
            raise RuntimeError("Cannot retrieve parameter: fit_model has not been built yet.")

        # getVariables() returns a pointer to the live set of variables in the PDF
        all_vars = self.fit_model.getVariables()
        param = all_vars.find(name)

        if not param:
            raise ValueError(f"Parameter '{name}' not found in active fit model.")

        return param

    def fit(self, dataset, fit_range='full', fit_norm_range='full', printlevel=ROOT.RooFit.PrintLevel(-1), param_err_tolerance=1E-3, use_minos=False, asym_err=False):
        fit_args = [
            dataset,
            ROOT.RooFit.Save(),
            ROOT.RooFit.Range(fit_range),
            # ROOT.RooFit.NormRange(fit_norm_range),
            printlevel,
            # ROOT.RooFit.Extended(True),
            ROOT.RooFit.Minos(True if use_minos else False),
            ROOT.RooFit.AsymptoticError(True if asym_err else False),
        ]

        if self.constraints:
            # Avoid garbage collection of RooArgSet
            self._active_constraint_set = ROOT.RooArgSet()
            for c in self.constraints.values():
                self._active_constraint_set.add(c)

            fit_args.append(ROOT.RooFit.ExternalConstraints(self._active_constraint_set))

        self.fit_result = self.fit_model.fitTo(*fit_args)

        # Basic limit checking
        
        for param in self.fit_result.floatParsFinal():
            val = param.getVal()
            min_val = param.getMin()
            max_val = param.getMax()
            name = param.GetName()

            if abs(val - min_val) < param_err_tolerance:
                print(f'⚠️ {self.name} WARNING: Parameter "{name}" is at its lower limit ({val:.5f} ≈ {min_val:.5f})')
            elif abs(val - max_val) < param_err_tolerance:
                print(f'⚠️  {self.name} WARNING: Parameter "{name}" is at its upper limit ({val:.5f} ≈ {max_val:.5f})')

        # Fit status check
        status = self.fit_result.status()
        cov_qual = self.fit_result.covQual()
        # cov_matrix = self.fit_result.covarianceMatrix()
        # cov_matrix.Print()

        if not (status == 0 and cov_qual == 3):
            print(f'\n  ❌ {self.name} FIT ISSUES:')
            if status != 0:
                print(f'     - Minimization Failed (Status {status})')
            if cov_qual < 3:
                print(f'     - Bad Covariance Matrix (Qual {cov_qual}/3)')
            print('\n')

    def plot_fit(
        self,
        branch,
        dataset,
        output_filepath,
        fit_components=[],
        bins=None,
        fit_range='full',
        fit_norm_range='full',
        yrange=None,
        file_formats=['pdf', 'png'],
        fit_result=None,
        legend=None,
        stat_text_pos='right',
        extra_text=None,
        file_label=None,
        data_error=ROOT.RooAbsData.Auto
    ):

        assert self.fit_model is not None, "Must assign 'fit_model'"
        plot_model = self.fit_model

        if fit_range != 'full':
            dataset = dataset.reduce(ROOT.RooFit.CutRange(fit_range))

        fit_range = ROOT.RooFit.Range(fit_range)
        fit_norm_range = ROOT.RooFit.NormRange(fit_norm_range)

        hex_colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        root_colors = [ROOT.TColor.GetColor(col) for col in hex_colors]
        get_color = (col for col in root_colors)

        frame = branch.frame(
            ROOT.RooFit.Title(' '),
            fit_range,
            # fit_norm_range,
        )

        legend_coords = {
            'ul': (.1, .575, .4, .9),
            'ur': (.5, .575, .9, .9),
            'uc': (.3, .575, .7, .9),
            'll': (.1, .1, .4, .4),
            'lr': (.5, .1, .9, .4),
            'lc': (.3, .1, .7, .4)
        }

        coords = legend_coords.get(legend, (.1, .6, .4, .9))
        leg = ROOT.TLegend(*coords)

        if bins is not None:
            if isinstance(bins, int):
                nbins = bins
                bins = [bins, branch.getMin(), branch.getMax()]
            else:
                nbins = bins[0]
            bins = ROOT.RooBinning(*bins)
            dataset.plotOn(frame, ROOT.RooFit.Name(dataset.GetName()), ROOT.RooFit.Binning(bins), ROOT.RooFit.DataError(data_error))
        else:
            nbins = frame.GetNbinsX()
            dataset.plotOn(frame, ROOT.RooFit.Name(dataset.GetName()), ROOT.RooFit.DataError(data_error))

        leg.AddEntry(frame.findObject(dataset.GetName()), dataset.GetTitle(), 'PE')

        plot_model.plotOn(
            frame,
            fit_range,
            fit_norm_range,
            ROOT.RooFit.Name(plot_model.GetName()),
            ROOT.RooFit.LineStyle(ROOT.kSolid),
            ROOT.RooFit.LineColor(next(get_color)),
            # ROOT.RooFit.VisualizeError(fit_result),
        )
        leg.AddEntry(frame.findObject(plot_model.GetName()), plot_model.GetTitle(), 'L')

        h_pull = frame.pullHist()
        frame_pull = branch.frame(ROOT.RooFit.Title(' '), fit_range)
        frame_pull.addPlotable(h_pull, 'P')

        if isinstance(fit_components, list):
            for comp in fit_components:
                comp_name = f'comp_{comp.GetName()}'
                plot_argset = ROOT.RooArgSet(comp)
                plot_comp = ROOT.RooFit.Components(plot_argset)
                plot_model.plotOn(
                    frame,
                    plot_comp,
                    fit_range,
                    # fit_norm_range,
                    ROOT.RooFit.Name(comp_name),
                    ROOT.RooFit.LineStyle(ROOT.kDashed),
                    ROOT.RooFit.LineColor(next(get_color))
                )

                comp_curve = frame.findObject(comp_name)
                if comp_curve:
                    leg.AddEntry(comp_curve, comp.GetTitle(), 'L')
                else:
                    print(f"ERROR: Could not find plotted object with name: {comp_name}")

        elif isinstance(fit_components, dict):
            for name, comp in fit_components.items():
                comp_name = f'comp_{comp.GetName()}'
                plot_argset = ROOT.RooArgSet(comp)
                plot_comp = ROOT.RooFit.Components(plot_argset)
                plot_model.plotOn(
                    frame,
                    plot_comp,
                    fit_range,
                    fit_norm_range,
                    ROOT.RooFit.Name(comp_name),
                    ROOT.RooFit.LineStyle(ROOT.kDashed),
                    ROOT.RooFit.LineColor(next(get_color))
                )

                comp_curve = frame.findObject(comp_name)
                if comp_curve:
                    leg.AddEntry(comp_curve, name, 'L')
                else:
                    print(f"ERROR: Could not find plotted object with name: {comp_name}")

        if fit_result is not None:
            chi2 = frame.chiSquare(
                plot_model.GetName(),
                dataset.GetName(),
                len(fit_result.floatParsFinal()),
            )
            # fit_ndf = nbins-2-len(fit_result.floatParsFinal())
            # chi2_var = plot_model.createChi2(dataset)
            # pvalue = ROOT.Math.chisquared_cdf_c(chi2, fit_ndf)
            # pvalue = calculate_pvalue(branch, dataset, plot_model, fit_result)
            # print(pvalue)

        c = ROOT.TCanvas('c', ' ', 800, 600)
        pad1 = ROOT.TPad('pad1', 'pad1', 0, 0.3, 1, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.SetGridx()
        pad1.Draw()
        c.cd()
        pad2 = ROOT.TPad('pad2', 'pad2', 0, 0.05, 1, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.2)
        pad2.SetGridx()
        pad2.Draw()

        pad1.cd()
        frame.Draw()
        # ROOT.gPad.SetLogy()
        # ROOT.gPad.Update()
        ax_y_main = frame.GetYaxis()
        ax_x_main = frame.GetXaxis()
        ax_x_main.SetLabelOffset(3.)

        if yrange:
            ax_y_main.SetRangeUser(*yrange)

        if legend:
            leg.Draw()

        stat_text_pos_x = .48 if ('middle' in stat_text_pos) else .63
        if fit_result is not None:
            chi2_text = ROOT.TLatex(stat_text_pos_x, 0.8, '#chi^{{2}}/ndf = {}'.format(round(chi2, 2)))
            chi2_text.SetTextSize(0.06)
            chi2_text.SetNDC(ROOT.kTRUE)
            chi2_text.Draw()

            '''
            pvalue_text = ROOT.TLatex(0.6, 0.7, 'p = {}'.format(round(pvalue,3)))
            pvalue_text.SetTextSize(0.06)
            pvalue_text.SetNDC(ROOT.kTRUE)
            pvalue_text.Draw()
            '''

        if extra_text:
            text = ROOT.TLatex(stat_text_pos_x, 0.7, extra_text)
            text.SetTextSize(0.06)
            text.SetNDC(ROOT.kTRUE)
            text.Draw()

        pad2.cd()
        frame_pull.Draw()

        ax_y_pull = frame_pull.GetYaxis()
        ax_x_pull = frame_pull.GetXaxis()

        line = ROOT.TLine(ax_x_pull.GetXmin(), 0, ax_x_pull.GetXmax(), 0)
        line.SetLineStyle(7)
        line.Draw()

        ax_y_pull.SetTitle('#frac{y - y_{fit}}{#sigma_{y}}')
        ax_y_pull.SetTitleOffset(.35)
        ax_y_pull.SetNdivisions(8)
        ax_y_pull.SetRangeUser(-5, 5)

        ax_y_pull.SetTitleSize(2.8*ax_y_main.GetTitleSize())
        ax_y_pull.SetLabelSize(2.8*ax_y_main.GetLabelSize())
        ax_x_pull.SetTitleSize(2.8*ax_x_main.GetTitleSize())
        ax_x_pull.SetLabelSize(2.8*ax_x_main.GetLabelSize())

        if isinstance(output_filepath, Path):
            output_filepath = str(output_filepath)
        path_stem, path_ext = output_filepath.rsplit('.', 1)
        path_stem = path_stem if file_label is None else path_stem+f'_{file_label}'
        for fmt in file_formats:
            c.SaveAs(path_stem+'.'+fmt)
        c.Close()
