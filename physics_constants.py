"""
Physics Constants and Sample Definitions for Run 3 RK Analysis.

Units:
- Cross-sections: fb (femtobarns)
- Luminosity: 1/fb (inverse femtobarns)
- Branching Fractions: Dimensionless
"""

# ==============================================================================
# 1. GLOBAL CONSTANTS
# ==============================================================================
# LUMI_2022_EE = 38.85
LUMI_2023_EE = 24.225    # Total 2022 Lumi (1/fb)
SIGMA_BB = 4.70e11      # bb cross-section (fb)
FRAG_FRAC = 0.4         # fragmentation fraction (assuming fu = fd)

# ==============================================================================
# 2. BRANCHING FRACTIONS (PDG)
# ==============================================================================
# --- Lepton Decays ---
BR_JPSI_EE = 0.0594
BR_PSI2S_EE = 0.00772

# --- Kaon/Star Decays ---
BR_KSTAR_PLUS_KPI0 = 0.3323    # K*+ -> K+ pi0
BR_KSTAR_PLUS_K0PI = 0.6657    # K*+ -> K0 pi+
BR_K0STAR_KPI = 0.6657         # K*0 -> K+ pi-
BR_K0_KS = 0.5                 # K0 -> KS0

# --- B Decays (J/psi Modes) ---
BR_B_PLUS_JPSI_K = 0.001014        # B+ -> J/psi K+
BR_B_PLUS_JPSI_KSTAR = 0.00143     # B+ -> J/psi K*+
BR_B_PLUS_JPSI_PI = 0.000049       # B+ -> J/psi pi+
BR_B_ZERO_JPSI_KSTAR = 0.00133     # B0 -> J/psi K*0
BR_B_PLUS_CHIC1_K = 0.00046        # B+ -> chi_c1 K+
BR_CHIC1_JPSI_GAMMA = 0.344        # chi_c1 -> J/psi gamma

# --- B Decays (psi(2S) Modes) ---
BR_B_PLUS_PSI2S_K = 0.000646       # B+ -> psi(2S) K+
BR_B_PLUS_PSI2S_KSTAR = 0.00062    # B+ -> psi(2S) K*+
BR_B_ZERO_PSI2S_KSTAR = 0.00133    # B0 -> psi(2S) K*0
BR_B_PLUS_PSI2S_PI = 0.0000244     # B+ -> psi(2S) pi+

# --- B Decays (Rare/Non-Resonant Modes) ---
BR_B_PLUS_K_EE = 5.50e-7           # B+ -> K+ e+ e-
BR_B_PLUS_KSTAR_EE = 1.55e-6       # B+ -> K*+ e+ e-
BR_B_ZERO_KSTAR_EE = 1.03e-6       # B0 -> K*0 e+ e-


# ==============================================================================
# 3. SAMPLE DATABASE
# ==============================================================================
# 'file_key' corresponds to the key in fit_cfg.yml
# 'n_gen' corresponds to "N toys thrown" in the CSV

SAMPLES = {
    # ---------------------------------------------------------
    # J/psi Control Region
    # ---------------------------------------------------------
    'jpsi_resonant': {
        'file_key': 'jpsi_file',
        'label': 'B^{+} #rightarrow J/#psi K^{+}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_JPSI_K * BR_JPSI_EE,
        'n_gen': 450492489,
        'analysis_axe': 0.0,
    },
    'kstar_jpsi_kaon': {
        'file_key': 'kstar_jpsi_kaon_file',
        'label': 'B^{+} #rightarrow J/#psi K^{*+} (K^{+}#pi^{0})',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_JPSI_KSTAR * BR_KSTAR_PLUS_KPI0 * BR_JPSI_EE,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    'kstar_jpsi_pion': {
        'file_key': 'kstar_jpsi_pion_file',
        'label': 'B^{+} #rightarrow J/#psi K^{*+} (K^{0}#pi^{+})',
        'xs_prod': SIGMA_BB,
        # Note: Chain usually implies K0->KS->pipi filter in efficiency,
        # but pure BF chain for production is:
        'bf_chain': BR_B_PLUS_JPSI_KSTAR * BR_KSTAR_PLUS_K0PI * BR_JPSI_EE,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    'chic1_jpsi_kaon': {
        'file_key': 'chic1_jpsi_kaon_file',
        'label': 'B^{+} #rightarrow #chi_{c1} K^{+}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_CHIC1_K * BR_CHIC1_JPSI_GAMMA * BR_JPSI_EE,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    'jpsipi_jpsi_pion': {
        'file_key': 'jpsipi_jpsi_kaon_file',
        'label': 'B^{+} #rightarrow J/#psi #pi^{+}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_JPSI_PI * BR_JPSI_EE,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    # Note: K0* samples share the same N_GEN because they come from the same physical process
    # but represent different reconstructed candidates (K vs Pi track)
    'k0star_jpsi_kaon': {
        'file_key': 'k0star_jpsi_kaon_file',
        'label': 'B^{0} #rightarrow J/#psi K^{*0} (K^{+}#pi^{-})$ [K^{#pm} cand]',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_ZERO_JPSI_KSTAR * BR_K0STAR_KPI * BR_JPSI_EE,
        'n_gen': 81423360,
        'analysis_axe': 0.0,
    },
    'k0star_jpsi_pion': {
        'file_key': 'k0star_jpsi_pion_file',
        'label': 'B^{0} #rightarrow J/#psi K^{*0} (K^{+}#pi^{-})$ [#pi^{#pm} cand]',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_ZERO_JPSI_KSTAR * BR_K0STAR_KPI * BR_JPSI_EE,
        'n_gen': 81423360,
        'analysis_axe': 0.0,
    },

    # ---------------------------------------------------------
    # psi(2S) Control Region
    # ---------------------------------------------------------
    'psi2s_resonant': {
        'file_key': 'psi2s_file',
        'label': 'B^{+} #rightarrow #psi(2S) K^{+}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_PSI2S_K * BR_PSI2S_EE,
        'n_gen': 48337118,
        'analysis_axe': 0.0,
    },
    'kstar_psi2s_kaon': {
        'file_key': 'kstar_psi2s_kaon_file',
        'label': 'B^{+} #rightarrow #psi(2S) K^{*+} (K^{+}#pi^{0})',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_PSI2S_KSTAR * BR_KSTAR_PLUS_KPI0 * BR_PSI2S_EE,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    'kstar_psi2s_pion': {
        'file_key': 'kstar_psi2s_pion_file',
        'label': 'B^{+} #rightarrow #psi(2S) K^{*+} (K^{0}#pi^{+})',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_PSI2S_KSTAR * BR_KSTAR_PLUS_K0PI * BR_PSI2S_EE,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    'k0star_psi2s_kaon': {
        'file_key': 'k0star_psi2s_kaon_file',
        'label': 'B^{0} #rightarrow #psi(2S) K^{*0}$ [K^{#pm} cand]',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_ZERO_PSI2S_KSTAR * BR_K0STAR_KPI * BR_PSI2S_EE,
        'n_gen': 7043423,
        'analysis_axe': 0.0,
    },
    'k0star_psi2s_pion': {
        'file_key': 'k0star_psi2s_pion_file',
        'label': 'B^{0} #rightarrow #psi(2S) K^{*0}$ [#pi^{#pm} cand]',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_ZERO_PSI2S_KSTAR * BR_K0STAR_KPI * BR_PSI2S_EE,
        'n_gen': 7043423,
        'analysis_axe': 0.0,
    },
    'psi2spi_psi2s_pion': {
        'file_key': 'psi2spi_psi2s_kaon_file',
        'label': 'B^{+} #rightarrow #psi(2S) #pi^{+}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_PSI2S_PI * BR_PSI2S_EE,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },

    # ---------------------------------------------------------
    # Low-q2 (Rare) Signal Region
    # ---------------------------------------------------------
    'rare_signal': {
        'file_key': 'rare_file',
        'label': 'B^{+} #rightarrow K^{+} e^{+} e^{-}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_K_EE,
        'n_gen': 412148214,
        'analysis_axe': 0.0,
    },
    'kstar_kaon': {
        'file_key': 'kstar_kaon_file',
        'label': 'B^{+} #rightarrow K^{*+} e^{+} e^{-}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_KSTAR_EE * BR_KSTAR_PLUS_KPI0,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    'kstar_pion': {
        'file_key': 'kstar_pion_file',
        'label': 'B^{+} #rightarrow K^{*+} e^{+} e^{-}',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_PLUS_KSTAR_EE * BR_KSTAR_PLUS_K0PI,
        'n_gen': 100000000,
        'analysis_axe': 0.0,
    },
    'k0star_kaon': {
        'file_key': 'k0star_kaon_file',
        'label': 'B^{0} #rightarrow K^{*0} e^{+} e^{-}$ [K^{#pm} cand]',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_ZERO_KSTAR_EE * BR_K0STAR_KPI,
        'n_gen': 94970053,
        'analysis_axe': 0.0,
    },
    'k0star_pion': {
        'file_key': 'k0star_pion_file',
        'label': 'B^{0} #rightarrow K^{*0} e^{+} e^{-}$ [#pi^{#pm} cand]',
        'xs_prod': SIGMA_BB,
        'bf_chain': BR_B_ZERO_KSTAR_EE * BR_K0STAR_KPI,
        'n_gen': 94970053,
        'analysis_axe': 0.0,
    }
}


# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

def get_theoretical_yield(sample_key, lumi=LUMI_2023_EE, bb_xsec=SIGMA_BB, frag_frac=FRAG_FRAC):
    """
    Calculates the expected theoretical yield (Total events produced).
    Formula needed to ensure that you get at least one B meson from bb quark prod.
    N_exp = Lumi * Sigma_prod * (1 - (1 - (FRAG_FRAC * BF_chain)i)^2 )
    """
    if sample_key not in SAMPLES:
        raise KeyError(f"Sample {sample_key} not defined in physics_constants.py")

    s = SAMPLES[sample_key]
    return lumi * s['xs_prod'] * (1-(1-(frag_frac*s['bf_chain']))**2)


def get_exp_yield(sample_key, lumi=LUMI_2023_EE, bb_xsec=SIGMA_BB, frag_frac=FRAG_FRAC):
    """
    Calculates the scale factor to normalize MC to expected data luminosity.
    Scale Factor (simplified) = (Lumi * Sigma * BF * acc. * eff.) / N_generated
    """
    s = SAMPLES[sample_key]
    expected = get_theoretical_yield(sample_key, lumi)
    return expected * s['analysis_axe'] / s['n_gen']


def get_mc_scale_factor(sample_key, lumi=LUMI_2023_EE, bb_xsec=SIGMA_BB, frag_frac=FRAG_FRAC):
    """
    Calculates the scale factor to normalize raw MC events to the expected data luminosity.
    Scale Factor = Total_Theoretical_Yield / N_generated

    When applied to MC events:
    Sum(Weights) = N_pass * (N_produced / N_gen)
                 = N_produced * Efficiency
                 = Expected_Yield
    """
    if sample_key not in SAMPLES:
        raise KeyError(f"Sample {sample_key} not defined in physics_constants.py")

    s = SAMPLES[sample_key]

    # 1. Get the total number of events produced in reality (before cuts)
    n_produced = get_theoretical_yield(sample_key, lumi, bb_xsec, frag_frac)

    # 2. Divide by the total number of MC events generated to get the weight per event
    return n_produced / s['n_gen']
