#include "Particles/PulsarParameters.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXConst.H"
#include <ablastr/coarsen/sample.H>
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_Particles.H>
#include <AMReX_Scan.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Math.H>


std::string Pulsar::m_pulsar_type;
amrex::Real Pulsar::m_omega_star;
amrex::Real Pulsar::m_R_star;
amrex::Real Pulsar::m_B_star;
amrex::Real Pulsar::m_dR_star;
amrex::Real Pulsar::m_omega_ramp_time = 1.0;
amrex::Real Pulsar::m_field_damping_scale;
int Pulsar::m_do_EB_external = 0;
int Pulsar::m_do_E_external_monopole = 0;
amrex::Array<amrex::Real, 3> Pulsar::m_center_star = {{0.}};
amrex::Real Pulsar::m_max_ndens;
amrex::Real Pulsar::m_Ninj_fraction = 1.0;
int Pulsar::m_ModifyParticleWtAtInjection = 1.0;
amrex::Real Pulsar::m_rhoGJ_scale;
int Pulsar::m_do_damp_EB_internal = 0;
amrex::Real Pulsar::m_max_EBcorotating_radius;
amrex::Real Pulsar::m_max_EBdamping_radius;
int Pulsar::m_turnoffdeposition = 0;
amrex::Real Pulsar::m_max_nodepos_radius;
int Pulsar::m_turnoff_plasmaEB_gather = 0;
amrex::Real Pulsar::m_max_nogather_radius;
int Pulsar::m_verbose;
amrex::Real Pulsar::m_max_particle_absorption_radius;
amrex::Real Pulsar::m_particle_inject_rmin;
amrex::Real Pulsar::m_particle_inject_rmax;
amrex::Real Pulsar::m_cell_inject_rmin;
amrex::Real Pulsar::m_cell_inject_rmax;
amrex::Real Pulsar::m_corotatingE_maxradius;
amrex::Real Pulsar::m_enforceDipoleB_maxradius;
int Pulsar::m_do_InitializeGrid_with_Pulsar_Bfield = 1;
int Pulsar::m_do_InitializeGrid_with_Pulsar_Efield = 1;
int Pulsar::m_enforceCorotatingE = 1;
int Pulsar::m_enforceDipoleB = 1;
int Pulsar::m_singleParticleTest = 0;
int Pulsar::m_do_DampBDipoleInRing = 0;
amrex::Real Pulsar::m_Bdamping_scale;
amrex::Real Pulsar::m_injection_time = 0.;
int Pulsar::m_continuous_injection = 1;
amrex::Real Pulsar::m_removeparticle_theta_min = 180.;
amrex::Real Pulsar::m_removeparticle_theta_max = 0.;
int Pulsar::m_use_theoreticalEB = 0;
amrex::Real Pulsar::m_theory_max_rstar;
int Pulsar::m_LimitDipoleBInit = 0;
amrex::Real Pulsar::m_DipoleB_init_maxradius;
int Pulsar::m_AddExternalMonopoleOnly = 1;
int Pulsar::m_AddMonopoleInsideRstarOnGrid = 0;
int Pulsar::m_EnforceTheoreticalEBInGrid = 0;
std::string Pulsar::m_str_conductor_function;
std::unique_ptr<amrex::Parser> Pulsar::m_conductor_parser;
bool Pulsar::m_do_conductor = false;
int Pulsar::m_ApplyEfieldBCusingConductor = 0;
bool Pulsar::m_do_FilterWithConductor = false;
int Pulsar::m_do_InitializeGridWithCorotatingAndExternalEField = 0;
int Pulsar::m_AddBdipoleExternal = 0;
int Pulsar::m_AddVacuumEFieldsIntAndExt = 0;
int Pulsar::m_AddVacuumBFieldsIntAndExt = 0;
amrex::Real Pulsar::m_injection_endtime;
amrex::Real Pulsar::m_injection_rate;
amrex::Real Pulsar::m_GJ_injection_rate;
amrex::Real Pulsar::m_Sigma0_threshold;
amrex::Real Pulsar::m_Sigma0_baseline;
utils::parser::IntervalsParser Pulsar::m_injection_tuning_interval;
amrex::Real Pulsar::m_min_Sigma0;
amrex::Real Pulsar::m_max_Sigma0;
amrex::Real Pulsar::m_sum_injection_rate = 0.;
std::string Pulsar::m_sigma_tune_method;
int Pulsar::ROI_avg_window_size = 50;
int Pulsar::modify_sigma_threshold = 0.;
int Pulsar::m_print_injected_celldata;
int Pulsar::m_print_celldata_starttime;
amrex::Real Pulsar::m_lbound_ndens_magnetization = 1.e-16;
amrex::Real Pulsar::m_ubound_reldiff_sigma0 = 0.1;
amrex::Real Pulsar::m_Chi;
int Pulsar::EnforceParticleInjection = 0;
amrex::Real Pulsar::m_injection_sigma_reldiff = 0;
int Pulsar::WeightedParticleInjection = 0;
amrex::Real Pulsar::m_bufferdR_forCCBounds = 0.;
int Pulsar::TotalParticlesIsSumOfSpecies = 1;
int Pulsar::sigma_ref_avg_window_size = 0;
amrex::Real Pulsar::m_sigma_threshold_sum = 0.;
int Pulsar::use_single_sigma_ref = 0;
amrex::Real Pulsar::sigma_ref;
int Pulsar::sigma_tune_method_TCTP = 1;
int Pulsar::m_InjCell_avg_window_size = 1;
amrex::Real Pulsar::m_InjCell_sum = 0.;
amrex::Real Pulsar::m_particle_wt;
amrex::Real Pulsar::m_particle_scale_fac;
int Pulsar::m_use_Sigma0_avg = 1;
amrex::Real Pulsar::m_particle_weight_scaling = 1;
int Pulsar::m_use_LstSqFit_TC;
int Pulsar::m_use_maxsigma_for_Sigma0 = 0;
amrex::Real Pulsar::m_min_TCTP_ratio;
amrex::Real Pulsar::m_maxsigma_fraction = 1;
amrex::Real Pulsar::m_injRing_radius;
amrex::Real Pulsar::m_part_bulkVelocity = 0.0;
int Pulsar::m_pair_injection_flag = 1; // default is to inject particles in pairs
amrex::Real Pulsar::m_RLC;
amrex::Real Pulsar::m_PC_radius;
amrex::Real Pulsar::m_PC_theta;
int Pulsar::m_flag_polarcap;
int Pulsar::m_use_FixedSigmaInput = 0;       // if sigma is fixed, then inject in sigma > sigma_fixed
int Pulsar::m_onlyPCinjection = 0;           // if 1, always inject particles in PC region
int Pulsar::m_usePCflagcount_minInjCell = 1; // count cells with sigma > sigma0, and ensure it is greater than
                                     // fraction of minInjCell which is obtained from fraction * totalPCcells
amrex::Real Pulsar::m_PCInjectionCellFraction = 1.; // between  0 and 1 , 1 will choose all cells in Polar cap
amrex::Real Pulsar::m_totalpolarcap_cells;
int Pulsar::m_use_injection_rate = 1;
int Pulsar::m_GJdensity_limitinjection;
amrex::Real Pulsar::m_GJdensity_thresholdfactor = 0.;
int Pulsar::injectiontype = 0;   // 0 for GJ, 1 for sigma, 2 for hybrid
amrex::Real Pulsar::m_injection_GJdensitythreshold = 0.;
amrex::Real Pulsar::m_limit_GJfactor = 1.;
int Pulsar::m_use_BC_smoothening;
amrex::Real Pulsar::m_min_BC_radius;
amrex::Real Pulsar::m_BC_width;
amrex::Real Pulsar::m_gatherbuffer_min = 0.;
amrex::Real Pulsar::m_gatherbuffer_max = 0.;
amrex::Real Pulsar::m_depositbuffer_min = 0.;
amrex::Real Pulsar::m_depositbuffer_max = 0.;
int Pulsar::m_pml_cubic_sigma;
amrex::Real Pulsar::m_gammarad_real = 1.;
amrex::Real Pulsar::m_gammarad_scaled = 1.;
amrex::Real Pulsar::m_damping_strength = 4.;
amrex::Real Pulsar::m_totalcells_injectionring = 0;
int Pulsar::m_do_scale_re_RR = 0;
amrex::Real Pulsar::m_re_scaled;
amrex::Real Pulsar::m_gammarad_RR;
amrex::Real Pulsar::m_BLC;
amrex::Real Pulsar::m_beta_rec_RR;
amrex::Real Pulsar::m_re_scaledratio = 1.;
int Pulsar::m_do_zero_uperpB_driftframe = 0;

Pulsar::Pulsar ()
{
    ReadParameters();
}

void
Pulsar::ReadParameters () {
    amrex::Print() << " pulsar read data \n";
    amrex::ParmParse pp("pulsar");
    pp.query("pulsarType",m_pulsar_type);

    pp.get("omega_star",m_omega_star);

    amrex::Vector<amrex::Real> center_star_v(AMREX_SPACEDIM);
    pp.getarr("center_star",center_star_v);
    std::copy(center_star_v.begin(),center_star_v.end(),m_center_star.begin());

    // Obliquity of the pulsar
    pp.get("Chi", m_Chi);
    // convert degrees to radians
    m_Chi = m_Chi * MathConst::pi / 180._rt;
    amrex::Print() << "Oblique angle between B-axis and Omega-axis " << m_Chi << " radians \n";
    pp.get("R_star",m_R_star);
    pp.get("B_star",m_B_star);
    pp.get("dR",m_dR_star);
    pp.query("verbose",m_verbose);
    pp.query("EB_external",m_do_EB_external);
    pp.query("E_external_monopole",m_do_E_external_monopole);
    pp.query("damp_EB_internal",m_do_damp_EB_internal);
    pp.query("damping_scale",m_field_damping_scale);
    pp.query("ramp_omega_time",m_omega_ramp_time);
    amrex::Print() << " Pulsar center: " << m_center_star[0] << " " << m_center_star[1] << " " << m_center_star[2] << "\n";
    amrex::Print() << " Pulsar omega: " << m_omega_star << "\n";
    amrex::Print() << " Pulsar B_star : " << m_B_star << "\n";
    pp.get("max_ndens", m_max_ndens);
    pp.get("Ninj_fraction",m_Ninj_fraction);
    pp.get("plasma_injection_rate", m_injection_rate);
    // Bstar is the magnetic field strength at the equator
    // injection rate is set for the entire simulation
    //m_GJ_injection_rate = ( 8._rt * MathConst::pi * PhysConst::ep0 * m_B_star
    //                        * m_omega_star * m_omega_star * m_R_star * m_R_star * m_R_star
    //                      ) / PhysConst::q_e;
    m_GJ_injection_rate = ( 12.32_rt * MathConst::pi * PhysConst::ep0 * m_B_star
                            * m_omega_star * m_R_star * m_R_star * PhysConst::c
                          ) / PhysConst::q_e;
    amrex::Print() << " m GJ injection rate : " << m_GJ_injection_rate << "\n";
//                            * std::cos(m_Chi)
    // the factor of 2 is because B at pole = 2*B at equator
    m_Sigma0_threshold = (2. * m_B_star * 2 * m_B_star)  / (m_injection_rate * m_max_ndens * PhysConst::mu0 * PhysConst::m_e
                                         * PhysConst::c * PhysConst::c);
    amrex::Print() << " compute sigma0 at init " << m_Sigma0_threshold << "\n";
    m_Sigma0_baseline = m_Sigma0_threshold;
//    pp.get("Sigma0_threshold_init", m_Sigma0_threshold);
    amrex::Print() << " injection rate : " << m_injection_rate << " GJ injection rate " << m_GJ_injection_rate << " Sigma0 " << m_Sigma0_threshold << "\n";
//    m_max_Sigma0 = m_Sigma0_threshold * 10;
    pp.get("minimum_Sigma0", m_min_Sigma0);
    pp.get("maximum_Sigma0", m_max_Sigma0);

    m_particle_inject_rmin = m_R_star - m_dR_star;
    m_particle_inject_rmax = m_R_star;
    pp.query("particle_inj_rmin", m_particle_inject_rmin);
    pp.query("particle_inj_rmax", m_particle_inject_rmax);
    amrex::Print() << " min radius of particle injection : " << m_particle_inject_rmin << "\n";
    amrex::Print() << " max radius of particle injection : " << m_particle_inject_rmax << "\n";
    pp.get("cell_inj_rmin", m_cell_inject_rmin);
    pp.get("cell_inj_rmax", m_cell_inject_rmax);

    pp.query("ModifyParticleWeight", m_ModifyParticleWtAtInjection);
    // The maximum radius within which particles are absorbed/deleted every timestep.
    m_max_particle_absorption_radius = m_R_star;
    pp.query("max_particle_absorption_radius", m_max_particle_absorption_radius);
    pp.get("rhoGJ_scale",m_rhoGJ_scale);
    amrex::Print() << " pulsar max ndens " << m_max_ndens << "\n";
    amrex::Print() << " pulsar ninj fraction " << m_Ninj_fraction << "\n";
    amrex::Print() << " pulsar modify particle wt " << m_ModifyParticleWtAtInjection << "\n";
    amrex::Print() << " pulsar rhoGJ scaling " << m_rhoGJ_scale << "\n";
    amrex::Print() << " EB_external : " << m_do_EB_external << "\n";
    if (m_do_EB_external == 1) {
        // Max corotating radius defines the region where the EB field shifts from
        // corotating (v X B) to quadrapole. default is R_star.
        m_max_EBcorotating_radius = m_R_star;
        pp.query("EB_corotating_maxradius", m_max_EBcorotating_radius);
        amrex::Print() << " EB coratating maxradius : " << m_max_EBcorotating_radius << "\n";
    }
    if (m_do_damp_EB_internal == 1) {
        // Radius of the region within which the EB fields are damped
        m_max_EBdamping_radius = m_R_star;
        pp.query("damp_EB_radius", m_max_EBdamping_radius);
        amrex::Print() << " max EB damping radius : " << m_max_EBdamping_radius << "\n";
    }
    // query to see if particle j,rho deposition should be turned off
    // within some region interior to the star
    // default is 0
    pp.query("turnoffdeposition", m_turnoffdeposition);
    amrex::Print() << " is deposition off ? " << m_turnoffdeposition << "\n";
    if (m_turnoffdeposition == 1) {
        m_max_nodepos_radius = m_R_star;
        pp.query("max_nodepos_radius", m_max_nodepos_radius);
        amrex::Print() << " deposition turned off within radius : " << m_max_nodepos_radius << "\n";
    }
    pp.query("turnoff_plasmaEB_gather", m_turnoff_plasmaEB_gather);
    amrex::Print() << " is plasma EB gather off ? " << m_turnoff_plasmaEB_gather << "\n";
    if (m_turnoff_plasmaEB_gather == 1) {
        m_max_nogather_radius = m_R_star;
        pp.query("max_nogather_radius", m_max_nogather_radius);
        amrex::Print() << " gather off within radius : " << m_max_nogather_radius << "\n";
    }
    m_corotatingE_maxradius = m_R_star;
    m_enforceDipoleB_maxradius = m_R_star;
    pp.query("corotatingE_maxradius", m_corotatingE_maxradius);
    pp.query("enforceDipoleB_maxradius", m_enforceDipoleB_maxradius);
    pp.query("init_dipoleBfield", m_do_InitializeGrid_with_Pulsar_Bfield);
    amrex::Print() << " init dipole : " << m_do_InitializeGrid_with_Pulsar_Bfield << "\n";
    pp.query("init_corotatingEfield", m_do_InitializeGrid_with_Pulsar_Efield);
    amrex::Print() << " init cor E : " << m_do_InitializeGrid_with_Pulsar_Efield << "\n";
    pp.query("init_corotatingAndExternalEField", m_do_InitializeGridWithCorotatingAndExternalEField);
    amrex::Print() << " init cor E and ext E : " << m_do_InitializeGridWithCorotatingAndExternalEField<< "\n";
    pp.query("enforceCorotatingE", m_enforceCorotatingE);
    pp.query("enforceDipoleB", m_enforceDipoleB);
    pp.query("singleParticleTest", m_singleParticleTest);
    pp.query("DampBDipoleInRing", m_do_DampBDipoleInRing);
    if (m_do_DampBDipoleInRing == 1) pp.query("Bdamping_scale", m_Bdamping_scale);
    pp.query("injection_time", m_injection_time);
    pp.query("continuous_injection", m_continuous_injection);
    pp.query("removeparticle_theta_min",m_removeparticle_theta_min);
    pp.query("removeparticle_theta_max",m_removeparticle_theta_max);
    pp.query("use_theoreticalEB",m_use_theoreticalEB);
    amrex::Print() << "use theory EB " << m_use_theoreticalEB << "\n";
    pp.query("theory_max_rstar",m_theory_max_rstar);
    amrex::Print() << " theory max rstar : " << m_theory_max_rstar << "\n";
    pp.query("LimitDipoleBInit", m_LimitDipoleBInit);
    if (m_LimitDipoleBInit == 1) {
       pp.query("DipoleB_init_maxradius", m_DipoleB_init_maxradius);
    }
    pp.query("AddExternalMonopoleOnly", m_AddExternalMonopoleOnly);
    pp.query("AddBdipoleExternal", m_AddBdipoleExternal);
    pp.query("AddMonopoleInsideRstarOnGrid", m_AddMonopoleInsideRstarOnGrid);
    pp.query("EnforceTheoreticalEBInGrid", m_EnforceTheoreticalEBInGrid);
    if (pp.query("conductor_function(x,y,z)", m_str_conductor_function)) {
        utils::parser::Store_parserString(pp, "conductor_function(x,y,z)", m_str_conductor_function);
        m_conductor_parser = std::make_unique<amrex::Parser>(
                                  utils::parser::makeParser(m_str_conductor_function,{"x","y","z"}));
        m_do_conductor = true;
        amrex::Print() << " do conductor : " << m_do_conductor << "\n";
        pp.query("ApplyEfieldBCusingConductor", m_ApplyEfieldBCusingConductor);
        amrex::Print() << " ApplyEfieldBCusingConductor " << m_ApplyEfieldBCusingConductor << "\n";
        int FilterWithConductor = 0;
        pp.query("FilterWithConductor", FilterWithConductor);
        if (FilterWithConductor == 1) {
            m_do_FilterWithConductor = true;
        } else {
            m_do_FilterWithConductor = false;
        }
    }
    pp.query("AddVacuumEFieldsIntAndExt", m_AddVacuumEFieldsIntAndExt );
    pp.query("AddVacuumBFieldsIntAndExt", m_AddVacuumBFieldsIntAndExt );
    m_injection_endtime = 1000; // 1000 s upper limit
    pp.query("ParticleInjectionEndTime",m_injection_endtime);
    std::vector<std::string> intervals_string_vec = {"0"};
    pp.getarr("injection_tuning_interval", intervals_string_vec);
    m_injection_tuning_interval = utils::parser::IntervalsParser(intervals_string_vec);
    pp.get("sigma_tune_method",m_sigma_tune_method);
    amrex::Print() << " sigma tune method " << m_sigma_tune_method << "\n";
    pp.get("ROI_avg_size",ROI_avg_window_size);
    pp.get("modify_sigma_threshold", modify_sigma_threshold);
    pp.get("print_injected_celldata", m_print_injected_celldata);
    pp.get("print_celldata_starttime", m_print_celldata_starttime);
    pp.query("lowerBound_ndens_magnetization", m_lbound_ndens_magnetization);
    if (m_sigma_tune_method == "relative_difference") {
        pp.query("upperBound_reldiff_sigma0", m_ubound_reldiff_sigma0);
    }
    pp.query("EnforceParticleInjection",EnforceParticleInjection);
    amrex::Print() << " enforce particle inj : " << EnforceParticleInjection << "\n";
    pp.query("injection_sigma_reldiff", m_injection_sigma_reldiff);
    amrex::Print() << " injection sigma rel diff : " << m_injection_sigma_reldiff << "\n";
    pp.query("WeightedParticleInjection",WeightedParticleInjection);
    amrex::Print() << " weighted particle injection : " << WeightedParticleInjection << "\n";
    pp.query("BufferdRForCCBounds",m_bufferdR_forCCBounds);
    amrex::Print() << " buffer dR for CC Bounds \n";
    pp.query("TotalParticlesIsSumOfSpecies", TotalParticlesIsSumOfSpecies);
    amrex::Print() << " total particles is sum of species " << TotalParticlesIsSumOfSpecies << "\n";
    pp.query("sigma_ref_avg_window_size", sigma_ref_avg_window_size);
    amrex::Print() << " sigma ref avg window size : " << sigma_ref_avg_window_size << "\n";
    pp.query("use_single_sigma_ref",use_single_sigma_ref);
    if (use_single_sigma_ref == 1) {
        pp.query("sigma_ref",sigma_ref);
    }
    pp.query("sigma_tune_method_TCTP", sigma_tune_method_TCTP);
    pp.query("InjCell_avg_window_size",m_InjCell_avg_window_size);
    // to average Sigma0_threshold new and old
    pp.query("use_Sigma0_avg", m_use_Sigma0_avg);
    pp.query("particle_weight_scaling",m_particle_weight_scaling);
    pp.get("use_LstSqFit_TC",m_use_LstSqFit_TC);
    pp.get("use_maxsigma_for_Sigm0_ifTC0",m_use_maxsigma_for_Sigma0);
    pp.get("min_TCTP_ratio",m_min_TCTP_ratio);
    pp.get("maxsigma_fraction", m_maxsigma_fraction);
    pp.get("injRing_radius", m_injRing_radius);
    pp.get("BulkVelocity", m_part_bulkVelocity);
    pp.query("pair_injection", m_pair_injection_flag);
    // polar cap radius
    m_RLC = PhysConst::c / m_omega_star;
    m_PC_radius = m_R_star * std::sqrt(m_R_star/m_RLC);
    m_PC_theta = std::asin(m_PC_radius/m_R_star);
    pp.query("use_sigmainput",m_use_FixedSigmaInput);
    if (m_use_FixedSigmaInput == 1) {
        pp.get("sigma0_threshold", m_Sigma0_threshold);
        m_flag_polarcap = 1;
        m_onlyPCinjection = 0;
        pp.get("usePCflagcount_minInjCell",m_usePCflagcount_minInjCell);
        pp.get("PCInjectionCellFraction",m_PCInjectionCellFraction);
        amrex::Print() << " m sigma0 " << m_Sigma0_threshold << "\n";
    }
    pp.query("onlyPCinjection",m_onlyPCinjection);
    amrex::Print() << " RLC : " << m_RLC << "\n";
    amrex::Print() << " PC radius " << m_PC_radius << "\n";
    amrex::Print() << " PC theta " << m_PC_theta << "\n";
    pp.query("use_injection_rate",m_use_injection_rate);
    pp.get("GJdensity_limitedinjection",m_GJdensity_limitinjection);
    pp.get("limitGJfactor",m_limit_GJfactor);
    pp.query("GJdensity_thresholdfactor",m_GJdensity_thresholdfactor);
    pp.query("injectiontype", injectiontype);   // 0 for GJ, 1 for sigma, 2 for hybrid
    pp.query("GJdensity_injectionthreshold",m_injection_GJdensitythreshold);
    pp.get("use_BC_smoothening",m_use_BC_smoothening);
    pp.get("min_BC_radius", m_min_BC_radius);
    pp.get("BC_width", m_BC_width);
    pp.query("gatherbuffer_min",m_gatherbuffer_min);
    pp.query("gatherbuffer_max",m_gatherbuffer_max);
    pp.query("depositbuffer_min",m_depositbuffer_min);
    pp.query("depositbuffer_max",m_depositbuffer_max);
    pp.get("pml_cubic_sigma", m_pml_cubic_sigma);
    pp.query("gammarad_real", m_gammarad_real);
    pp.query("gammarad_scaled", m_gammarad_scaled);
    pp.get("damping_strength", m_damping_strength);
    amrex::Print() << " damping strength in PML " << m_damping_strength << "\n";
    amrex::Print() << " cubic pml flag : " << m_pml_cubic_sigma  << "\n";
    // for radiation reaction with scaled electron radius
    pp.get("scale_re_for_RR", m_do_scale_re_RR);
    if (m_do_scale_re_RR == 1) {
        pp.get("gammarad_RR",m_gammarad_RR); // gammathreshold at which re is scaled
        pp.get("beta_rec_RR",m_beta_rec_RR); // reconnection rate
        amrex::Real S = m_RLC/m_R_star; // Scale separation ratio
        m_BLC = m_B_star/(S*S*S); // Bfield at LC
        m_re_scaled = m_beta_rec_RR * (3./2.) * PhysConst::q_e
                    / ( 4 * MathConst::pi * PhysConst::ep0 * m_gammarad_RR * m_gammarad_RR * PhysConst::c * m_BLC);
        m_re_scaled = std::sqrt( m_re_scaled);
        amrex::Print() << " re_scaled at gammarad : " << m_gammarad_RR << " is : " << m_re_scaled << "\n";
        m_re_scaledratio = m_re_scaled / PhysConst::r_e;
        amrex::Print() << " ratio : re_scaled/re : " << m_re_scaledratio << "\n";
    }
    pp.get("do_zero_uperpB_driftframe",m_do_zero_uperpB_driftframe);
    amrex::Print() << " do zero uperp B " << m_do_zero_uperpB_driftframe << "\n";
}


void
Pulsar::InitDataAtRestart ()
{
    amrex::Print() << " pulsar init data at restart \n";
    auto & warpx = WarpX::GetInstance();
    const int nlevs_max = warpx.finestLevel() + 1;
    //allocate number density multifab
    m_plasma_number_density.resize(nlevs_max);
    m_magnetization.resize(nlevs_max);
    m_injection_flag.resize(nlevs_max);
    m_injected_cell.resize(nlevs_max);
    m_sigma_reldiff.resize(nlevs_max);
    m_pcount.resize(nlevs_max);
    m_injection_ring.resize(nlevs_max);
    m_injection_ringCC.resize(nlevs_max);
    m_sigma_inj_ring.resize(nlevs_max);
    m_sigma_threshold.resize(nlevs_max);
    amrex::ParmParse pp_particles("particles");
    std::vector<std::string> species_names;
    pp_particles.queryarr("species_names", species_names);
    const int ndensity_comps = species_names.size();
    const int magnetization_comps = 1;

    for (int lev = 0; lev < nlevs_max; ++lev) {
        amrex::BoxArray ba = warpx.boxArray(lev);
        amrex::DistributionMapping dm = warpx.DistributionMap(lev);
        const amrex::IntVect ng_EB_alloc = warpx.getngEB();
        // allocate cell-centered number density multifab
        m_plasma_number_density[lev] = std::make_unique<amrex::MultiFab>(
                                     ba, dm, ndensity_comps, ng_EB_alloc);
        // allocate cell-centered magnetization multifab
        m_magnetization[lev] = std::make_unique<amrex::MultiFab>(
                               ba, dm, magnetization_comps, ng_EB_alloc);
        // allocate multifab to store flag for cells injected with particles
        m_injection_flag[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_injected_cell[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_sigma_reldiff[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_pcount[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_injection_ring[lev] = std::make_unique<amrex::MultiFab>(
                                amrex::convert(ba,amrex::IntVect::TheNodeVector()), dm, 1, ng_EB_alloc);
        m_injection_ringCC[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_sigma_inj_ring[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_sigma_threshold[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
 
        // initialize number density
        m_plasma_number_density[lev]->setVal(0._rt);
        // initialize magnetization
        m_magnetization[lev]->setVal(0._rt);
        // initialize flag for plasma injection
        m_injection_flag[lev]->setVal(0._rt);
        // rel diff
        m_sigma_reldiff[lev]->setVal(0._rt);
        m_pcount[lev]->setVal(0._rt);
        m_injection_ring[lev]->setVal(0._rt);
        m_injection_ringCC[lev]->setVal(0._rt);
        m_sigma_inj_ring[lev]->setVal(0._rt);
        m_sigma_threshold[lev]->setVal(0._rt);

        FlagCellsInInjectionRing(m_injection_ring[lev].get(), lev, m_cell_inject_rmin, m_cell_inject_rmax);
    }

    if (m_flag_polarcap == 1) {
        m_PC_flag.resize(nlevs_max);
        for (int lev = 0; lev < nlevs_max; ++lev) {
            amrex::BoxArray ba = warpx.boxArray(lev);
            amrex::DistributionMapping dm = warpx.DistributionMap(lev);
            amrex::IntVect PC_nodal_flag = amrex::IntVect::TheNodeVector();
            const amrex::IntVect ng_EB_alloc = warpx.getngEB() + amrex::IntVect::TheNodeVector()*2;
            m_PC_flag[lev] = std::make_unique<amrex::MultiFab>(
                                    amrex::convert(ba,PC_nodal_flag), dm, 1, ng_EB_alloc);
            FlagCellsInPolarCap(m_PC_flag[lev].get(),lev,m_cell_inject_rmin,m_cell_inject_rmax);
        }
        m_totalpolarcap_cells = SumPolarCapFlag();
        amrex::Print() << " total cells in PC " << m_totalpolarcap_cells << "\n";
    }

    if (m_do_conductor == true) {
        m_conductor_fp.resize(nlevs_max);
        amrex::IntVect conductor_nodal_flag = amrex::IntVect::TheNodeVector();
        const int ncomps = 1;
        for (int lev = 0; lev < nlevs_max; ++lev) {
            amrex::BoxArray ba = warpx.boxArray(lev);
            amrex::DistributionMapping dm = warpx.DistributionMap(lev);
            const amrex::IntVect ng_EB_alloc = warpx.getngEB() + amrex::IntVect::TheNodeVector()*2;
            m_conductor_fp[lev] = std::make_unique<amrex::MultiFab>(
                                    amrex::convert(ba,conductor_nodal_flag), dm, ncomps, ng_EB_alloc);
            InitializeConductorMultifabUsingParser(m_conductor_fp[lev].get(), m_conductor_parser->compile<3>(), lev);
        }
    }

}

void
Pulsar::InitData ()
{
    amrex::Print() << " pulsar init data \n";
    auto & warpx = WarpX::GetInstance();
    const int nlevs_max = warpx.finestLevel() + 1;
    //allocate number density multifab
    m_plasma_number_density.resize(nlevs_max);
    m_magnetization.resize(nlevs_max);
    m_injection_flag.resize(nlevs_max);
    m_sigma_reldiff.resize(nlevs_max);
    m_injected_cell.resize(nlevs_max);
    m_pcount.resize(nlevs_max);
    m_injection_ring.resize(nlevs_max);
    m_injection_ringCC.resize(nlevs_max);
    m_sigma_inj_ring.resize(nlevs_max);
    m_sigma_threshold.resize(nlevs_max);
    amrex::ParmParse pp_particles("particles");
    std::vector<std::string> species_names;
    pp_particles.queryarr("species_names", species_names);
    const int ndensity_comps = species_names.size();
    const int magnetization_comps = 1;

    for (int lev = 0; lev < nlevs_max; ++lev) {
        amrex::BoxArray ba = warpx.boxArray(lev);
        amrex::DistributionMapping dm = warpx.DistributionMap(lev);
        const amrex::IntVect ng_EB_alloc = warpx.getngEB();
        // allocate cell-centered number density multifab
        m_plasma_number_density[lev] = std::make_unique<amrex::MultiFab>(
                                     ba, dm, ndensity_comps, ng_EB_alloc);
        // allocate cell-centered magnetization multifab
        m_magnetization[lev] = std::make_unique<amrex::MultiFab>(
                               ba, dm, magnetization_comps, ng_EB_alloc);
        // allocate multifab to store flag for cells injected with particles
        m_injection_flag[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_injected_cell[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_sigma_reldiff[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_pcount[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_injection_ring[lev] = std::make_unique<amrex::MultiFab>(
                                amrex::convert(ba,amrex::IntVect::TheNodeVector() ), dm, 1, ng_EB_alloc);
        m_injection_ringCC[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_sigma_inj_ring[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        m_sigma_threshold[lev] = std::make_unique<amrex::MultiFab>(
                                ba, dm, 1, ng_EB_alloc);
        // initialize number density
        m_plasma_number_density[lev]->setVal(0._rt);
        // initialize magnetization
        m_magnetization[lev]->setVal(0._rt);
        m_injection_flag[lev]->setVal(0._rt);
        m_injected_cell[lev]->setVal(0._rt);
        // rel diff
        m_sigma_reldiff[lev]->setVal(0._rt);
        m_pcount[lev]->setVal(0._rt);
        m_injection_ring[lev]->setVal(0._rt);
        m_injection_ringCC[lev]->setVal(0._rt);
        m_sigma_inj_ring[lev]->setVal(0._rt);
        m_sigma_threshold[lev]->setVal(0._rt);

        FlagCellsInInjectionRing(m_injection_ring[lev].get(), lev, m_cell_inject_rmin, m_cell_inject_rmax);
    }

//    if (m_flag_polarcap == 1) {
        m_PC_flag.resize(nlevs_max);
        for (int lev = 0; lev < nlevs_max; ++lev) {
            amrex::BoxArray ba = warpx.boxArray(lev);
            amrex::DistributionMapping dm = warpx.DistributionMap(lev);
            amrex::IntVect PC_nodal_flag = amrex::IntVect::TheNodeVector();
            const amrex::IntVect ng_EB_alloc = warpx.getngEB() + amrex::IntVect::TheNodeVector()*2;
            m_PC_flag[lev] = std::make_unique<amrex::MultiFab>(
                                    amrex::convert(ba,PC_nodal_flag), dm, 1, ng_EB_alloc);
            FlagCellsInPolarCap(m_PC_flag[lev].get(),lev,m_cell_inject_rmin,m_cell_inject_rmax);
//        }
        m_totalpolarcap_cells = SumPolarCapFlag();
        amrex::Print() << " total cells in PC " << m_totalpolarcap_cells << "\n"; 
    }

    if (m_do_conductor == true) {
        m_conductor_fp.resize(nlevs_max);
        amrex::IntVect conductor_nodal_flag = amrex::IntVect::TheNodeVector();
        const int ncomps = 1;
        for (int lev = 0; lev < nlevs_max; ++lev) {
            amrex::BoxArray ba = warpx.boxArray(lev);
            amrex::DistributionMapping dm = warpx.DistributionMap(lev);
            const amrex::IntVect ng_EB_alloc = warpx.getngEB() + amrex::IntVect::TheNodeVector()*2;
            m_conductor_fp[lev] = std::make_unique<amrex::MultiFab>(
                                    amrex::convert(ba,conductor_nodal_flag), dm, ncomps, ng_EB_alloc);
            InitializeConductorMultifabUsingParser(m_conductor_fp[lev].get(), m_conductor_parser->compile<3>(), lev);
        }
    }
    for (int lev = 0; lev < nlevs_max; ++lev) {
        if (m_do_InitializeGrid_with_Pulsar_Bfield == 1) {
            bool Init_Bfield = true;
            InitializeExternalPulsarFieldsOnGrid (warpx.get_pointer_Bfield_fp(lev, 0),
                                                 warpx.get_pointer_Bfield_fp(lev, 1),
                                                 warpx.get_pointer_Bfield_fp(lev, 2),
                                                 lev, Init_Bfield);
            if (lev > 0) {
                InitializeExternalPulsarFieldsOnGrid (warpx.get_pointer_Bfield_aux(lev, 0),
                                                     warpx.get_pointer_Bfield_aux(lev, 1),
                                                     warpx.get_pointer_Bfield_aux(lev, 2),
                                                     lev, Init_Bfield);
                InitializeExternalPulsarFieldsOnGrid (warpx.get_pointer_Bfield_cp(lev, 0),
                                                     warpx.get_pointer_Bfield_cp(lev, 1),
                                                     warpx.get_pointer_Bfield_cp(lev, 2),
                                                     lev, Init_Bfield);
            }
        }
        if (m_do_InitializeGrid_with_Pulsar_Efield == 1 || m_do_InitializeGridWithCorotatingAndExternalEField == 1) {
            bool Init_Bfield = false;
            InitializeExternalPulsarFieldsOnGrid (warpx.get_pointer_Efield_fp(lev, 0),
                                                 warpx.get_pointer_Efield_fp(lev, 1),
                                                 warpx.get_pointer_Efield_fp(lev, 2),
                                                 lev, Init_Bfield);
            if (lev > 0) {
                InitializeExternalPulsarFieldsOnGrid (warpx.get_pointer_Efield_aux(lev, 0),
                                                     warpx.get_pointer_Efield_aux(lev, 1),
                                                     warpx.get_pointer_Efield_aux(lev, 2),
                                                     lev, Init_Bfield);
                InitializeExternalPulsarFieldsOnGrid (warpx.get_pointer_Efield_cp(lev, 0),
                                                     warpx.get_pointer_Efield_cp(lev, 1),
                                                     warpx.get_pointer_Efield_cp(lev, 2),
                                                     lev, Init_Bfield);
            }
        }
    }

}


void
Pulsar::FlagCellsInInjectionRing(const int lev)
{
    FlagCellsInInjectionRing(m_injection_ring[lev].get(), lev, m_cell_inject_rmin, m_cell_inject_rmax);
}

void
Pulsar::FlagCellsInInjectionRing(
        amrex::MultiFab *mf, const int lev, amrex::Real cell_inject_rmin, amrex::Real cell_inject_rmax)
{
    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = mf->ixType().toIntVect();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xc;
    for (int idim = 0; idim < 3; ++idim) {
        xc[idim] = m_center_star[idim];
    }

    for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // includes guard cells
        const amrex::Box& tb = mfi.tilebox(iv, mf->nGrowVect());
        amrex::Array4<amrex::Real> const& inj_ring_fab = mf->array(mfi);
        amrex::ParallelFor(tb,
             [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type
                amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                amrex::Real r, theta, phi;
                ConvertCartesianToSphericalCoord(x, y, z, xc,
                                                 r, theta, phi);
                amrex::Real eps = 1.e-12;
                inj_ring_fab(i,j,k) = 0.;
                if ( (r > (cell_inject_rmin - eps)) and (r < (cell_inject_rmax + eps)) ){
                    inj_ring_fab(i,j,k) = 1.;
                }
            }
        );
    }

}


void
Pulsar::FlagCellsInPolarCap(const int lev)
{
    if (m_flag_polarcap == 1) {
        FlagCellsInPolarCap(m_PC_flag[lev].get(),lev,m_cell_inject_rmin,m_cell_inject_rmax);
    }
    if (lev == 0) {
        m_totalpolarcap_cells = SumPolarCapFlag();
        amrex::Print() << " total cells in PC " << m_totalpolarcap_cells << "\n";
    }
}

void
Pulsar::FlagCellsInPolarCap(
        amrex::MultiFab *mf, const int lev, amrex::Real cell_inject_rmin, amrex::Real cell_inject_rmax)
{
    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = mf->ixType().toIntVect();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xc;
    for (int idim = 0; idim < 3; ++idim) {
        xc[idim] = m_center_star[idim];
    }
    amrex::Real PC_theta = m_PC_theta;
    amrex::Real PC_radius = m_PC_radius;
    amrex::Print() << "pc theta : " << PC_theta << "\n";
    amrex::Real Chi = m_Chi;

    for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // includes guard cells
        const amrex::Box& tb = mfi.tilebox(iv, mf->nGrowVect());
        amrex::Array4<amrex::Real> const& PC_fab = mf->array(mfi);
        amrex::ParallelFor(tb,
             [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type
                amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                amrex::Real r, theta, phi;
                ConvertCartesianToSphericalCoord(x, y, z, xc,
                                                 r, theta, phi);
                amrex::Real eps = 1.e-12;
                PC_fab(i,j,k) = 0.;
                if ( (r > (cell_inject_rmin - eps)) and (r < (cell_inject_rmax + eps)) ){
                    amrex::Real rcyl = std::sqrt( (x-xc[0])*(x-xc[0]) + (y-xc[1])*(y-xc[1]) );
                    //if (rcyl < (cell_inject_rmax * std::sin(PC_theta))) {
//                    if (rcyl < (PC_radius)) {
                       //( (theta < (0.5*3.14 + PC_theta) ) and ( theta > (0.5*3.14_rt - PC_theta)) ) ) {
                    if( (theta < (Chi + 3.14) ) and ( theta > ( Chi + 3.14_rt - PC_theta)) ) {
                        PC_fab(i,j,k) = 1.;
                    } else if ((theta < ( Chi + PC_theta) ) and (theta > (Chi) )) {
                        PC_fab(i,j,k) = 1.;
                    }
                }
             }
        );
    }

}

void
Pulsar::InitializeConductorMultifabUsingParser(const int lev)
{
    InitializeConductorMultifabUsingParser(m_conductor_fp[lev].get(), m_conductor_parser->compile<3>(), lev);
}

void
Pulsar::InitializeConductorMultifabUsingParser(
        amrex::MultiFab *mf, amrex::ParserExecutor<3> const& conductor_parser,
        const int lev)
{
    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = mf->ixType().toIntVect();
    for ( amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        //InitializeGhost Cells also
        const amrex::Box& tb = mfi.tilebox(iv, mf->nGrowVect());
        amrex::Array4<amrex::Real> const& conductor_fab = mf->array(mfi);
        amrex::ParallelFor(tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type
                amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // initialize the macroparameter
                conductor_fab(i,j,k) = conductor_parser(x,y,z);
            }
        );
    }
}

/** To initialize the grid with dipole magnetic field everywhere and corotating vacuum
 *  electric field inside the pulsar radius.
 */
void
Pulsar::InitializeExternalPulsarFieldsOnGrid ( amrex::MultiFab *mfx, amrex::MultiFab *mfy,
    amrex::MultiFab *mfz, const int lev, const bool init_Bfield)
{
    auto & warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(lev).CellSizeArray();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    amrex::Real cur_time = warpx.gett_new(lev);
    amrex::IntVect x_nodal_flag = mfx->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = mfy->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = mfz->ixType().toIntVect();
    amrex::GpuArray<int, 3> x_IndexType;
    amrex::GpuArray<int, 3> y_IndexType;
    amrex::GpuArray<int, 3> z_IndexType;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;
    for (int idim = 0; idim < 3; ++idim) {
        x_IndexType[idim] = x_nodal_flag[idim];
        y_IndexType[idim] = y_nodal_flag[idim];
        z_IndexType[idim] = z_nodal_flag[idim];
        center_star_arr[idim] = m_center_star[idim];
    }
    amrex::Real chi = m_Chi;
    amrex::Real omega_star_data = m_omega_star;
    amrex::Real ramp_omega_time_data = m_omega_ramp_time;
    amrex::Real Bstar_data = m_B_star;
    amrex::Real Rstar_data = m_R_star;
    amrex::Real dRstar_data = m_dR_star;
    int LimitDipoleBInit_data = m_LimitDipoleBInit;
    amrex::Real DipoleB_init_maxradius_data = m_DipoleB_init_maxradius;
    amrex::Real corotatingE_maxradius_data = m_corotatingE_maxradius;
    int InitializeFullExternalEField = m_do_InitializeGridWithCorotatingAndExternalEField;
    int E_external_monopole_data = m_do_E_external_monopole;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*mfx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tbx = mfi.tilebox(x_nodal_flag, mfx->nGrowVect() );
        const amrex::Box& tby = mfi.tilebox(y_nodal_flag, mfy->nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox(z_nodal_flag, mfz->nGrowVect() );

        amrex::Array4<amrex::Real> const& mfx_arr = mfx->array(mfi);
        amrex::Array4<amrex::Real> const& mfy_arr = mfy->array(mfi);
        amrex::Array4<amrex::Real> const& mfz_arr = mfz->array(mfi);
        amrex::Array4<amrex::Real> const& conductor = m_conductor_fp[lev]->array(mfi);

        amrex::ParallelFor (tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, x_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                // Initialize with Bfield in spherical coordinates
                if (init_Bfield == 1) {
                    if (LimitDipoleBInit_data == 1) {
                        if (r < DipoleB_init_maxradius_data) {
                            ExternalBFieldSpherical( r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                        }
                    } else {
                        ExternalBFieldSpherical( r, theta, phi, chi, cur_time,
                                                omega_star_data,
                                                ramp_omega_time_data,
                                                Bstar_data, Rstar_data, dRstar_data,
                                                Fr, Ftheta, Fphi);
                    }
                }
                if (init_Bfield == 0) {
                    if (InitializeFullExternalEField == 0) {
                        if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1 ) {
                            // Edge-centered Efield is fully immersed in conductor
                            // Therefore, apply corotating electric field E = v X B
                            CorotatingEfieldSpherical(r, theta, phi, chi, cur_time,
                                                      omega_star_data,
                                                      ramp_omega_time_data,
                                                      Bstar_data, Rstar_data, dRstar_data,
                                                      Fr, Ftheta, Fphi);
                        }
                    } else {
                        int ApplyCorotatingEField = 0;
                        // Apply corotating electric field E = v X B, if edge-centered Efield is fully immersed
                        // in conductor
                        if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1) ApplyCorotatingEField = 1;
                        ExternalEFieldSpherical(r, theta, phi, chi, cur_time,
                                                omega_star_data,
                                                ramp_omega_time_data,
                                                Bstar_data, Rstar_data,
                                                corotatingE_maxradius_data,
                                                E_external_monopole_data, ApplyCorotatingEField,
                                                Fr, Ftheta, Fphi);
                    }
                }
                // Convert to x component
                ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                    r, theta, phi, mfx_arr(i,j,k));
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, y_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                // Initialize with Bfield in spherical coordinates
                if (init_Bfield == 1) {
                    if (LimitDipoleBInit_data == 1) {
                        if (r < DipoleB_init_maxradius_data) {
                            ExternalBFieldSpherical( r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                        }
                    } else {
                        ExternalBFieldSpherical( r, theta, phi, chi, cur_time,
                                                omega_star_data,
                                                ramp_omega_time_data,
                                                Bstar_data, Rstar_data, dRstar_data,
                                                Fr, Ftheta, Fphi);
                    }
                }
                if (init_Bfield == 0) {
                    if (InitializeFullExternalEField == 0) {
                        if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1 ) {
                            // Edge-centered Efield is fully immersed in conductor
                            // Therefore, apply corotating electric field E = v X B
                            CorotatingEfieldSpherical(r, theta, phi, chi, cur_time,
                                                      omega_star_data,
                                                      ramp_omega_time_data,
                                                      Bstar_data, Rstar_data, dRstar_data,
                                                      Fr, Ftheta, Fphi);
                        }
                    } else {
                        int ApplyCorotatingEField = 0;
                        // Apply corotating electric field E = v X B, if edge-centered Efield is fully immersed
                        // in conductor
                        if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1) ApplyCorotatingEField = 1;
                        ExternalEFieldSpherical(r, theta, phi, chi, cur_time,
                                                omega_star_data,
                                                ramp_omega_time_data,
                                                Bstar_data, Rstar_data,
                                                corotatingE_maxradius_data,
                                                E_external_monopole_data, ApplyCorotatingEField,
                                                Fr, Ftheta, Fphi);
                    }
                }
                // convert to y component
                ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                    r, theta, phi, mfy_arr(i,j,k));
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, z_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                // Initialize with Bfield in spherical coordinates
                if (init_Bfield == 1) {
                    if (LimitDipoleBInit_data == 1) {
                        if (r < DipoleB_init_maxradius_data) {
                            ExternalBFieldSpherical( r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                        }
                    } else {
                        ExternalBFieldSpherical( r, theta, phi, chi, cur_time,
                                                omega_star_data,
                                                ramp_omega_time_data,
                                                Bstar_data, Rstar_data, dRstar_data,
                                                Fr, Ftheta, Fphi);
                    }
                }
                // Initialize corotating Efield in r < corotating
                if (init_Bfield == 0) {
                    if (InitializeFullExternalEField == 0) {
                        if (conductor(i,j,k)==1 and conductor(i,j,k+1)==1 ) {
                            // Edge-centered Efield is fully immersed in conductor
                            // Therefore, apply corotating electric field E = v X B
                            CorotatingEfieldSpherical(r, theta, phi, chi, cur_time,
                                                      omega_star_data,
                                                      ramp_omega_time_data,
                                                      Bstar_data, Rstar_data, dRstar_data,
                                                      Fr, Ftheta, Fphi);
                        }
                    } else {
                        int ApplyCorotatingEField = 0;
                        // Apply corotating electric field E = v X B, if edge-centered Efield is fully immersed
                        // in conductor
                        if (conductor(i,j,k)==1 and conductor(i,j,k+1)==1) ApplyCorotatingEField = 1;
                        ExternalEFieldSpherical(r, theta, phi, chi, cur_time,
                                                omega_star_data,
                                                ramp_omega_time_data,
                                                Bstar_data, Rstar_data,
                                                corotatingE_maxradius_data,
                                                E_external_monopole_data, ApplyCorotatingEField,
                                                Fr, Ftheta, Fphi);
                    }
                }
                ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                    r, theta, phi, mfz_arr(i,j,k));
            }
        );
    } // mfiter loop
} // InitializeExternalField


void
Pulsar::ApplyCorotatingEfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Efield,
                                const int lev, const amrex::Real a_dt)
{
    auto & warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(lev).CellSizeArray();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    amrex::Real cur_time = warpx.gett_new(lev) + a_dt;
    amrex::IntVect x_nodal_flag = Efield[0]->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = Efield[1]->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = Efield[2]->ixType().toIntVect();
    amrex::GpuArray<int, 3> x_IndexType;
    amrex::GpuArray<int, 3> y_IndexType;
    amrex::GpuArray<int, 3> z_IndexType;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;
    for (int idim = 0; idim < 3; ++idim) {
        x_IndexType[idim] = x_nodal_flag[idim];
        y_IndexType[idim] = y_nodal_flag[idim];
        z_IndexType[idim] = z_nodal_flag[idim];
        center_star_arr[idim] = m_center_star[idim];
    }
    amrex::Real chi = m_Chi;
    amrex::Real omega_star_data = m_omega_star;
    amrex::Real ramp_omega_time_data = m_omega_ramp_time;
    amrex::Real Bstar_data = m_B_star;
    amrex::Real Rstar_data = m_R_star;
    amrex::Real dRstar_data = m_dR_star;
    amrex::Real corotatingE_maxradius_data = m_corotatingE_maxradius;
    int E_external_monopole_data = m_do_E_external_monopole;
    int EnforceTheoreticalEBInGrid_data = m_EnforceTheoreticalEBInGrid;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tex = mfi.tilebox(x_nodal_flag);
        const amrex::Box& tey = mfi.tilebox(y_nodal_flag);
        const amrex::Box& tez = mfi.tilebox(z_nodal_flag);
        amrex::Array4<amrex::Real> const& Ex_arr = Efield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Ey_arr = Efield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Efield[2]->array(mfi);
        amrex::Array4<amrex::Real> const& conductor = m_conductor_fp[lev]->array(mfi);
        // loop over cells and set Efield for r < corotating
        amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, x_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                if (EnforceTheoreticalEBInGrid_data == 0) {
                    if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1) {
//                    if ( r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, chi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        Fr = 0.;
                        ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ex_arr(i,j,k));
//                    }
                    }
                } else {
                    int ApplyCorotatingEField = 0;
                    if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1) ApplyCorotatingEField = 1;
                    ExternalEFieldSpherical(r, theta, phi, chi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data,
                                            corotatingE_maxradius_data,
                                            E_external_monopole_data, ApplyCorotatingEField,
                                            Fr, Ftheta, Fphi);
                    ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Ex_arr(i,j,k));
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, y_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                if (EnforceTheoreticalEBInGrid_data == 0) {
                    if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1) {
//                    if ( r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, chi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        Fr = 0.;
                        ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ey_arr(i,j,k));
                    }
//                    }
                } else {
                    int ApplyCorotatingEField = 0;
                    if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1) ApplyCorotatingEField = 1;
                    ExternalEFieldSpherical(r, theta, phi, chi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data,
                                            corotatingE_maxradius_data,
                                            E_external_monopole_data,
                                            ApplyCorotatingEField,
                                            Fr, Ftheta, Fphi);
                    ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Ey_arr(i,j,k));
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, z_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                if (EnforceTheoreticalEBInGrid_data == 0) {
                    if (conductor(i,j,k)==1 and conductor(i,j,k+1)==1) {
//                    if ( r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, chi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        Fr = 0.;
                        ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ez_arr(i,j,k));
//                    }
                    }
                } else {
                    int ApplyCorotatingEField = 0;
                    if (conductor(i,j,k)==1 and conductor(i,j,k+1)==1) ApplyCorotatingEField = 1;
                    ExternalEFieldSpherical(r, theta, phi, chi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data,
                                            corotatingE_maxradius_data,
                                            E_external_monopole_data,
                                            ApplyCorotatingEField,
                                            Fr, Ftheta, Fphi);
                    ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Ez_arr(i,j,k));
                }
            }
        );
    }


}

void
Pulsar::ApplyDipoleBfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
                                const int lev, const amrex::Real a_dt)
{
    auto & warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(lev).CellSizeArray();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    amrex::Real cur_time = warpx.gett_new(lev) + a_dt;
    amrex::IntVect x_nodal_flag = Bfield[0]->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = Bfield[1]->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = Bfield[2]->ixType().toIntVect();
    amrex::GpuArray<int, 3> x_IndexType;
    amrex::GpuArray<int, 3> y_IndexType;
    amrex::GpuArray<int, 3> z_IndexType;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;
    for (int idim = 0; idim < 3; ++idim) {
        x_IndexType[idim] = x_nodal_flag[idim];
        y_IndexType[idim] = y_nodal_flag[idim];
        z_IndexType[idim] = z_nodal_flag[idim];
        center_star_arr[idim] = m_center_star[idim];
    }
    amrex::Real chi = m_Chi;
    amrex::Real omega_star_data = m_omega_star;
    amrex::Real ramp_omega_time_data = m_omega_ramp_time;
    amrex::Real Bstar_data = m_B_star;
    amrex::Real Rstar_data = m_R_star;
    amrex::Real dRstar_data = m_dR_star;
    int EnforceTheoreticalEBInGrid_data = m_EnforceTheoreticalEBInGrid;
    amrex::Real corotatingE_maxradius_data = m_corotatingE_maxradius;
    amrex::Real enforceDipoleB_maxradius_data = m_enforceDipoleB_maxradius;
    int DampBDipoleInRing_data = m_do_DampBDipoleInRing;
    amrex::Real Bdamping_scale_data = m_Bdamping_scale;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Bfield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tbx = mfi.tilebox(x_nodal_flag);
        const amrex::Box& tby = mfi.tilebox(y_nodal_flag);
        const amrex::Box& tbz = mfi.tilebox(z_nodal_flag);
        amrex::Array4<amrex::Real> const& Bx_arr = Bfield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& By_arr = Bfield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Bz_arr = Bfield[2]->array(mfi);
        amrex::Array4<amrex::Real> const& conductor = m_conductor_fp[lev]->array(mfi);
        // loop over cells and set Efield for r < dipoleB_max_radius
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, x_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                if (EnforceTheoreticalEBInGrid_data == 0) {
                    if ( r <= enforceDipoleB_maxradius_data ) {
                        if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1 and conductor(i,j,k+1)==1 and conductor(i,j+1,k+1)==1) {
                            // Enforce magnetic field boundary condition (dipole field) if face-centered
                            // Bfield location is completely embedded in the conductor
                            ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                            Ftheta = 0.; Fphi = 0.;
                            ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Bx_arr(i,j,k));
                        }
                    }
                    else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
                        // Dipole magnetic field applied for r < corotating radius and then
                        // a smoothening function tanh damping is applied between radius of star
                        // to radius where corotating electric field is enforced.
                        if (DampBDipoleInRing_data == 1) {
                            // from inner ring to outer ring
                            ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                            amrex::Real Bx_dipole;
                            ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Bx_dipole);
                            // Damping Function : Fd = tanh(dampingscale * (1-r/Rinner))
                            // where Rinner = enforceDipoleB_maxradius
                            //                is the range where Bdipole is imposed
                            // Fd(Rinner) ~ 1
                            // Fd(R_domainboundary) ~ 0
                            amrex::Real Fd = 1._rt
                                           + std::tanh( Bdamping_scale_data
                                                      * (1._rt - r/enforceDipoleB_maxradius_data)
                                                      );
                            Bx_arr(i,j,k) += Fd * Bx_dipole;
                        }
                    }
                } else {
                    // Theoretical Bfield enforced everywhere on the grid
                    ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data, dRstar_data,
                                            Fr, Ftheta, Fphi);
                    ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Bx_arr(i,j,k));
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, y_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                if (EnforceTheoreticalEBInGrid_data == 0) {
                    if ( r <= enforceDipoleB_maxradius_data ) {
                        if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1 and conductor(i,j,k+1)==1 and conductor(i+1,j,k+1)==1) {
                            // Enforce magnetic field boundary condition (dipole field) if face-centered
                            // Bfield location is completely embedded in the conductor
                            ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                            Ftheta = 0.; Fphi = 0.;
                            ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, By_arr(i,j,k));
                        }
                    } else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
                        // Dipole magnetic field applied for r < corotating radius and then
                        // a smoothening function tanh damping is applied between radius of star
                        // to radius where corotating electric field is enforced.
                        if (DampBDipoleInRing_data == 1) {
                            // from inner ring to outer ring
                            ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                            amrex::Real By_dipole;
                            ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, By_dipole);
                            amrex::Real Fd = 1._rt
                                           + std::tanh( Bdamping_scale_data
                                                      * (1._rt - r/enforceDipoleB_maxradius_data)
                                                      );
                            By_arr(i,j,k) += Fd * By_dipole;
                        }
                    }
                } else {
                    // Theoretical Bfield enforced everywhere on the grid
                    ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data, dRstar_data,
                                            Fr, Ftheta, Fphi);
                    ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, By_arr(i,j,k));
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Fr, Ftheta, Fphi;
                Fr = 0.; Ftheta = 0.; Fphi = 0.;
                // compute cell coordinates
                ComputeCellCoordinates(i, j, k, z_IndexType, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                 r, theta, phi);
                if (EnforceTheoreticalEBInGrid_data == 0) {
                    if ( r <= enforceDipoleB_maxradius_data ) {
                        if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1 and conductor(i,j+1,k)==1 and conductor(i+1,j+1,k)==1) {
                            // Enforce magnetic field boundary condition (dipole field) if face-centered
                            // Bfield location is completely embedded in the conductor
                            ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                            Ftheta = 0.; Fphi = 0.;
                            ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Bz_arr(i,j,k));
                        }
                    } else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
                        // Dipole magnetic field applied for r < corotating radius and then
                        // a smoothening function tanh damping is applied between radius of star
                        // to radius where corotating electric field is enforced.
                        if (DampBDipoleInRing_data == 1) {
                            // from inner ring to outer ring
                            ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                                    omega_star_data,
                                                    ramp_omega_time_data,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                            amrex::Real Bz_dipole;
                            ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Bz_dipole);
                            amrex::Real Fd = 1._rt
                                           + std::tanh( Bdamping_scale_data
                                                      * (1._rt - r/enforceDipoleB_maxradius_data)
                                                      );
                            Bz_arr(i,j,k) += Fd * Bz_dipole;
                        }
                    }
                } else {
                    // Theoretical Bfield enforced everywhere on the grid
                    ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data, dRstar_data,
                                            Fr, Ftheta, Fphi);
                    ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Bz_arr(i,j,k));
                }
            }
        );
    }

}

void
Pulsar::SetTangentialEforInternalConductor( std::array <std::unique_ptr<amrex::MultiFab>, 3> &Efield,
                                const int lev, const amrex::Real a_dt)
{
    amrex::ignore_unused(a_dt);
    if (m_do_conductor == false) return;
    amrex::IntVect x_nodal_flag = Efield[0]->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = Efield[1]->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = Efield[2]->ixType().toIntVect();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tex = mfi.tilebox(x_nodal_flag);
        const amrex::Box& tey = mfi.tilebox(y_nodal_flag);
        const amrex::Box& tez = mfi.tilebox(z_nodal_flag);
        amrex::Array4<amrex::Real> const& Ex_arr = Efield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Ey_arr = Efield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Efield[2]->array(mfi);
        amrex::Array4<amrex::Real> const& conductor = m_conductor_fp[lev]->array(mfi);
        // loop over cells and set Efield for r < corotating
        amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1) {
                    //Ex is tangential and on the conductor. Setting value to 0
                    Ex_arr(i,j,k) = 0.;
                }
        },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1) {
                    //Ex is tangential and on the conductor. Setting value to 0
                    Ey_arr(i,j,k) = 0.;
                }
        },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if (conductor(i,j,k)==1 and conductor(i,j,k+1)==1) {
                    //Ex is tangential and on the conductor. Setting value to 0
                    Ez_arr(i,j,k) = 0.;
                }
        });
    }
}

void
Pulsar::ComputePlasmaNumberDensity ()
{
    // compute plasma number density using particle to mesh
    auto &warpx = WarpX::GetInstance();
    const int nlevs_max = warpx.finestLevel() + 1;
    std::vector<std::string> species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    for (int lev = 0; lev < nlevs_max; ++lev) {
        m_plasma_number_density[lev]->setVal(0._rt);
        for (int isp = 0; isp < nspecies; ++isp) {
            // creating single-component tmp multifab
            auto& pc = warpx.GetPartContainer().GetParticleContainer(isp);
            // Add the weight for each particle -- total number of particles of this species
            ParticleToMesh(pc, *m_plasma_number_density[lev], lev,
                [=] AMREX_GPU_DEVICE (const WarpXParticleContainer::SuperParticleType& p,
                    amrex::Array4<amrex::Real> const& out_array,
                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
                {
                    // Get position in WarpX convention to use in parser. Will be different from
                    // p.pos() for 1D and 2D simulations.
                    amrex::ParticleReal xw = 0._rt, yw = 0._rt, zw = 0._rt;
                    get_particle_position(p, xw, yw, zw);

                    // Get position in AMReX convention to calculate corresponding index.
                    int ii = 0, jj = 0, kk = 0;
                    amrex::ParticleReal x = p.pos(0);
                    amrex::Real lx = (x - plo[0]) * dxi[0];
                    ii = static_cast<int>(amrex::Math::floor(lx));
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
                    amrex::ParticleReal y = p.pos(1);
                    amrex::Real ly = (y - plo[1]) * dxi[1];
                    jj = static_cast<int>(amrex::Math::floor(ly));
#endif
#if defined(WARPX_DIM_3D)
                    amrex::ParticleReal z = p.pos(2);
                    amrex::Real lz = (z - plo[2]) * dxi[2];
                    kk = static_cast<int>(amrex::Math::floor(lz));
#endif
                    amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, isp), p.rdata(PIdx::w));
                }
                , false // setting zero_out_input to false and
                        // instead managing setVal(0.) before ParticleToMesh
            );
        }

        const amrex::Geometry& geom = warpx.Geom(lev);
        const auto dx = geom.CellSizeArray();
#if defined WARPX_DIM_3D
        amrex::Real inv_vol = 1._rt/(dx[0] * dx[1] * dx[2]);
#elif defined(WARPX_DIM_XZ)
        amrex::Real inv_vol = 1._rt/(dx[0] * dx[1] );
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(*m_plasma_number_density[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            amrex::Array4<amrex::Real> const& density = m_plasma_number_density[0]->array(mfi);
            amrex::Box const& tbx = mfi.tilebox();
            const int ncomps = m_plasma_number_density[lev]->nComp();
            amrex::ParallelFor(tbx, ncomps,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                    density(i,j,k,n) *= inv_vol;
            });
        }
    } // loop over levels
}


void
Pulsar::ComputePlasmaMagnetization ()
{
    auto& warpx = WarpX::GetInstance();
    const int nlevs_max = warpx.finestLevel() + 1;
    std::vector<std::string> species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    const amrex::Real min_ndens = m_lbound_ndens_magnetization;
    constexpr amrex::Real mu0_m_c2_inv = 1._rt / (PhysConst::mu0 * PhysConst::m_e
                                                 * PhysConst::c * PhysConst::c);
    for (int lev = 0; lev < nlevs_max; ++lev) {
        m_magnetization[lev]->setVal(0._rt);
        const amrex::MultiFab& Bx_mf = warpx.getBfield(lev, 0);
        const amrex::MultiFab& By_mf = warpx.getBfield(lev, 1);
        const amrex::MultiFab& Bz_mf = warpx.getBfield(lev, 2);
        for (amrex::MFIter mfi(*m_magnetization[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const& mag = m_magnetization[lev]->array(mfi);
            amrex::Array4<amrex::Real> const& ndens = m_plasma_number_density[lev]->array(mfi);
            amrex::Array4<const amrex::Real> const& Bx = Bx_mf[mfi].array();
            amrex::Array4<const amrex::Real> const& By = By_mf[mfi].array();
            amrex::Array4<const amrex::Real> const& Bz = Bz_mf[mfi].array();

            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real Bx_cc = (Bx(i,j,k) + Bx(i+1, j, k)) / 2.0;
                    amrex::Real By_cc = (By(i,j,k) + By(i, j+1, k)) / 2.0;
                    amrex::Real Bz_cc = (Bz(i,j,k) + Bz(i, j, k+1)) / 2.0;
                    // magnitude of magnetic field
                    amrex::Real B_mag = std::sqrt( Bx_cc * Bx_cc
                                                 + By_cc * By_cc
                                                 + Bz_cc * Bz_cc );
                    // total number density
                    amrex::Real total_ndens = 0._rt;
                    for (int isp = 0; isp < nspecies; ++isp) {
                        total_ndens += ndens(i, j, k, isp);
                    }
                    // ensure that there is no 0 in the denominator when number density is 0
                    // due to absence of particles
                    if (total_ndens > 0.) {
                        mag(i, j, k) = (B_mag * B_mag) * mu0_m_c2_inv /total_ndens;
                    } else {
                        // using minimum number density if there are no particles in the cell
                        mag(i, j, k) = (B_mag * B_mag) * mu0_m_c2_inv /min_ndens;
                    }
                }
            );
        } // mfiter loop
    } // loop over levels
}

void
Pulsar::TuneSigma0Threshold (const int step)
{
    auto& warpx = WarpX::GetInstance();
    std::vector<std::string> species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    amrex::Real total_weight_allspecies = 0._rt;
    amrex::Real dt = warpx.getdt(0);
    // Total number of cells that have injected particles
//    amrex::Real total_injected_cells = SumInjectionFlag();
//    // for debugging print injected cell data
//    if (m_print_injected_celldata == 1) {
//        if (warpx.getistep(0) >= m_print_celldata_starttime) {
//            PrintInjectedCellValues();
//        }
//    }

    using PTDType = typename WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;
    for (int isp = 0; isp < nspecies; ++isp) {
        amrex::Real ws_total = 0._rt;
        auto& pc = warpx.GetPartContainer().GetParticleContainer(isp);
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
        amrex::Real cur_time = warpx.gett_new(0);
        auto ws_r = amrex::ParticleReduce<
                        amrex::ReduceData < amrex::ParticleReal> >
                    ( pc,
                        [=] AMREX_GPU_DEVICE (const PTDType &ptd, const int i) noexcept
                        {
                            auto p = ptd.getSuperParticle(i);
                            amrex::ParticleReal wp = p.rdata(PIdx::w);
                            amrex::Real filter = 0._rt;
                            amrex::ParticleReal injectiontime = ptd.m_runtime_rdata[0][i];
                            if ( injectiontime < cur_time + 0.1_rt*dt and
                                 injectiontime > cur_time - 0.1_rt*dt ) {
                                filter = 1._rt;
                            }
                            return (wp * filter);
                        },
                        reduce_ops
                    );
        ws_total = amrex::get<0>(ws_r);
        amrex::ParallelDescriptor::ReduceRealSum(ws_total);
        total_weight_allspecies += ws_total;
    }
    // injection rate is sum of particle weight over all species per timestep
    amrex::Real current_injection_rate = total_weight_allspecies / dt;
    if (TotalParticlesIsSumOfSpecies == 0) {
        current_injection_rate = total_weight_allspecies / 2._rt / dt;
    }
//    if (list_size < ROI_avg_window_size) {
//        ROI_list.push_back(current_injection_rate);
//        m_sum_injection_rate += current_injection_rate;
//        list_size++;
//    } else {
//        m_sum_injection_rate -= ROI_list.front();
//        ROI_list.pop_front();
//        ROI_list.push_back(current_injection_rate);
//        m_sum_injection_rate += ROI_list.back();
//    }
//
//    if (sigma_list_size < sigma_ref_avg_window_size) {
//        sigma_list.push_back(m_Sigma0_threshold);
//        m_sigma_threshold_sum += sigma_list.back();
//        sigma_list_size++;
//    } else {
//        m_sigma_threshold_sum -= sigma_list.front();
//        sigma_list.pop_front();
//        sigma_list.push_back(m_Sigma0_threshold);
//        m_sigma_threshold_sum += sigma_list.back();
//    }
//
//    amrex::Real total_injection_cells = SumInjectionFlag();
//    if (InjCell_list_size < m_InjCell_avg_window_size) {
//        InjCell_list.push_back(total_injection_cells);
//        m_InjCell_sum += total_injection_cells;
//        InjCell_list_size++;
//    } else {
//        m_InjCell_sum -= InjCell_list.front();
//        InjCell_list.pop_front();
//        InjCell_list.push_back(total_injection_cells);
//        m_InjCell_sum += total_injection_cells;
//    }
//    amrex::Real avg_InjCells = 0;
//    if (m_use_LstSqFit_TC == 1) {
//        avg_InjCells = LeastSquareFitforTC();
//    } else {
//        avg_InjCells = m_InjCell_sum/(InjCell_list_size * 1._rt);
//    }
//    amrex::Print() << " curent TC= " << total_injection_cells << " list size : " << InjCell_list_size << " sum : " << m_InjCell_sum << " avg_inj cells : " << avg_InjCells << "\n";
    //particles to be injected
    auto& pc = warpx.GetPartContainer().GetParticleContainer(0);
    auto& phys_pc = dynamic_cast<PhysicalParticleContainer&>(pc);
    amrex::Real num_ppc = phys_pc.getPlasmaInjector()->num_particles_per_cell;
    amrex::Print() << " num ppc : " << num_ppc << "\n";
    const int lev = 0;
    const auto dx_lev = warpx.Geom(lev).CellSizeArray();
    amrex::Real scale_factor = dx_lev[0] * dx_lev[1] * dx_lev[2] / num_ppc * m_particle_weight_scaling;
    int ParticlesToBeInjected = TotalParticlesToBeInjected(scale_factor);
    amrex::Print() << " TP " << ParticlesToBeInjected << "\n";
//    amrex::Print() << " TP " << ParticlesToBeInjected << " current TC : " << total_injection_cells<< "\n";

//    amrex::Print() << " current_injection rate " << current_injection_rate << " sum : " << m_sum_injection_rate << "\n";
//    amrex::Real avg_sigma_threshold = m_sigma_threshold_sum/(sigma_list_size * 1._rt);
//    if (use_single_sigma_ref == 1) {
//        avg_sigma_threshold = sigma_ref;
//    }
//    amrex::Print() << " sigma list size : " << sigma_list_size << " sigma sum " << m_sigma_threshold_sum << " current sigma " << m_Sigma0_threshold << " avg : " << avg_sigma_threshold << "\n";
    amrex::Real specified_injection_rate = m_GJ_injection_rate * m_injection_rate;
//    amrex::Real max_sigma = MaxMagnetization();
//    if (m_injection_tuning_interval.contains(step+1) and m_use_FixedSigmaInput==0 ) {
//        // Sigma0 before modification
//        amrex::Real m_Sigma0_pre = m_Sigma0_threshold;
//        // running average of injection rate over average window
//        amrex::Real avg_injection_rate = m_sum_injection_rate/(list_size * 1._rt);
//        amrex::Real new_sigma0_threshold = m_Sigma0_threshold;
//        if (sigma_tune_method_TCTP == 0) {
//        // sigma is tuned by comparing the difference between desired and computed ROI
//            if (avg_injection_rate < specified_injection_rate) {
//                // reduce sigma0 so more particles can be injected
//                if (m_sigma_tune_method == "10percent") {
//                    // Modify threshold magnetization threshold by 10%
//                    new_sigma0_threshold = m_Sigma0_threshold - 0.1 * avg_sigma_threshold;
//                }
//                if (m_sigma_tune_method == "relative_difference") {
//                    if (avg_injection_rate == 0.) {
//                        // If no particles are injected only change the threshold sigma by 10%
//                        new_sigma0_threshold = m_Sigma0_threshold - 0.1 * avg_sigma_threshold;
//                    } else {
//                        amrex::Real rel_diff = (specified_injection_rate - avg_injection_rate)/specified_injection_rate;
//                        if (rel_diff < m_ubound_reldiff_sigma0) {
//                            // If relative difference is less than user-defined upper bound, reduce magnetization by rel_diff
//                            new_sigma0_threshold = m_Sigma0_threshold - rel_diff * avg_sigma_threshold;
//                        } else {
//                            // Maximum relative difference is set by user-defined upper bound
//                            new_sigma0_threshold = m_Sigma0_threshold - m_ubound_reldiff_sigma0 * avg_sigma_threshold;
//                        }
//                    }
//                }
//            } else if (avg_injection_rate > specified_injection_rate ) {
//                // Increase sigma0 so fewer particles are injected
//                if (m_sigma_tune_method == "10percent") {
//                    // Modify threshold magnetization threshold by 10%
//                    new_sigma0_threshold = m_Sigma0_threshold + 0.1 * avg_sigma_threshold;
//                }
//                if (m_sigma_tune_method == "relative_difference") {
//                    amrex::Real rel_diff = (avg_injection_rate - specified_injection_rate)/specified_injection_rate;
//                    if (rel_diff < m_ubound_reldiff_sigma0) {
//                        // If relative difference is less than user-defined upper bound, increase magnetization by rel_diff
//                        new_sigma0_threshold = m_Sigma0_threshold + rel_diff * avg_sigma_threshold;
//                    } else {
//                        // If rel_diff > upper bound, then increase sigma0 by m_ubound_reldiff_sigma0
//                        new_sigma0_threshold = m_Sigma0_threshold + m_ubound_reldiff_sigma0 * avg_sigma_threshold;
//                    }
//                }
//            }
//        } else {
//            amrex::Print() << " sigma tuning using TCTP \n";
//        // sigma is tuned by comparing TC = TP
//            if (avg_InjCells < ParticlesToBeInjected) {
//            // decrease sigma by relative difference
//                if (m_sigma_tune_method == "relative_difference") {
//                    amrex::Real rel_diff =  (ParticlesToBeInjected - avg_InjCells)
//                                          / (ParticlesToBeInjected * 1._rt);
//                    if (rel_diff < m_ubound_reldiff_sigma0) {
//                    // If relative difference is less than user-defined upper bound, reduce magnetization by rel diff
//                        new_sigma0_threshold = m_Sigma0_threshold - rel_diff * m_Sigma0_threshold;
//                    } else {
//                    // If relative difference is >= ubound, reduce magnetization by ubound
//                        new_sigma0_threshold = m_Sigma0_threshold - m_ubound_reldiff_sigma0 * m_Sigma0_threshold;
//                    } // decrease by rel_diff or ubound
//                } // ref diff method
//            } else if (avg_InjCells > ParticlesToBeInjected) {
//            // increase sigma by relative difference
//                if (m_sigma_tune_method == "relative_difference") {
//                    amrex::Real rel_diff = (avg_InjCells - ParticlesToBeInjected)
//                                         / (ParticlesToBeInjected * 1._rt);
//                    if (rel_diff < m_ubound_reldiff_sigma0) {
//                    // if relative difference is less than user-defined upper bound, increase magnetization by rel diff
//                        new_sigma0_threshold = m_Sigma0_threshold + rel_diff * m_Sigma0_threshold;
//                    } else {
//                    // if rel diff is >= ubound, increase magnetization by ubound
//                        new_sigma0_threshold = m_Sigma0_threshold + m_ubound_reldiff_sigma0 * m_Sigma0_threshold;
//                    } // increase by rel diff or ubound
//                } // ref diff method
//            }
//        }
//        if (new_sigma0_threshold < m_min_Sigma0) new_sigma0_threshold = m_min_Sigma0;
//        if (new_sigma0_threshold > m_max_Sigma0) new_sigma0_threshold = m_max_Sigma0;
//        // Store modified new sigma0 in member variable, m_Sigma0_threshold
//        amrex::Print() << " old sigma " << m_Sigma0_threshold << " new : " << new_sigma0_threshold << "\n";
////        if (total_injection_cells <= 0.1 * ParticlesToBeInjected) {i
////            new_sigma0_threshold = 0.99*max_sigma*(14000/12000)*(14000/12000)*(14000/12000);
////            amrex::Print() << " injec cell is " << total_injection_cells << " <= 0.1*TP " << ParticlesToBeInjected << " sigma0_new modified to " << new_sigma0_threshold << " using max sigma : " << max_sigma<< "\n";
////        }
//        if (m_use_Sigma0_avg == 1) {
//            amrex::Real sigma_avg = (new_sigma0_threshold + m_Sigma0_threshold)/2._rt;
//            m_Sigma0_threshold = sigma_avg;
//        } else {
//            m_Sigma0_threshold = new_sigma0_threshold;
//        }
//        amrex::AllPrintToFile("RateOfInjection") << warpx.getistep(0) << " " << warpx.gett_new(0) << " " << dt <<  " " << specified_injection_rate << " " << avg_injection_rate << " " << m_Sigma0_pre << " "<< m_Sigma0_threshold << " " << m_min_Sigma0 << " " << m_max_Sigma0 << " " << m_Sigma0_baseline << " " << total_injection_cells << " " << avg_InjCells << " " << ParticlesToBeInjected<< "\n";
//    }
    //if (total_injection_cells <= m_min_TCTP_ratio * ParticlesToBeInjected) {
    //    if (m_use_maxsigma_for_Sigma0 == 1 and m_use_FixedSigmaInput == 0) {
    //        amrex::Real r_rstar_fac = m_injRing_radius/m_R_star;
    //        amrex::Real new_sigma0_threshold = m_maxsigma_fraction * max_sigma * r_rstar_fac * r_rstar_fac * r_rstar_fac;
    //        amrex::Print() << " injec cell is " << total_injection_cells << " <= " << m_min_TCTP_ratio <<" *TP " << ParticlesToBeInjected << " sigma0_new modified to " << new_sigma0_threshold << " using max sigma : " << max_sigma<< "\n";
    //        m_Sigma0_threshold = new_sigma0_threshold;
    //    }
    //}
//    amrex::Real max_sigma_threshold = MaxThresholdSigma();
//    amrex::Real theta = 0.0;
//    amrex::Real c_theta = std::cos(theta);
//    amrex::Real s_theta = std::sin(theta);
//    amrex::Real c_chi = std::cos(m_Chi);
//    amrex::Real s_chi = std::sin(m_Chi);
//    amrex::Real omega = Omega(m_omega_star, warpx.gett_new(0), m_omega_ramp_time);
//    amrex::Real phi = 0.;
//    amrex::Real omega_t_integral;
//    if (warpx.gett_new(0) < m_omega_ramp_time) {
//        // omega returned from function above is Omega_star * t / tramp
//        omega_t_integral = omega * warpx.gett_new(0) / 2.;
//    } else {
//        omega_t_integral = omega*(warpx.gett_new(0) - m_omega_ramp_time) + omega*m_omega_ramp_time/2.0;
//    }
//    amrex::Real psi = phi - omega_t_integral;
//    amrex::Real c_psi = std::cos(psi);
//    amrex::Real s_psi = std::sin(psi);
//    amrex::Real Brp = 2. * m_B_star * (c_chi * c_theta + s_chi * s_theta * c_psi);
//    amrex::Real Btp = m_B_star * (c_chi * s_theta - s_chi * c_theta * c_psi);
//    amrex::Real Bphip = m_B_star * s_chi * s_psi;
    amrex::AllPrintToFile("ROI") << warpx.getistep(0) << " " << warpx.gett_new(0) << " " << dt <<  " " << specified_injection_rate << " " << current_injection_rate  <<"\n";
}

amrex::Real
Pulsar::LeastSquareFitforTC ()
{
    auto& warpx = WarpX::GetInstance();
    amrex::Real dt = warpx.getdt(0);
    amrex::Real cur_time = warpx.gett_new(0);
    int sample_size = InjCell_list.size();
    if (InjCell_list_size < m_InjCell_avg_window_size) {
        return *std::next(InjCell_list.begin(), InjCell_list_size-1);
    } else {
        amrex::Real p11 = 0.0;
        amrex::Real p12 = 0.0;
        amrex::Real p21 = 0.0;
        amrex::Real p22 = sample_size;
        amrex::Real q11 = 0.0;
        amrex::Real q21 = 0.0;
        for (int i = 0; i < sample_size; ++i) {
            amrex::Real ti = cur_time - i * dt;
            p11 += ti * ti;
            p12 += ti;
            p21 = p12;
            amrex::Real TC_at_ti = *(std::next(InjCell_list.begin(), sample_size - 1 - i) );
            q11 += ti * TC_at_ti;
            q21 += TC_at_ti;
        }
        amrex::Real b = ( q11 - (p11/p21) * q21 ) / ( p12 - (p11/p21) * p22 );
        amrex::Real a = ( q21 - p22*b ) / p21;
        return (a * cur_time + b);
    }
    return 0.;
}


int
Pulsar::TotalParticlesToBeInjected (amrex::Real scale_factor)
{
    auto& warpx = WarpX::GetInstance();
    amrex::Real dt = warpx.getdt(0);
    amrex::Real specified_injection_rate = m_GJ_injection_rate * m_injection_rate;
    amrex::Real part_weight = m_max_ndens * scale_factor;
    m_particle_wt = part_weight;

    if (TotalParticlesIsSumOfSpecies == 1) {
        return (specified_injection_rate * dt / part_weight) / 2._rt;
    } else {
        return (specified_injection_rate * dt / part_weight);
    }
}

void
Pulsar::TotalParticles ()
{
    auto& warpx = WarpX::GetInstance();
    std::vector<std::string> species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    amrex::Long total_particles = 0;

    for (int isp = 0; isp < nspecies; ++isp) {
        auto& pc = warpx.GetPartContainer().GetParticleContainer(isp);
        amrex::Long np_total = pc.TotalNumberOfParticles();
        amrex::ParallelDescriptor::ReduceLongSum(np_total);
        total_particles += np_total;
    }
}

amrex::Real
Pulsar::SumPolarCapFlag ()
{
    return m_PC_flag[0]->sum();
}

amrex::Real
Pulsar::SumInjectionFlag ()
{
     return m_injection_flag[0]->sum();
}

amrex::Real
Pulsar::SumInjectedCells ()
{
    return m_injected_cell[0]->sum();
}

amrex::Real
Pulsar::SumSigmaRelDiff ()
{
    return m_sigma_reldiff[0]->sum();
}

amrex::Real
Pulsar::MaxMagnetization ()
{
    return m_sigma_inj_ring[0]->max(0);
}

amrex::Real
Pulsar::MaxThresholdSigma()
{
    return m_sigma_threshold[0]->max(0);
}

amrex::Real
Pulsar::PcountSum ()
{
    return m_pcount[0]->sum();
}

amrex::Real
Pulsar::SumMagnetizationinInjectionRing ()
{
    return m_sigma_inj_ring[0]->sum();
}

void
Pulsar::PrintInjectedCellValues ()
{
    auto& warpx = WarpX::GetInstance();
    std::vector<std::string> species_names = warpx.GetPartContainer().GetSpeciesNames();
    int total_injected_cells = static_cast<int>(SumInjectionFlag());
    // x, y, z, r, theta, phi, injection_flag, magnetization, ndens_p, ndens_e, Bx, By, Bz, Bmag, rho
    int total_diags = 15;
    amrex::Gpu::DeviceVector<amrex::Real> InjectedCellDiag(total_injected_cells*total_diags,0.0);
    amrex::Real * InjectedCellDiagData = InjectedCellDiag.data();
    const int lev = 0;
    const auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox real_box = warpx.Geom(lev).ProbDomain();
    const amrex::MultiFab& Bx_mf = warpx.getBfield(lev,0);
    const amrex::MultiFab& By_mf = warpx.getBfield(lev,1);
    const amrex::MultiFab& Bz_mf = warpx.getBfield(lev,2);
    const amrex::MultiFab& injectionflag_mf = *m_injection_flag[lev];
    const amrex::MultiFab& magnetization_mf = *m_magnetization[lev];
    const amrex::MultiFab& ndens_mf = *m_plasma_number_density[lev];
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;
    for (int idim = 0; idim < 3; ++idim) {
        center_star_arr[idim] = m_center_star[idim];
    }
    std::unique_ptr<amrex::MultiFab> rho;
    auto& mypc = warpx.GetPartContainer();
    rho = mypc.GetChargeDensity(lev, true);
    amrex::MultiFab & rho_mf = *rho;

    amrex::Gpu::DeviceScalar<int> cell_counter(0);
    int* cell_counter_d = cell_counter.dataPtr();
    for (amrex::MFIter mfi(injectionflag_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.tilebox();
        amrex::Array4<const amrex::Real> const& Bx = Bx_mf[mfi].array();
        amrex::Array4<const amrex::Real> const& By = By_mf[mfi].array();
        amrex::Array4<const amrex::Real> const& Bz = Bz_mf[mfi].array();
        amrex::Array4<const amrex::Real> const& ndens = ndens_mf[mfi].array();
        amrex::Array4<const amrex::Real> const& injection = injectionflag_mf[mfi].array();
        amrex::Array4<const amrex::Real> const& mag = magnetization_mf[mfi].array();
        amrex::Array4<const amrex::Real> const& rho_arr = rho_mf[mfi].array();
        amrex::LoopOnCpu( bx,
            [=] (int i, int j, int k)
            {
                if (injection(i,j,k) == 1) {
                    // Cartesian Coordinates
                    amrex::Real x = i * dx_lev[0] + real_box.lo(0) + 0.5_rt * dx_lev[0];
                    amrex::Real y = j * dx_lev[1] + real_box.lo(1) + 0.5_rt * dx_lev[1];
                    amrex::Real z = k * dx_lev[2] + real_box.lo(2) + 0.5_rt * dx_lev[2];
                    // convert cartesian to spherical coordinates
                    amrex::Real r, theta, phi;
                    ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                     r, theta, phi);
                    amrex::Real Bx_cc = (Bx(i,j,k) + Bx(i+1, j, k)) / 2.0;
                    amrex::Real By_cc = (By(i,j,k) + By(i, j+1, k)) / 2.0;
                    amrex::Real Bz_cc = (Bz(i,j,k) + Bz(i, j, k+1)) / 2.0;
                    // magnitude of magnetic field
                    amrex::Real B_mag = std::sqrt( Bx_cc * Bx_cc
                                                 + By_cc * By_cc
                                                 + Bz_cc * Bz_cc );
                    int counter = *cell_counter_d;
                    InjectedCellDiagData[counter*total_diags+0] = x;
                    InjectedCellDiagData[counter*total_diags+1] = y;
                    InjectedCellDiagData[counter*total_diags+2] = z;
                    InjectedCellDiagData[counter*total_diags+3] = r;
                    InjectedCellDiagData[counter*total_diags+4] = theta;
                    InjectedCellDiagData[counter*total_diags+5] = phi;
                    InjectedCellDiagData[counter*total_diags+6] = injection(i, j, k);
                    InjectedCellDiagData[counter*total_diags+7] = mag(i, j, k);
                    InjectedCellDiagData[counter*total_diags+8] = ndens(i, j, k, 0);
                    InjectedCellDiagData[counter*total_diags+9] = ndens(i, j, k, 1);
                    InjectedCellDiagData[counter*total_diags+10] = Bx_cc;
                    InjectedCellDiagData[counter*total_diags+11] = By_cc;
                    InjectedCellDiagData[counter*total_diags+12] = Bz_cc;
                    InjectedCellDiagData[counter*total_diags+13] = B_mag;
                    InjectedCellDiagData[counter*total_diags+14] = rho_arr(i, j, k);
                    const int unity = 1;
                    amrex::HostDevice::Atomic::Add(cell_counter_d, unity);
                }
            });
    }
    amrex::Print() << " counter : " << cell_counter.dataValue() << " total cells injected " << total_injected_cells << "\n";
    std::stringstream ss;
    ss << amrex::Concatenate("InjectionCellData", warpx.getistep(0), 5);
    amrex::AllPrintToFile(ss.str()) << " cell_index x y z r theta phi injection magnetization ndens_p ndens_e Bx By Bz Bmag rho \n" ;
    for (int icell = 0; icell < total_injected_cells; ++icell ) {
        if (InjectedCellDiagData[icell*total_diags + 6] == 1) {
            amrex::AllPrintToFile(ss.str()) << icell << " ";
            for (int idata = 0; idata < total_diags; ++idata) {
                amrex::AllPrintToFile(ss.str()) << InjectedCellDiagData[icell * total_diags + idata] << " ";
            }
        }
        amrex::AllPrintToFile(ss.str()) << "\n";
    }
}

void
Pulsar::TotalParticlesInjected ()
{
    auto& warpx = WarpX::GetInstance();
    std::vector<std::string> species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    amrex::Real total_weight_allspecies = 0._rt;
    amrex::Real dt = warpx.getdt(0);
    // Total number of cells that have injected particles
//    amrex::Real total_injected_cells = SumInjectionFlag();
    // for debugging print injected cell data
    if (m_print_injected_celldata == 1) {
        if (warpx.getistep(0) >= m_print_celldata_starttime) {
            PrintInjectedCellValues();
        }
    }

    using PTDType = typename WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;

    for (int isp = 0; isp < nspecies; ++isp) {
        amrex::Real ws_total = 0._rt;
        auto& pc = warpx.GetPartContainer().GetParticleContainer(isp);
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
        amrex::Real cur_time = warpx.gett_new(0);
        auto ws_r = amrex::ParticleReduce<
                        amrex::ReduceData < amrex::ParticleReal> >
                    ( pc,
                        [=] AMREX_GPU_DEVICE (const PTDType &ptd, const int i) noexcept
                        {
                            auto p = ptd.getSuperParticle(i);
                            amrex::ParticleReal wp = p.rdata(PIdx::w);
                            amrex::Real filter = 0._rt;
                            amrex::ParticleReal injectiontime = ptd.m_runtime_rdata[0][i];
                            if ( injectiontime < cur_time + 0.1_rt*dt and
                                 injectiontime > cur_time - 0.1_rt*dt ) {
                                filter = 1._rt;
                            }
                            return (filter);
                        },
                        reduce_ops
                    );
        ws_total = amrex::get<0>(ws_r);
        amrex::ParallelDescriptor::ReduceRealSum(ws_total);
        amrex::Print() << " sp : " << isp << " total particles injected call: " << ws_total << "\n";
    }
}

void
Pulsar::FlagCellsForInjectionWithPcounts ()
{
    //Flag cells within desired injection region that have sigma > sigma0
    WarpX& warpx = WarpX::GetInstance();
    const int lev = 0;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = m_injection_flag[lev]->ixType().toIntVect();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xc;
    for (int idim = 0; idim < 3; ++idim) {
        xc[idim] = m_center_star[idim];
    }
    amrex::Real pulsar_cell_inject_rmin = m_cell_inject_rmin;
    amrex::Real pulsar_cell_inject_rmax = m_cell_inject_rmax;
    amrex::Real Rstar = m_R_star;
    amrex::Real Sigma0_threshold = m_Sigma0_threshold;
    amrex::Print() << "sigma0 threshold used to check sigma_loc " << m_Sigma0_threshold << "\n";
    int modify_Sigma0_threshold = modify_sigma_threshold;
    int onlyPCinjection = m_onlyPCinjection;

    //particles to be injected
    auto& pc = warpx.GetPartContainer().GetParticleContainer(0);
    auto& phys_pc = dynamic_cast<PhysicalParticleContainer&>(pc);
    amrex::Real num_ppc = phys_pc.getPlasmaInjector()->num_particles_per_cell;
    amrex::Print() << " num ppc : " << num_ppc << "\n";
    amrex::Real scale_factor = dx_lev[0] * dx_lev[1] * dx_lev[2] / num_ppc * m_particle_weight_scaling;
    m_particle_scale_fac = scale_factor;
    int ParticlesToBeInjected = TotalParticlesToBeInjected(scale_factor);
    amrex::Print() << " particles to be injected " << ParticlesToBeInjected <<"\n";

    m_injection_flag[lev]->setVal(0);
    m_injected_cell[lev]->setVal(0);
    m_sigma_inj_ring[lev]->setVal(0);
    m_injection_ringCC[lev]->setVal(0._rt);
    m_pcount[lev]->setVal(0);
    int use_FixedSigmaInput = m_use_FixedSigmaInput;
    for (amrex::MFIter mfi(*m_injection_flag[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        //InitializeGhost Cells also
        const amrex::Box& tb = mfi.tilebox(iv);
        amrex::Array4<amrex::Real> const& injection_flag = m_injection_flag[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& injected_cell = m_injected_cell[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& sigma = m_magnetization[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& inj_ring = m_injection_ring[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& inj_ringCC = m_injection_ringCC[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& sigma_inj_ring = m_sigma_inj_ring[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& sigma_threshold_loc = m_sigma_threshold[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& PCflag = m_PC_flag[lev]->array(mfi);
        amrex::ParallelFor(tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                sigma_threshold_loc(i,j,k) = 0.;
                sigma_inj_ring(i,j,k) = 0.;
                if (onlyPCinjection == 1) {
                    if ( PCflag(i,j,k) == 1 || PCflag(i+1,j,k) == 1 || PCflag(i,j+1,k)==1
                       || PCflag(i,j,k+1) == 1 || PCflag(i+1,j+1,k) == 1 || PCflag(i+1,j,k+1) == 1
                       || PCflag(i,j+1,k+1) == 1 || PCflag(i+1,j+1,k+1) == 1) {
                        injection_flag(i,j,k) = 1;
                    }
                } else {
                    if ( inj_ring(i,j,k) == 1 || inj_ring(i+1,j,k) == 1 || inj_ring(i,j+1,k) == 1
                       || inj_ring(i,j,k+1) == 1 || inj_ring(i+1,j+1,k) == 1 || inj_ring(i+1,j,k+1) == 1
                       || inj_ring(i,j+1,k+1) == 1 || inj_ring(i+1,j+1,k+1) == 1) {

                        // cell-centered position based on index type
                        amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                        amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                        amrex::Real y = 0._rt;
                        amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                        amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                        amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                        amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                        amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                        amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                        amrex::Real rad = std::sqrt( (x-xc[0]) * (x-xc[0])
                                                   + (y-xc[1]) * (y-xc[1])
                                                   + (z-xc[2]) * (z-xc[2]));
                        inj_ringCC(i,j,k) = 1;
                        if (use_FixedSigmaInput == 1) {
                            if (sigma(i,j,k) > Sigma0_threshold) {
                        //if (modify_Sigma0_threshold == 1) {
                        //    Sigma_threshold = Sigma0_threshold * (Rstar/rad) * (Rstar/rad) * (Rstar/rad);
                        //}
                        //sigma_threshold_loc(i,j,k) = Sigma_threshold;
                        //// flag cells with sigma > sigma0_threshold
                        //if (sigma(i,j,k) > Sigma_threshold ) {
                        //    injection_flag(i,j,k) = 1;
                        //}
                                injection_flag(i,j,k) = 1;
                                sigma_inj_ring(i, j, k) = sigma(i, j, k) * injection_flag(i,j,k);
                            }
                        } else {
                            injection_flag(i,j,k) = 1;
                            sigma_inj_ring(i, j, k) = sigma(i, j, k) * injection_flag(i,j,k);

                        }
                    }
                }
            }
        );
    }

    // Total injection cells
    m_totalcells_injectionring = m_injection_ringCC[lev]->sum();
    amrex::Real TotalInjectionCells = SumInjectionFlag();
    amrex::Print() << " total inj cells : " << TotalInjectionCells << "\n";
    m_sum_inj_magnetization = 1;
    if (injectiontype > 0) m_sum_inj_magnetization = SumMagnetizationinInjectionRing();
    amrex::Print() << " sum inj magnetization " << m_sum_inj_magnetization << "\n";

    int num_ppc_modified = 0;
    int minInjectionCells = 1;
    if (m_use_FixedSigmaInput == 1 && m_usePCflagcount_minInjCell == 1)
    {
        minInjectionCells = m_totalpolarcap_cells * m_PCInjectionCellFraction;
        amrex::Print() << " total pc cells " << m_totalpolarcap_cells << " " << m_PCInjectionCellFraction << "\n";
    }
    amrex::Print() << " min injection cell : " << minInjectionCells << "\n";
//    if (TotalInjectionCells < minInjectionCells) {
//        if (m_use_maxsigma_for_Sigma0 == 1) {
//            amrex::Real r_rstar_fac = m_injRing_radius/m_R_star;
//            amrex::Real max_sigma = MaxMagnetization();
//            amrex::Real new_sigma0_threshold = m_maxsigma_fraction * max_sigma * r_rstar_fac * r_rstar_fac * r_rstar_fac;
//            amrex::Print() << " injec cell is 0 at pcount! " << TotalInjectionCells << " <= " << m_min_TCTP_ratio <<" *TP " << ParticlesToBeInjected << " sigma0_new modified to " << new_sigma0_threshold << " using max sigma : " << max_sigma<< "\n";
//            m_Sigma0_threshold = new_sigma0_threshold;
//            amrex::Real Sigma0_threshold = m_Sigma0_threshold;
//            m_injection_flag[lev]->setVal(0);
//            m_injected_cell[lev]->setVal(0);
//            m_sigma_inj_ring[lev]->setVal(0);
//            m_pcount[lev]->setVal(0);
//            for (amrex::MFIter mfi(*m_injection_flag[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
//            {
//                //InitializeGhost Cells also
//                const amrex::Box& tb = mfi.tilebox(iv);
//                amrex::Array4<amrex::Real> const& injection_flag = m_injection_flag[lev]->array(mfi);
//                amrex::Array4<amrex::Real> const& injected_cell = m_injected_cell[lev]->array(mfi);
//                amrex::Array4<amrex::Real> const& sigma = m_magnetization[lev]->array(mfi);
//                amrex::Array4<amrex::Real> const& inj_ring = m_injection_ring[lev]->array(mfi);
//                amrex::Array4<amrex::Real> const& sigma_inj_ring = m_sigma_inj_ring[lev]->array(mfi);
//                amrex::Array4<amrex::Real> const& sigma_threshold_loc = m_sigma_threshold[lev]->array(mfi);
//                amrex::ParallelFor(tb,
//                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
//                        sigma_threshold_loc(i,j,k) = 0.;
//                        if ( inj_ring(i,j,k) == 1 || inj_ring(i+1,j,k) == 1 || inj_ring(i,j+1,k)
//                           || inj_ring(i,j,k+1) == 1 || inj_ring(i+1,j+1,k) == 1 || inj_ring(i,j+1,k+1) == 1
//                           || inj_ring(i,j+1,k+1) == 1 || inj_ring(i+1,j+1,k+1) == 1) {
//                            // cell-centered position based on index type
//                            amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
//                            amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
//#if (AMREX_SPACEDIM==2)
//                            amrex::Real y = 0._rt;
//                            amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
//                            amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
//#else
//                            amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
//                            amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
//                            amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
//                            amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
//#endif
//                            amrex::Real rad = std::sqrt( (x-xc[0]) * (x-xc[0])
//                                                       + (y-xc[1]) * (y-xc[1])
//                                                       + (z-xc[2]) * (z-xc[2]));
//
//                            sigma_inj_ring(i, j, k) = sigma(i, j, k);
//                            amrex::Real Sigma_threshold = Sigma0_threshold;
//                            if (modify_Sigma0_threshold == 1) {
//                                Sigma_threshold = Sigma0_threshold * (Rstar/rad) * (Rstar/rad) * (Rstar/rad);
//                            }
//                            sigma_threshold_loc(i,j,k) = Sigma_threshold;
//                            // flag cells with sigma > sigma0_threshold
//                            if (sigma(i,j,k) > Sigma_threshold ) {
//                                injection_flag(i,j,k) = 1;
//                            }
//                        }
//                    }
//                );
//            }
//        TotalInjectionCells = SumInjectionFlag();
//        amrex::Print() << " redefined sigma to get total inj cells : " << TotalInjectionCells << "\n";
//        } else if (m_use_FixedSigmaInput == 1 and m_usePCflagcount_minInjCell==1) {
//            m_injection_flag[lev]->setVal(0);
//            m_injected_cell[lev]->setVal(0);
//            m_pcount[lev]->setVal(0);
//            for (amrex::MFIter mfi(*m_injection_flag[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
//            {
//                //InitializeGhost Cells also
//                const amrex::Box& tb = mfi.tilebox(iv);
//                amrex::Array4<amrex::Real> const& injection_flag = m_injection_flag[lev]->array(mfi);
//                amrex::Array4<amrex::Real> const& injected_cell = m_injected_cell[lev]->array(mfi);
//                amrex::Array4<amrex::Real> const& PCflag = m_PC_flag[lev]->array(mfi);
//                amrex::ParallelFor(tb,
//                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
//                        if ( PCflag(i,j,k) == 1 || PCflag(i+1,j,k) == 1 || PCflag(i,j+1,k)==1
//                        || PCflag(i,j,k+1) == 1 || PCflag(i+1,j+1,k) == 1 || PCflag(i+1,j,k+1) == 1
//                        || PCflag(i,j+1,k+1) == 1 || PCflag(i+1,j+1,k+1) == 1) {
//                            injection_flag(i,j,k) = 1;
//                        }
//                    }
//                );
//            }
//            TotalInjectionCells = SumInjectionFlag();
//            amrex::Print() << " total inj cells is equal to totalPCCells: " << TotalInjectionCells << " total PC cells " << m_totalpolarcap_cells << "\n";
//        }
//    }
//    num_ppc_modified = static_cast<int>( ParticlesToBeInjected/TotalInjectionCells);
    amrex::Print() << " particle to be inj " << ParticlesToBeInjected << "\n";
//    amrex::Real num_ppc_modified_real = 0.;
//    amrex::Real num_ppc_PC_real = 0.;
//    amrex::Real num_ppc_eq_real = 0.;
    int GJParticles = ParticlesToBeInjected/m_injection_rate;
    if (injectiontype == 0) {
        GJParticles = ParticlesToBeInjected;
    } else if (injectiontype == 2) {
        GJParticles = 0;
    }
    int PairPlasmaParticles = ParticlesToBeInjected - GJParticles;
    amrex::Real num_GJParticles = 0.;
    if (use_FixedSigmaInput == 1) {
        num_GJParticles = GJParticles/(m_totalcells_injectionring);
        amrex::Print() << " GJParticles : " << GJParticles << " total cells in inj ting " << m_totalcells_injectionring << " TotalInjCels " << TotalInjectionCells << " num gp particles per cell " << num_GJParticles << "\n";
    } else {
        if (TotalInjectionCells > 0) {
//            num_ppc_modified_real = ParticlesToBeInjected/TotalInjectionCells;
//            num_ppc_PC_real = 0.67*ParticlesToBeInjected/m_totalpolarcap_cells;
//            num_ppc_eq_real = 0.33*ParticlesToBeInjected/(TotalInjectionCells - m_totalpolarcap_cells);
            num_GJParticles = GJParticles/TotalInjectionCells;
        }
    }
//    int num_ppc_PC = static_cast<int>(num_ppc_PC_real);
//    int num_ppc_eq = static_cast<int>(num_ppc_eq_real);
//    amrex::Print() << " num pcc real " << num_ppc_modified_real << "\n";
//    amrex::Print() << " num pcc PC " << num_ppc_PC_real << "\n";
//    amrex::Print() << " num pcc eq " << num_ppc_eq_real << "\n";
//    amrex::Print() << " ppc int : " << num_ppc_modified << "\n";
    amrex::Print() << " GJ particles : " << GJParticles << " Plasmapair particles  " << PairPlasmaParticles << "\n";

    amrex::Real rho_GJ_fac = 2. * m_omega_star * m_B_star * 8.85e-12;
    amrex::Real particle_wt = m_particle_wt;
    amrex::Print() << " particle wt " << particle_wt << "\n";
    amrex::Real injection_fac = m_injection_rate;
    amrex::Real dt = warpx.getdt(0);
    int use_injection_rate = m_use_injection_rate;
    amrex::Real density_thresholdfactor = m_GJdensity_thresholdfactor;
    amrex::Real sum_magnetization = m_sum_inj_magnetization;
    amrex::Real GJdensitythreshold = m_injection_GJdensitythreshold;
    // fill pcounts and injected cell flag
    amrex::Real chi = m_Chi;
    for (amrex::MFIter mfi(*m_injection_flag[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tb = mfi.tilebox(iv);
        amrex::Array4<amrex::Real> const& injection_flag = m_injection_flag[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& injected_cell = m_injected_cell[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& pcount = m_pcount[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& PCflag = m_PC_flag[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& sigma_inj_ring = m_sigma_inj_ring[lev]->array(mfi);
        amrex::ParallelForRNG(tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
            {
                injected_cell(i,j,k) = 0;
                pcount(i,j,k) = 0;
                //if (injection_flag(i,j,k) == 1) {
                //    pcount(i,j,k) = num_ppc_modified;
                //    if ( PCflag(i,j,k) == 1 || PCflag(i+1,j,k) == 1 || PCflag(i,j+1,k)==1
                //    || PCflag(i,j,k+1) == 1 || PCflag(i+1,j+1,k) == 1 || PCflag(i+1,j,k+1) == 1
                //    || PCflag(i,j+1,k+1) == 1 || PCflag(i+1,j+1,k+1) == 1) {
		//        if (num_ppc_PC == 0) {
		//            amrex::Real r1 = amrex::Random(engine);
                //            if (r1 <= num_ppc_PC_real) {
                //                injected_cell(i,j,k) = 1;
                //                pcount(i,j,k) = 1;
		//            }
		//        } else if (num_ppc_PC > 0) {
                //            amrex::Real frac = num_ppc_PC_real - num_ppc_PC;
		//            amrex::Real r1 = amrex::Random(engine);
                //            if (r1 <= frac) {
                //                injected_cell(i,j,k) = 1;
                //                pcount(i,j,k) = num_ppc_PC + 1;
                //            }
		//        }
		//    } else {
		//        if (num_ppc_eq == 0) {
                //            amrex::Real r1 = amrex::Random(engine);
                //            if (r1 <= num_ppc_eq_real) {
                //                injected_cell(i,j,k) = 1;
                //                pcount(i,j,k) = 1;
                //            }
                //        } else if (num_ppc_eq > 0) {
                //            amrex::Real frac = num_ppc_eq_real - num_ppc_eq;
                //            amrex::Real r1 = amrex::Random(engine);
                //            if (r1 <= frac) {
                //                injected_cell(i,j,k) = 1;
                //                pcount(i,j,k) = num_ppc_eq + 1;
                //            }
                //        }
		//    }
                //    //if (num_ppc_modified == 0) {
                //    //    // particle injection done probabilistically
                //    //    amrex::Real r1 = amrex::Random(engine);
                //    //    if (r1 <= num_ppc_modified_real) {
                //    //        injected_cell(i,j,k) = 1;
                //    //        pcount(i,j,k) = 1;
                //    //    }
                //    //} else if (num_ppc_modified > 0) {
                //    //    amrex::Real particle_fraction = num_ppc_modified_real - num_ppc_modified;
                //    //    amrex::Real r1 = amrex::Random(engine);
                //    //    if (r1 <= particle_fraction) {
                //    //        // additional particle included if probability is satisfied
                //    //        pcount(i,j,k) = num_ppc_modified + 1;
                //    //    }
                //    //    injected_cell(i,j,k) = 1;
                //    //}
                //}
                if (injection_flag(i,j,k) == 1) {
                    // Shift x, y, z position based on index type
                    amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                    amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                    amrex::Real y = 0._rt;
                    amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                    amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                    amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                    amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                    amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                    amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                    amrex::Real r, theta, phi;
                    ConvertCartesianToSphericalCoord(x, y, z, xc,
                                                     r, theta, phi);
                    amrex::Real q = 1.609e-19;
                    amrex::Real rho_GJ = rho_GJ_fac * (1. - 3. * std::cos(theta) * std::cos(theta) );
                    amrex::Real n_GJ = amrex::Math::abs(rho_GJ)/q;
		    //amrex::Real num_part_real = 0.;
		    amrex::Real shifted_theta = theta - chi;
                    amrex::Real GJ_factor = amrex::Math::abs( 1. - 3. * std::cos(shifted_theta)*std::cos(shifted_theta));
		    if (GJ_factor < GJdensitythreshold) GJ_factor = GJdensitythreshold;
                    amrex::Real sigma_factor = sigma_inj_ring(i,j,k)/sum_magnetization;
                    amrex::Real num_part_cell = (num_GJParticles * GJ_factor) + (PairPlasmaParticles * sigma_factor);
                    //amrex::Real num_part_cell = num_ppc_modified_real * factor;
                    if (use_injection_rate == 1) {
		        amrex::Real GJ_inj_rate = n_GJ * dx_lev[0] * dx_lev[0] * 3.e8;
		        num_part_cell = GJ_inj_rate * injection_fac * dt / particle_wt;
                    }
                    int numpart_int = static_cast<int>(num_part_cell);
                    if (numpart_int == 0) {
                        amrex::Real r1 = amrex::Random(engine);
                        if (r1 <= num_part_cell) {
                            injected_cell(i,j,k) = 1;
                            pcount(i,j,k) = 1;
                        }
                    } else if (numpart_int > 0){
                        injected_cell(i,j,k) = 1;
                        pcount(i,j,k) = numpart_int;
                        amrex::Real particle_fraction = num_part_cell - numpart_int;
                        amrex::Real r1 = amrex::Random(engine);
                        if (r1 <= particle_fraction) {
                            pcount(i,j,k) = numpart_int + 1;
                        }
                    }
		    //if (use_injection_rate == 1) {
		    //    amrex::Real GJ_inj_rate = n_GJ * dx_lev[0] * dx_lev[0] * 3.e8;
		    //    num_part_real = GJ_inj_rate * injection_fac * dt / particle_wt;
		    //} else {
		    //    amrex::Real GJ_inj_rate = n_GJ * dx_lev[0] * dx_lev[1] * dx_lev[2];
		    //    num_part_real = GJ_inj_rate * injection_fac / particle_wt;
		    //}
		    //int num_part = static_cast<int>(num_part_real);
		    //if (num_part_real >0 and num_part == 0) {
		    //    amrex::Real r1 = amrex::Random(engine);
		    //    if (r1 <= num_part_real) {
		    //        pcount(i,j,k) = 1;
		    //        injected_cell(i,j,k) = 1;
		    //    }
		    //} else if (num_part_real >0 and num_part > 0) {
		    //    pcount(i,j,k) = num_part;
		    //    injected_cell(i,j,k) = 1;
		    //    amrex::Real particle_fraction = num_part_real - num_part;
		    //    amrex::Real r1 = amrex::Random(engine);
		    //    if (r1 <= particle_fraction) {
		    //        pcount(i,j,k) = num_part + 1;
		    //    }
		    //}
		    if (density_thresholdfactor >= 0 ) {
		        if ( amrex::Math::abs(rho_GJ) < (density_thresholdfactor * rho_GJ_fac) ) {
		            pcount(i,j,k) = 0;
		            injected_cell(i,j,k) = 0;
		        }
		    }
		    //if (density_thresholdfactor < 0 ) {
		    //    if ( amrex::Math::abs(rho_GJ) < ( amrex::Math::abs(density_thresholdfactor) * rho_GJ_fac) ) {
                    //        rho_GJ = rho_GJ_fac * amrex::Math::abs(density_thresholdfactor);
                    //        n_GJ = amrex::Math::abs(rho_GJ)/q;
		    //        num_part_real = 0.;
                    //        if (use_injection_rate == 1) {
                    //            amrex::Real GJ_inj_rate = n_GJ * dx_lev[0] * dx_lev[0] * 3.e8;
                    //            num_part_real = GJ_inj_rate * injection_fac * dt / particle_wt;
                    //        } else {
                    //            amrex::Real GJ_inj_rate = n_GJ * dx_lev[0] * dx_lev[1] * dx_lev[2];
                    //            num_part_real = GJ_inj_rate * injection_fac / particle_wt;
                    //        }
                    //        num_part = static_cast<int>(num_part_real);
                    //        if (num_part_real >0 and num_part == 0) {
                    //            amrex::Real r1 = amrex::Random(engine);
                    //            if (r1 <= num_part_real) {
                    //                pcount(i,j,k) = 1;
                    //                injected_cell(i,j,k) = 1;
                    //            }
                    //        } else if (num_part_real >0 and num_part > 0) {
                    //            pcount(i,j,k) = num_part;
                    //            injected_cell(i,j,k) = 1;
                    //            amrex::Real particle_fraction = num_part_real - num_part;
                    //            amrex::Real r1 = amrex::Random(engine);
                    //            if (r1 <= particle_fraction) {
                    //                pcount(i,j,k) = num_part + 1;
                    //            }
                    //        } // num part real
		    //    } // if rho GJ is small
		    //} // density threshold factor neg
		} // injection flag is 1
            }
        );
    }
    amrex::Print() << "pcount sum " << PcountSum() << "\n";
    int limit_injection = m_GJdensity_limitinjection;
    amrex::Real limit_GJ_factor = m_limit_GJfactor;
    //const amrex::MultiFab& rho_mf = warpx.getrho_fp(lev);
    std::unique_ptr<amrex::MultiFab> rho;
    auto& mypc = warpx.GetPartContainer();
    rho = mypc.GetChargeDensity(lev, true);
    amrex::MultiFab & rho_mf = *rho;
    if (limit_injection == 1) {
    for (amrex::MFIter mfi(*m_injection_flag[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
	amrex::Array4<const amrex::Real> const& rho_arr = rho_mf[mfi].array();
        amrex::Array4<amrex::Real> const& injected_cell = m_injected_cell[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& pcount = m_pcount[lev]->array(mfi);
        amrex::Array4<amrex::Real> const& ndens = m_plasma_number_density[lev]->array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type
                amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                amrex::Real r, theta, phi;
                ConvertCartesianToSphericalCoord(x, y, z, xc,
                                                 r, theta, phi);
		amrex::Real q = 1.609e-19;
		amrex::Real rho_GJ = rho_GJ_fac * (1. - 3. * std::cos(theta) * std::cos(theta) );
		amrex::Real n_GJ = amrex::Math::abs(rho_GJ)/q;
		//amrex::Real n = amrex::Math::abs(rho(i,j,k))/q;
		amrex::Real n = ndens(i,j,k,0)+ndens(i,j,k,1);
		if ( (n > (limit_GJ_factor * n_GJ) ) && limit_injection == 1 ) {
		    injected_cell(i,j,k) = 0;
		    pcount(i,j,k) = 0;
		}

	    }
        );
    }
    }
//    amrex::Real TotalInjectedCells = SumInjectedCells();
//    amrex::Print() << " total injected cells : " << TotalInjectedCells << "\n";



}
