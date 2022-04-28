#include "PulsarParameters.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXConst.H"
#include "Utils/CoarsenIO.H"
#include "Utils/IntervalsParser.H"
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
IntervalsParser Pulsar::m_injection_tuning_interval;
amrex::Real Pulsar::m_min_Sigma0;
amrex::Real Pulsar::m_max_Sigma0;
amrex::Real Pulsar::m_sum_injection_rate = 0.;
std::string Pulsar::m_sigma_tune_method;
int Pulsar::ROI_avg_window_size = 50;


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
    m_GJ_injection_rate = ( 8._rt * MathConst::pi * PhysConst::ep0 * m_B_star
                            * m_omega_star * m_omega_star * m_R_star * m_R_star * m_R_star
                          ) / PhysConst::q_e;
    // the factor of 2 is because B at pole = 2*B at equator
    m_Sigma0_threshold = (2. * m_B_star * 2 * m_B_star)  / (m_max_ndens * 2. * PhysConst::mu0 * PhysConst::m_e
                                         * PhysConst::c * PhysConst::c);
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
        Store_parserString(pp, "conductor_function(x,y,z)", m_str_conductor_function);
        m_conductor_parser = std::make_unique<amrex::Parser>(
                                  makeParser(m_str_conductor_function,{"x","y","z"}));
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
    m_injection_tuning_interval = IntervalsParser(intervals_string_vec);
    pp.get("sigma_tune_method",m_sigma_tune_method);
    amrex::Print() << " sigma tune method " << m_sigma_tune_method << "\n";
    pp.get("ROI_avg_size",ROI_avg_window_size);
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
        // initialize number density
        m_plasma_number_density[lev]->setVal(0._rt);
        // initialize magnetization
        m_magnetization[lev]->setVal(0._rt);
        // initialize flag for plasma injection
        m_injection_flag[lev]->setVal(0._rt);
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
Pulsar::InitDataAtRestart ()
{
    auto & warpx = WarpX::GetInstance();
    const int nlevs_max = warpx.maxLevel() + 1;
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
        // initialize number density
        m_plasma_number_density[lev]->setVal(0._rt);
        // initialize magnetization
        m_magnetization[lev]->setVal(0._rt);
        m_injection_flag[lev]->setVal(0._rt);
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
    for ( amrex::MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
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
    GpuArray<int, 3> x_IndexType;
    GpuArray<int, 3> y_IndexType;
    GpuArray<int, 3> z_IndexType;
    GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;
    for (int idim = 0; idim < 3; ++idim) {
        x_IndexType[idim] = x_nodal_flag[idim];
        y_IndexType[idim] = y_nodal_flag[idim];
        z_IndexType[idim] = z_nodal_flag[idim];
        center_star_arr[idim] = m_center_star[idim];
    }
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
    for (MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
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
                            ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                        }
                    } else {
                        ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                Bstar_data, Rstar_data, dRstar_data,
                                                Fr, Ftheta, Fphi);
                    }
                }
                if (init_Bfield == 0) {
                    if (InitializeFullExternalEField == 0) {
                        if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1 ) {
                            // Edge-centered Efield is fully immersed in conductor
                            // Therefore, apply corotating electric field E = v X B
                            CorotatingEfieldSpherical(r, theta, phi, cur_time,
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
                        ExternalEFieldSpherical(r, theta, phi, cur_time,
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
                            ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                        }
                    } else {
                        ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                Bstar_data, Rstar_data, dRstar_data,
                                                Fr, Ftheta, Fphi);
                    }
                }
                if (init_Bfield == 0) {
                    if (InitializeFullExternalEField == 0) {
                        if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1 ) {
                            // Edge-centered Efield is fully immersed in conductor
                            // Therefore, apply corotating electric field E = v X B
                            CorotatingEfieldSpherical(r, theta, phi, cur_time,
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
                        ExternalEFieldSpherical(r, theta, phi, cur_time,
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
                            ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
                        }
                    } else {
                        ExternalBFieldSpherical( r, theta, phi, cur_time,
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
                            CorotatingEfieldSpherical(r, theta, phi, cur_time,
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
                        ExternalEFieldSpherical(r, theta, phi, cur_time,
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
    GpuArray<int, 3> x_IndexType;
    GpuArray<int, 3> y_IndexType;
    GpuArray<int, 3> z_IndexType;
    GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;
    for (int idim = 0; idim < 3; ++idim) {
        x_IndexType[idim] = x_nodal_flag[idim];
        y_IndexType[idim] = y_nodal_flag[idim];
        z_IndexType[idim] = z_nodal_flag[idim];
        center_star_arr[idim] = m_center_star[idim];
    }
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
    for (MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
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
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ex_arr(i,j,k));
                    }
                } else {
                    int ApplyCorotatingEField = 0;
                    if (conductor(i,j,k)==1 and conductor(i+1,j,k)==1) ApplyCorotatingEField = 1;
                    ExternalEFieldSpherical(r, theta, phi, cur_time,
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
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ey_arr(i,j,k));
                    }
                } else {
                    int ApplyCorotatingEField = 0;
                    if (conductor(i,j,k)==1 and conductor(i,j+1,k)==1) ApplyCorotatingEField = 1;
                    ExternalEFieldSpherical(r, theta, phi, cur_time,
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
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ez_arr(i,j,k));
                    }
                } else {
                    int ApplyCorotatingEField = 0;
                    if (conductor(i,j,k)==1 and conductor(i,j,k+1)==1) ApplyCorotatingEField = 1;
                    ExternalEFieldSpherical(r, theta, phi, cur_time,
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
    GpuArray<int, 3> x_IndexType;
    GpuArray<int, 3> y_IndexType;
    GpuArray<int, 3> z_IndexType;
    GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;
    for (int idim = 0; idim < 3; ++idim) {
        x_IndexType[idim] = x_nodal_flag[idim];
        y_IndexType[idim] = y_nodal_flag[idim];
        z_IndexType[idim] = z_nodal_flag[idim];
        center_star_arr[idim] = m_center_star[idim];
    }
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
    for (MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
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
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                    Fr, Ftheta, Fphi);
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
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                    ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                    Bstar_data, Rstar_data, dRstar_data,
                                                   Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, By_arr(i,j,k));
                        }
                    } else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
                        // Dipole magnetic field applied for r < corotating radius and then
                        // a smoothening function tanh damping is applied between radius of star
                        // to radius where corotating electric field is enforced.
                        if (DampBDipoleInRing_data == 1) {
                            // from inner ring to outer ring
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                    ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                   Bstar_data, Rstar_data, dRstar_data,
                                                   Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Bz_arr(i,j,k));
                        }
                    } else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
                        // Dipole magnetic field applied for r < corotating radius and then
                        // a smoothening function tanh damping is applied between radius of star
                        // to radius where corotating electric field is enforced.
                        if (DampBDipoleInRing_data == 1) {
                            // from inner ring to outer ring
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                    ExternalBFieldSpherical(r, theta, phi, cur_time,
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
    for (MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
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
    std::vector species_names = warpx.GetPartContainer().GetSpeciesNames();
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
                    // Ideally this will be replaced with the AMReX NGP interpolator
                    // Always do x direction. No RZ case because it's not implemented, and code
                    // will have aborted
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

 
        const Geometry& geom = warpx.Geom(lev);
        const auto dx = geom.CellSizeArray();
#if defined WARPX_DIM_3D
        amrex::Real inv_vol = 1._rt/(dx[0] * dx[1] * dx[2]);
#elif defined(WARPX_DIM_XZ)
        amrex::Real inv_vol = 1._rt/(dx[0] * dx[1] );
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*m_plasma_number_density[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
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
    std::vector species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    const amrex::Real min_ndens = 1.e-16; // make this user-defined
    constexpr amrex::Real mu0_m_c2_inv = 1._rt / (PhysConst::mu0 * PhysConst::m_e
                                                 * PhysConst::c * PhysConst::c);
    for (int lev = 0; lev < nlevs_max; ++lev) {
        m_magnetization[lev]->setVal(0._rt);
        const amrex::MultiFab& Bx_mf = warpx.GetInstance().getBfield(lev, 0);
        const amrex::MultiFab& By_mf = warpx.GetInstance().getBfield(lev, 1);
        const amrex::MultiFab& Bz_mf = warpx.GetInstance().getBfield(lev, 2);
        for (MFIter mfi(*m_magnetization[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
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
                    // magnitude of magnetic field
                    amrex::Real B_mag = std::sqrt( Bx(i, j, k) * Bx(i, j, k)
                                                 + By(i, j, k) * By(i, j, k)
                                                 + Bz(i, j, k) * Bz(i, j, k) );
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
    std::vector species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    amrex::Real total_weight_allspecies = 0._rt;
    amrex::Real dt = warpx.getdt(0);

    using PTDType = typename WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;
    for (int isp = 0; isp < nspecies; ++isp) {
        amrex::Real ws_total = 0._rt;
        auto& pc = warpx.GetPartContainer().GetParticleContainer(isp);
        amrex::ReduceOps<ReduceOpSum> reduce_ops;
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
        amrex::ParallelDescriptor::ReduceRealSum(ws_total, ParallelDescriptor::IOProcessorNumber());
        total_weight_allspecies += ws_total;
    }
    amrex::Real current_injection_rate = total_weight_allspecies / dt;
    if (list_size < ROI_avg_window_size) {
        ROI_list.push_back(current_injection_rate);
        m_sum_injection_rate += current_injection_rate;
        list_size++;
    } else {
        amrex::Print() << " front : " << ROI_list.front() << "\n";
        m_sum_injection_rate -= ROI_list.front();
        ROI_list.pop_front();
        ROI_list.push_back(current_injection_rate);
        amrex::Print() << " current injection rate : " << current_injection_rate << " list back " << ROI_list.back() << "\n";
        m_sum_injection_rate += ROI_list.back();
    }
    amrex::Print() << " current_injection rate " << current_injection_rate << " sum : " << m_sum_injection_rate << "\n";
    amrex::Real specified_injection_rate = m_GJ_injection_rate * m_injection_rate;
    if (m_injection_tuning_interval.contains(step+1) ) {
        amrex::Print() << " period for injection tuning : " << m_injection_tuning_interval.localPeriod(step+1) << "\n";
        amrex::Print() << " species rate is :  " << specified_injection_rate << " current rate : " << current_injection_rate << "\n";
        amrex::Print() << " Sigma0 before mod : " << m_Sigma0_threshold << "\n";
        amrex::Real m_Sigma0_pre = m_Sigma0_threshold;
//        amrex::Real avg_injection_rate = m_sum_injection_rate/(m_injection_tuning_interval.localPeriod(step+1) * 1.);
        amrex::Print() << " list size : " << list_size << "\n";
        amrex::Real avg_injection_rate = m_sum_injection_rate/(list_size * 1.);
        amrex::Print() << " avg inj rate " << avg_injection_rate << "\n";
        if (avg_injection_rate < specified_injection_rate) {
            // reduce sigma0 so more particles can be injected
            //m_Sigma0_threshold *= total_weight_allspecies/specified_injection_rate;
            if (m_sigma_tune_method == "10percent") {
                m_Sigma0_threshold = m_Sigma0_threshold - 0.1 * m_Sigma0_threshold;
            }
            if (m_sigma_tune_method == "relative_difference") {
                if (avg_injection_rate == 0.) {
                    // if no particles are injection only change the threshold sigma by 10%
                    m_Sigma0_threshold = m_Sigma0_threshold - 0.1 * m_Sigma0_threshold;
                } else {
                    amrex::Real rel_diff = (specified_injection_rate - avg_injection_rate)/specified_injection_rate;
                    if (rel_diff < 0.1) {
                        amrex::Print() << " rel_diff " << rel_diff << "\n";
                        m_Sigma0_threshold = m_Sigma0_threshold - rel_diff * m_Sigma0_threshold;
                    } else {
                        amrex::Print() << " rel_diff " << rel_diff << " using upper bound 10%"<< "\n";
                        m_Sigma0_threshold = m_Sigma0_threshold - 0.1 * m_Sigma0_threshold;
                    }
                }
            }
        } else if (avg_injection_rate > specified_injection_rate ) {
            //m_Sigma0_threshold *= specified_injection_rate/total_weight_allspecies;
            if (m_sigma_tune_method == "10percent") {
                m_Sigma0_threshold = m_Sigma0_threshold + 0.1 * m_Sigma0_threshold;
            }
            if (m_sigma_tune_method == "relative_difference") {
                amrex::Real rel_diff = (avg_injection_rate - specified_injection_rate)/specified_injection_rate;
                if (rel_diff < 0.1) {
                    amrex::Print() << " rel_diff " << rel_diff << "\n";
                    m_Sigma0_threshold = m_Sigma0_threshold + rel_diff * m_Sigma0_threshold;
                } else {
                    amrex::Print() << " rel_diff " << rel_diff << " using upper bound 10%"<< "\n";
                    m_Sigma0_threshold = m_Sigma0_threshold + 0.1 * m_Sigma0_threshold;
                }
            }
        }
        if (m_Sigma0_threshold < m_min_Sigma0) m_Sigma0_threshold = m_min_Sigma0;
        if (m_Sigma0_threshold > m_max_Sigma0) m_Sigma0_threshold = m_max_Sigma0;
        amrex::Print() << " Simg0 modified to : " << m_Sigma0_threshold << "\n";
        amrex::AllPrintToFile("RateOfInjection") << warpx.getistep(0) << " " << warpx.gett_new(0) << " " << dt <<  " " << specified_injection_rate << " " << avg_injection_rate << " " << m_Sigma0_pre << " "<< m_Sigma0_threshold << " " << m_min_Sigma0 << " " << m_max_Sigma0 << " " << m_Sigma0_baseline<< "\n";
    }
    amrex::AllPrintToFile("ROI") << warpx.getistep(0) << " " << warpx.gett_new(0) << " " << dt <<  " " << specified_injection_rate << " " << current_injection_rate  << " "<< m_Sigma0_threshold << " " << "\n";
}

void
Pulsar::TotalParticles ()
{
    auto& warpx = WarpX::GetInstance();
    std::vector species_names = warpx.GetPartContainer().GetSpeciesNames();
    const int nspecies = species_names.size();
    amrex::Long total_particles = 0; 

    for (int isp = 0; isp < nspecies; ++isp) {
        auto& pc = warpx.GetPartContainer().GetParticleContainer(isp);
        amrex::Long np_total = pc.TotalNumberOfParticles();
        amrex::ParallelDescriptor::ReduceLongSum(np_total, ParallelDescriptor::IOProcessorNumber());
        total_particles += np_total;
    }
    amrex::Print() << " total particles " << total_particles << "\n";
}
