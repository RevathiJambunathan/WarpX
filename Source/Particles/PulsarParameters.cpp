#include "PulsarParameters.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include "WarpX.H"


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


Pulsar::Pulsar ()
{
    ReadParameters();
}

void
Pulsar::ReadParameters () {
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
    pp.query("init_corotatingEfield", m_do_InitializeGrid_with_Pulsar_Efield);
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
    pp.query("AddMonopoleInsideRstarOnGrid", m_AddMonopoleInsideRstarOnGrid);
    pp.query("EnforceTheoreticalEBInGrid", m_EnforceTheoreticalEBInGrid);
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
    GpuArray<amrex::Real, 3> center_star_arr;
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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tbx = mfi.tilebox(x_nodal_flag, mfx->nGrowVect() );
        const amrex::Box& tby = mfi.tilebox(y_nodal_flag, mfx->nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox(z_nodal_flag, mfx->nGrowVect() );

        amrex::Array4<amrex::Real> const& mfx_arr = mfx->array(mfi);
        amrex::Array4<amrex::Real> const& mfy_arr = mfy->array(mfi);
        amrex::Array4<amrex::Real> const& mfz_arr = mfz->array(mfi);

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
                // Initialize corotating EField in r < corotating
                if (init_Bfield == 0) {
                    if (r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
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
                // Initialize corotating Efield in r < corotating
                if (init_Bfield == 0) {
                    if (r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
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
                    if (r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
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
    amrex::Print() << " applying corotating Efield BC\n";
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
    GpuArray<amrex::Real, 3> center_star_arr;
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
                    if (r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ex_arr(i,j,k));
                    }
                } else {
                    ExternalEFieldSpherical(r, theta, phi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data,
                                            corotatingE_maxradius_data,
                                            E_external_monopole_data,
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
                    if (r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ey_arr(i,j,k));
                    }
                } else {
                    ExternalEFieldSpherical(r, theta, phi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data,
                                            corotatingE_maxradius_data,
                                            E_external_monopole_data,
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
                    if (r <= corotatingE_maxradius_data) {
                        CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                  omega_star_data,
                                                  ramp_omega_time_data,
                                                  Bstar_data, Rstar_data, dRstar_data,
                                                  Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ez_arr(i,j,k));
                    }
                } else {
                    ExternalEFieldSpherical(r, theta, phi, cur_time,
                                            omega_star_data,
                                            ramp_omega_time_data,
                                            Bstar_data, Rstar_data,
                                            corotatingE_maxradius_data,
                                            E_external_monopole_data,
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
    amrex::Print() << " applying corotating Bfield BC\n";
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
    GpuArray<amrex::Real, 3> center_star_arr;
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
                    if (r <= enforceDipoleB_maxradius_data) {
                        ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                Bstar_data, Rstar_data, dRstar_data,
                                                Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Bx_arr(i,j,k));
                    }
                    else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
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
                    if (r <= enforceDipoleB_maxradius_data) {
                        ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                Bstar_data, Rstar_data, dRstar_data,
                                               Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, By_arr(i,j,k));
                    } else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
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
                    if (r <= enforceDipoleB_maxradius_data) {
                        ExternalBFieldSpherical(r, theta, phi, cur_time,
                                               Bstar_data, Rstar_data, dRstar_data,
                                               Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Bz_arr(i,j,k));
                    } else if ( r > enforceDipoleB_maxradius_data && r <= corotatingE_maxradius_data) {
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
