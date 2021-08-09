#include "PulsarParameters.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include "WarpX.H"

namespace PulsarParm
{
    std::string pulsar_type;

    AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real ramp_omega_time = -1.0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real dR_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real damping_scale = 10.0;
    AMREX_GPU_DEVICE_MANAGED int EB_external = 0;
    AMREX_GPU_DEVICE_MANAGED int E_external_monopole = 0;
    AMREX_GPU_DEVICE_MANAGED
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star;
    AMREX_GPU_DEVICE_MANAGED int damp_EB_internal = 0;
    AMREX_GPU_DEVICE_MANAGED int verbose = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_ndens;
    AMREX_GPU_DEVICE_MANAGED amrex::Real Ninj_fraction;
    AMREX_GPU_DEVICE_MANAGED int ModifyParticleWtAtInjection = 1;
    AMREX_GPU_DEVICE_MANAGED amrex::Real rhoGJ_scale;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_EBcorotating_radius;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_EBdamping_radius;
    AMREX_GPU_DEVICE_MANAGED int turnoffdeposition = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_nodepos_radius;
    AMREX_GPU_DEVICE_MANAGED int turnoff_plasmaEB_gather = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_nogather_radius;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_particle_absorption_radius;
    AMREX_GPU_DEVICE_MANAGED amrex::Real particle_inject_rmin;
    AMREX_GPU_DEVICE_MANAGED amrex::Real particle_inject_rmax;
    AMREX_GPU_DEVICE_MANAGED amrex::Real corotatingE_maxradius;
    AMREX_GPU_DEVICE_MANAGED amrex::Real enforceDipoleB_maxradius;
    AMREX_GPU_DEVICE_MANAGED amrex::Real InitializeGrid_with_Pulsar_Bfield;
    AMREX_GPU_DEVICE_MANAGED amrex::Real InitializeGrid_with_Pulsar_Efield;
    AMREX_GPU_DEVICE_MANAGED int enforceCorotatingE;
    AMREX_GPU_DEVICE_MANAGED int enforceDipoleB;
    AMREX_GPU_DEVICE_MANAGED int singleParticleTest;
    AMREX_GPU_DEVICE_MANAGED amrex::Real Bdamping_scale = 10;
    AMREX_GPU_DEVICE_MANAGED int DampBDipoleInRing = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real injection_time = 0;
    AMREX_GPU_DEVICE_MANAGED int continuous_injection = 1;
    AMREX_GPU_DEVICE_MANAGED amrex::Real removeparticle_theta_min = 90.;
    AMREX_GPU_DEVICE_MANAGED amrex::Real removeparticle_theta_max = 0.;
    AMREX_GPU_DEVICE_MANAGED int use_theoreticalEB = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real theory_max_rstar = 0.;
    AMREX_GPU_DEVICE_MANAGED int LimitDipoleBInit = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real DipoleB_init_maxradius;
    AMREX_GPU_DEVICE_MANAGED int AddExternalMonopoleOnly = 0;
    AMREX_GPU_DEVICE_MANAGED int EnforceTheoreticalEBInGrid = 0;

    void ReadParameters() {
        amrex::ParmParse pp("pulsar");
        pp.query("pulsarType",pulsar_type);
        pp.get("omega_star",omega_star);
        amrex::Vector<amrex::Real> center_star_v(AMREX_SPACEDIM);
        pp.getarr("center_star",center_star_v);
        std::copy(center_star_v.begin(),center_star_v.end(),center_star.begin());
        pp.get("R_star",R_star);
        pp.get("B_star",B_star);
        pp.get("dR",dR_star);
        pp.query("verbose",verbose);
        pp.query("EB_external",EB_external);
        pp.query("E_external_monopole",E_external_monopole);
        pp.query("damp_EB_internal",damp_EB_internal);
        pp.query("damping_scale",damping_scale);
        pp.query("ramp_omega_time",ramp_omega_time);
        amrex::Print() << " Pulsar center: " << center_star[0] << " " << center_star[1] << " " << center_star[2] << "\n";
        amrex::Print() << " Pulsar omega: " << omega_star << "\n";
        amrex::Print() << " Pulsar B_star : " << B_star << "\n";
        pp.get("max_ndens", max_ndens);
        pp.get("Ninj_fraction",Ninj_fraction);

        particle_inject_rmin = R_star - dR_star;
        particle_inject_rmax = R_star;
        pp.query("particle_inj_rmin", particle_inject_rmin);
        pp.query("particle_inj_rmax", particle_inject_rmax);
        amrex::Print() << " min radius of particle injection : " << particle_inject_rmin << "\n";
        amrex::Print() << " max radius of particle injection : " << particle_inject_rmax << "\n";

        pp.query("ModifyParticleWeight", ModifyParticleWtAtInjection);
        // The maximum radius within which particles are absorbed/deleted every timestep.
        max_particle_absorption_radius = R_star;
        pp.query("max_particle_absorption_radius", max_particle_absorption_radius);
        pp.get("rhoGJ_scale",rhoGJ_scale);
        amrex::Print() << " pulsar max ndens " << max_ndens << "\n";
        amrex::Print() << " pulsar ninj fraction " << Ninj_fraction << "\n";
        amrex::Print() << " pulsar modify particle wt " << ModifyParticleWtAtInjection << "\n";
        amrex::Print() << " pulsar rhoGJ scaling " << rhoGJ_scale << "\n";
        amrex::Print() << " EB_external : " << EB_external << "\n";
        if (EB_external == 1) {
            // Max corotating radius defines the region where the EB field shifts from
            // corotating (v X B) to quadrapole. default is R_star.
            max_EBcorotating_radius = R_star;
            pp.query("EB_corotating_maxradius", max_EBcorotating_radius);
            amrex::Print() << " EB coratating maxradius : " << max_EBcorotating_radius << "\n";
        }
        if (damp_EB_internal == 1) {
            // Radius of the region within which the EB fields are damped
            max_EBdamping_radius = R_star;
            pp.query("damp_EB_radius", max_EBdamping_radius);
            amrex::Print() << " max EB damping radius : " << max_EBdamping_radius << "\n";
        }
        // query to see if particle j,rho deposition should be turned off
        // within some region interior to the star
        // default is 0
        pp.query("turnoffdeposition", turnoffdeposition);
        amrex::Print() << " is deposition off ? " << turnoffdeposition << "\n";
        if (turnoffdeposition == 1) {
            max_nodepos_radius = R_star;
            pp.query("max_nodepos_radius", max_nodepos_radius);
            amrex::Print() << " deposition turned off within radius : " << max_nodepos_radius << "\n";
        }
        pp.query("turnoff_plasmaEB_gather", turnoff_plasmaEB_gather);
        amrex::Print() << " is plasma EB gather off ? " << turnoff_plasmaEB_gather << "\n";
        if (turnoff_plasmaEB_gather == 1) {
            max_nogather_radius = R_star;
            pp.query("max_nogather_radius", max_nogather_radius);
            amrex::Print() << " gather off within radius : " << max_nogather_radius << "\n";
        }
        corotatingE_maxradius = R_star;
        enforceDipoleB_maxradius = R_star;
        pp.query("corotatingE_maxradius", corotatingE_maxradius);
        pp.query("enforceDipoleB_maxradius", enforceDipoleB_maxradius);
        InitializeGrid_with_Pulsar_Bfield = 1;
        InitializeGrid_with_Pulsar_Efield = 1;
        pp.query("init_dipoleBfield", InitializeGrid_with_Pulsar_Bfield);
        pp.query("init_corotatingEfield", InitializeGrid_with_Pulsar_Efield);
        enforceCorotatingE = 1;
        enforceDipoleB = 1;
        pp.query("enforceCorotatingE", enforceCorotatingE);
        pp.query("enforceDipoleB", enforceDipoleB);
        singleParticleTest = 0;
        pp.query("singleParticleTest", singleParticleTest);
        pp.query("DampBDipoleInRing", DampBDipoleInRing);
        if (DampBDipoleInRing == 1) pp.query("Bdamping_scale", Bdamping_scale);
        pp.query("injection_time", injection_time);
        pp.query("continuous_injection", continuous_injection);
        pp.query("removeparticle_theta_min",removeparticle_theta_min);
        pp.query("removeparticle_theta_max",removeparticle_theta_max);
        pp.query("use_theoreticalEB",use_theoreticalEB);
        amrex::Print() << "use theory EB " << use_theoreticalEB << "\n"; 
        pp.query("theory_max_rstar",theory_max_rstar);
        amrex::Print() << " theory max rstar : " << theory_max_rstar << "\n";
        pp.query("LimitDipoleBInit", LimitDipoleBInit);
        if (LimitDipoleBInit == 1) {
           pp.query("DipoleB_init_maxradius", DipoleB_init_maxradius);
        }
        pp.query("AddExternalMonopoleOnly", AddExternalMonopoleOnly);
        pp.query("EnforceTheoreticalEBInGrid", EnforceTheoreticalEBInGrid);
    }

    /** To initialize the grid with dipole magnetic field everywhere and corotating vacuum
     *  electric field inside the pulsar radius.
     */
    void InitializeExternalPulsarFieldsOnGrid ( amrex::MultiFab *mfx, amrex::MultiFab *mfy,
        amrex::MultiFab *mfz, const int lev, const bool init_Bfield)
    {
        auto & warpx = WarpX::GetInstance();
        const auto dx = warpx.Geom(lev).CellSizeArray();
        const auto problo = warpx.Geom(lev).ProbLoArray();
        const auto probhi = warpx.Geom(lev).ProbHiArray();
        const RealBox& real_box = warpx.Geom(lev).ProbDomain();
        amrex::Real cur_time = warpx.gett_new(lev);
        amrex::IntVect x_nodal_flag = mfx->ixType().toIntVect();
        amrex::IntVect y_nodal_flag = mfy->ixType().toIntVect();
        amrex::IntVect z_nodal_flag = mfz->ixType().toIntVect();
        GpuArray<int, 3> x_IndexType;
        GpuArray<int, 3> y_IndexType;
        GpuArray<int, 3> z_IndexType;
        for (int idim = 0; idim < 3; ++idim) {
            x_IndexType[idim] = x_nodal_flag[idim];
            y_IndexType[idim] = y_nodal_flag[idim];
            z_IndexType[idim] = z_nodal_flag[idim];
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& tbx = mfi.tilebox(x_nodal_flag, mfx->nGrowVect() );
            const amrex::Box& tby = mfi.tilebox(y_nodal_flag, mfx->nGrowVect() );
            const amrex::Box& tbz = mfi.tilebox(z_nodal_flag, mfx->nGrowVect() );
            amrex::Print() << " tbx : " << tbx << "\n";    
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    // Initialize with Bfield in spherical coordinates
                    if (init_Bfield == 1) {
                        ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                Fr, Ftheta, Fphi);
                    }
                    // Initialize corotating EField in r < corotating 
                    if (init_Bfield == 0) {
                        ExternalEFieldSpherical( r, theta, phi, cur_time,
                                                Fr, Ftheta, Fphi);
                       // if (r <= corotatingE_maxradius) {
                       //     CorotatingEfieldSpherical(r, theta, phi, cur_time,
                       //                               Fr, Ftheta, Fphi);
                       // }
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    // Initialize with Bfield in spherical coordinates                    
                    if (init_Bfield == 1) {
                        ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                Fr, Ftheta, Fphi);
                    }
                    // Initialize corotating Efield in r < corotating
                    if (init_Bfield == 0) {
                        ExternalEFieldSpherical( r, theta, phi, cur_time,
                                                Fr, Ftheta, Fphi);
                        //if (r <= corotatingE_maxradius) {
                        //    CorotatingEfieldSpherical(r, theta, phi, cur_time,
                        //                              Fr, Ftheta, Fphi);
                        //}
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    // Initialize with Bfield in spherical coordinates
                    if (init_Bfield == 1) {
                       ExternalBFieldSpherical( r, theta, phi, cur_time,
                                               Fr, Ftheta, Fphi);
                    }
                    // Initialize corotating Efield in r < corotating
                    if (init_Bfield == 0) {
                        ExternalEFieldSpherical( r, theta, phi, cur_time,
                                                Fr, Ftheta, Fphi);
                        //if (r <= corotatingE_maxradius) {
                        //    CorotatingEfieldSpherical(r, theta, phi, cur_time,
                        //                              Fr, Ftheta, Fphi);
                        //}
                    }
                    ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                        r, theta, phi, mfz_arr(i,j,k));
                }
            );
        } // mfiter loop
    } // InitializeExternalField


    void ApplyCorotatingEfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Efield,
                                    const int lev, const amrex::Real a_dt)
    {
        amrex::Print() << " applying corotating Efield BC\n";
        auto & warpx = WarpX::GetInstance();
        const auto dx = warpx.Geom(lev).CellSizeArray();
        const auto problo = warpx.Geom(lev).ProbLoArray();
        const auto probhi = warpx.Geom(lev).ProbHiArray();
        const RealBox& real_box = warpx.Geom(lev).ProbDomain();
        amrex::Real cur_time = warpx.gett_new(lev) + a_dt;
        amrex::IntVect x_nodal_flag = Efield[0]->ixType().toIntVect();
        amrex::IntVect y_nodal_flag = Efield[1]->ixType().toIntVect();
        amrex::IntVect z_nodal_flag = Efield[2]->ixType().toIntVect();
        GpuArray<int, 3> x_IndexType;
        GpuArray<int, 3> y_IndexType;
        GpuArray<int, 3> z_IndexType;
        for (int idim = 0; idim < 3; ++idim) {
            x_IndexType[idim] = x_nodal_flag[idim];
            y_IndexType[idim] = y_nodal_flag[idim];
            z_IndexType[idim] = z_nodal_flag[idim];
        }
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    if (PulsarParm::EnforceTheoreticalEBInGrid == 0) {
                        if (r <= corotatingE_maxradius) {
                            CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                      Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Ex_arr(i,j,k));
                        }
                    } else {
                        ExternalEFieldSpherical(r, theta, phi, cur_time, Fr, Ftheta, Fphi);
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    if (PulsarParm::EnforceTheoreticalEBInGrid == 0) {
                        if (r <= corotatingE_maxradius) {
                            CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                      Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Ey_arr(i,j,k));
                        }
                    } else {
                        ExternalEFieldSpherical(r, theta, phi, cur_time, Fr, Ftheta, Fphi);
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    if (PulsarParm::EnforceTheoreticalEBInGrid == 0) {
                        if (r <= corotatingE_maxradius) {
                            CorotatingEfieldSpherical(r, theta, phi, cur_time,
                                                      Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Ez_arr(i,j,k));
                        }
                    } else {
                        ExternalEFieldSpherical(r, theta, phi, cur_time, Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Ez_arr(i,j,k));
                    }
                }
            );
        }
        
        
    }

    void ApplyDipoleBfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
                                    const int lev, const amrex::Real a_dt)
    {
        amrex::Print() << " applying corotating Bfield BC\n";
        auto & warpx = WarpX::GetInstance();
        const auto dx = warpx.Geom(lev).CellSizeArray();
        const auto problo = warpx.Geom(lev).ProbLoArray();
        const auto probhi = warpx.Geom(lev).ProbHiArray();
        const RealBox& real_box = warpx.Geom(lev).ProbDomain();
        amrex::Real cur_time = warpx.gett_new(lev) + a_dt;
        amrex::IntVect x_nodal_flag = Bfield[0]->ixType().toIntVect();
        amrex::IntVect y_nodal_flag = Bfield[1]->ixType().toIntVect();
        amrex::IntVect z_nodal_flag = Bfield[2]->ixType().toIntVect();
        GpuArray<int, 3> x_IndexType;
        GpuArray<int, 3> y_IndexType;
        GpuArray<int, 3> z_IndexType;
        for (int idim = 0; idim < 3; ++idim) {
            x_IndexType[idim] = x_nodal_flag[idim];
            y_IndexType[idim] = y_nodal_flag[idim];
            z_IndexType[idim] = z_nodal_flag[idim];
        }
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    if (PulsarParm::EnforceTheoreticalEBInGrid == 0) {
                        if (r <= enforceDipoleB_maxradius) {
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                   Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Bx_arr(i,j,k));
                        }
                        else if ( r > enforceDipoleB_maxradius && r <= corotatingE_maxradius) {
                            if (DampBDipoleInRing == 1) {
                                // from inner ring to outer ring
                                ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                                               + std::tanh( Bdamping_scale
                                                          * (1._rt - r/enforceDipoleB_maxradius)
                                                          );
                                Bx_arr(i,j,k) += Fd * Bx_dipole;
                            }
                        }
                    } else {
                        ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    if (PulsarParm::EnforceTheoreticalEBInGrid == 0) {
                        if (r <= enforceDipoleB_maxradius) {
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                   Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, By_arr(i,j,k));
                        } else if ( r > enforceDipoleB_maxradius && r <= corotatingE_maxradius) {
                            if (DampBDipoleInRing == 1) {
                                // from inner ring to outer ring
                                ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                       Fr, Ftheta, Fphi);
                                amrex::Real By_dipole;
                                ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                                       r, theta, phi, By_dipole);
                                amrex::Real Fd = 1._rt
                                               + std::tanh( Bdamping_scale
                                                          * (1._rt - r/enforceDipoleB_maxradius)
                                                          );
                                By_arr(i,j,k) += Fd * By_dipole;
                            }
                        }
                    } else {
                        ExternalBFieldSpherical(r, theta, phi, cur_time,
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
                    ConvertCartesianToSphericalCoord(x, y, z, problo, probhi, r, theta, phi);
                    if (PulsarParm::EnforceTheoreticalEBInGrid == 0) {
                        if (r <= enforceDipoleB_maxradius) {
                            ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                   Fr, Ftheta, Fphi);
                            ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                                   r, theta, phi, Bz_arr(i,j,k));
                        } else if ( r > enforceDipoleB_maxradius && r <= corotatingE_maxradius) {
                            if (DampBDipoleInRing == 1) {
                                // from inner ring to outer ring
                                ExternalBFieldSpherical(r, theta, phi, cur_time,
                                                       Fr, Ftheta, Fphi);
                                amrex::Real Bz_dipole;
                                ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                                       r, theta, phi, Bz_dipole);
                                amrex::Real Fd = 1._rt
                                               + std::tanh( Bdamping_scale
                                                          * (1._rt - r/enforceDipoleB_maxradius)
                                                          );
                                Bz_arr(i,j,k) += Fd * Bz_dipole;
                            }
                        }
                    } else {
                        ExternalBFieldSpherical(r, theta, phi, cur_time,
                                               Fr, Ftheta, Fphi);
                        ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                               r, theta, phi, Bz_arr(i,j,k));
                    }
                }
            );
        }

    }
}
