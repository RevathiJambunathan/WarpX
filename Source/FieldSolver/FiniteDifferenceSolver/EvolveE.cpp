/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>
#ifdef PULSAR
#    include "WarpX.H"
#    include "Particles/PulsarParameters.H"
#endif


using namespace amrex;

/**
 * \brief Update the E field, over one timestep
 */
void FiniteDifferenceSolver::EvolveE (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    std::unique_ptr<amrex::MultiFab> const& Ffield,
    amrex::Real const dt ) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    if (m_fdtd_algo == MaxwellSolverAlgo::Yee){

        EvolveECylindrical <CylindricalYeeAlgorithm> ( Efield, Bfield, Jfield, Ffield, dt );

#else
    if (m_do_nodal) {

        EvolveECartesian <CartesianNodalAlgorithm> ( Efield, Bfield, Jfield, Ffield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveECartesian <CartesianYeeAlgorithm> ( Efield, Bfield, Jfield, Ffield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveECartesian <CartesianCKCAlgorithm> ( Efield, Bfield, Jfield, Ffield, dt );

#endif
    } else {
        amrex::Abort("EvolveE: Unknown algorithm");
    }

}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveECartesian (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    std::unique_ptr<amrex::MultiFab> const& Ffield,
    amrex::Real const dt ) {

    Real constexpr c2 = PhysConst::c * PhysConst::c;
#ifdef PULSAR
    auto &warpx = WarpX::GetInstance();
    amrex::IntVect ex_type = Efield[0]->ixType().toIntVect();
    amrex::IntVect ey_type = Efield[1]->ixType().toIntVect();
    amrex::IntVect ez_type = Efield[2]->ixType().toIntVect();
    const auto dx = warpx.Geom(0).CellSizeArray();
    const auto problo = warpx.Geom(0).ProbLoArray();
    const auto probhi = warpx.Geom(0).ProbHiArray();
    amrex::GpuArray<int, 3> Ex_stag;
    amrex::GpuArray<int, 3> Ey_stag;
    amrex::GpuArray<int, 3> Ez_stag;
    for (int idim = 0; idim < 3; ++idim) {
        Ex_stag[idim] = ex_type[idim];
        Ey_stag[idim] = ey_type[idim];
        Ez_stag[idim] = ez_type[idim];
    }
    amrex::Real cur_time = warpx.gett_new(0);
#endif

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& jx = Jfield[0]->array(mfi);
        Array4<Real> const& jy = Jfield[1]->array(mfi);
        Array4<Real> const& jz = Jfield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // Extract tileboxes for which to loop
        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ex(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                    - PhysConst::mu0 * jx(i, j, k) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ey(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                    - PhysConst::mu0 * jy(i, j, k) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ez(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k)
                    - PhysConst::mu0 * jz(i, j, k) );
            }

        );

        // If F is not a null pointer, further update E using the grad(F) term
        // (hyperbolic correction for errors in charge conservation)
        if (Ffield) {

            // Extract field data for this grid/tile
            Array4<Real> F = Ffield->array(mfi);

            // Loop over the cells and update the fields
            amrex::ParallelFor(tex, tey, tez,

                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ex(i, j, k) += c2 * dt * T_Algo::UpwardDx(F, coefs_x, n_coefs_x, i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ey(i, j, k) += c2 * dt * T_Algo::UpwardDy(F, coefs_y, n_coefs_y, i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ez(i, j, k) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, k);
                }

            );

        }

#ifdef PULSAR
        if (PulsarParm::set_corotatingEongrid == 1) {
            amrex::ParallelFor(tex, tey, tez,

                [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                    // get cell coordinates
                    amrex::Real x, y, z;
                    PulsarParm::ComputeCellCoordinates(i, j, k, Ex_stag, problo, dx,
                                                       x, y, z);
                    // convert (x,y,z) to (r, theta, phi);
                    amrex::Real r, theta, phi;
                    PulsarParm::ConvertCartesianToSphericalCoord(x, y, z, problo, probhi,
                                                                r, theta, phi);
                    if (r <= PulsarParm::max_corotatingEongrid_radius) {
                        // Compute corotating E
                        amrex::Real Er, Etheta, Ephi;
                        PulsarParm::ExternalEFieldSpherical(r, theta, phi, cur_time,
                                                              Er, Etheta, Ephi);
                        amrex::Real Ex_loc;
                        PulsarParm::ConvertSphericalToCartesianXComponent( Er, Etheta, Ephi,
                                                                      r, theta, phi, Ex_loc);
                        Ex(i, j, k) += Ex_loc;
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                    // get cell coordinates
                    amrex::Real x, y, z;
                    PulsarParm::ComputeCellCoordinates(i, j, k, Ey_stag, problo, dx,
                                                       x, y, z);
                    // convert (x,y,z) to (r, theta, phi);
                    amrex::Real r, theta, phi;
                    PulsarParm::ConvertCartesianToSphericalCoord(x, y, z, problo, probhi,
                                                                r, theta, phi);
                    if (r <= PulsarParm::max_corotatingEongrid_radius) {
                        // Compute corotating E                      
                        amrex::Real Er, Etheta, Ephi;
                        PulsarParm::ExternalEFieldSpherical(r, theta, phi, cur_time,
                                                              Er, Etheta, Ephi);
                        amrex::Real Ey_loc;
                        PulsarParm::ConvertSphericalToCartesianYComponent( Er, Etheta, Ephi,
                                                                      r, theta, phi, Ey_loc);
                        Ey(i, j, k) += Ey_loc;
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                    // get cell coordinates
                    amrex::Real x, y, z;
                    PulsarParm::ComputeCellCoordinates(i, j, k, Ez_stag, problo, dx,
                                                       x, y, z);
                    // convert (x,y,z) to (r, theta, phi);
                    amrex::Real r, theta, phi;
                    PulsarParm::ConvertCartesianToSphericalCoord(x, y, z, problo, probhi,
                                                                r, theta, phi);
                    if (r <= PulsarParm::max_corotatingEongrid_radius) {
                        // Compute corotating E                      
                        amrex::Real Er, Etheta, Ephi;
                        PulsarParm::ExternalEFieldSpherical(r, theta, phi, cur_time,
                                                              Er, Etheta, Ephi);
                        amrex::Real Ez_loc;
                        PulsarParm::ConvertSphericalToCartesianZComponent( Er, Etheta, Ephi,
                                                                      r, theta, phi, Ez_loc);
                        Ez(i, j, k) += Ez_loc;
                    }
                }
            );    
        }
#endif
    }

}

#else // corresponds to ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveECylindrical (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    std::unique_ptr<amrex::MultiFab> const& Ffield,
    amrex::Real const dt ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Bt = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& jr = Jfield[0]->array(mfi);
        Array4<Real> const& jt = Jfield[1]->array(mfi);
        Array4<Real> const& jz = Jfield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = m_stencil_coefs_r.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        int const nmodes = m_nmodes;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop
        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        Real const c2 = PhysConst::c * PhysConst::c;

        // Loop over the cells and update the fields
        amrex::ParallelFor(ter, tet, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + (i + 0.5)*dr; // r on cell-centered point (Er is cell-centered in r)
                Er(i, j, 0, 0) +=  c2 * dt*(
                    - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 0)
                    - PhysConst::mu0 * jr(i, j, 0, 0) ); // Mode m=0
                for (int m=1; m<nmodes; m++) { // Higher-order modes
                    Er(i, j, 0, 2*m-1) += c2 * dt*(
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        + m * Bz(i, j, 0, 2*m  )/r
                        - PhysConst::mu0 * jr(i, j, 0, 2*m-1) );  // Real part
                    Er(i, j, 0, 2*m  ) += c2 * dt*(
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - m * Bz(i, j, 0, 2*m-1)/r
                        - PhysConst::mu0 * jr(i, j, 0, 2*m  ) ); // Imaginary part
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + i*dr; // r on a nodal grid (Et is nodal in r)
                if (r != 0) { // Off-axis, regular Maxwell equations
                    Et(i, j, 0, 0) += c2 * dt*(
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 0)
                        - PhysConst::mu0 * jt(i, j, 0, 0 ) ); // Mode m=0
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Et(i, j, 0, 2*m-1) += c2 * dt*(
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                            - PhysConst::mu0 * jt(i, j, 0, 2*m-1) ); // Real part
                        Et(i, j, 0, 2*m  ) += c2 * dt*(
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                            - PhysConst::mu0 * jt(i, j, 0, 2*m  ) ); // Imaginary part
                    }
                } else { // r==0: on-axis corrections
                    // Ensure that Et remains 0 on axis (except for m=1)
                    Et(i, j, 0, 0) = 0.; // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        if (m == 1){
                            // The bulk equation could in principle be used here since it does not diverge
                            // on axis. However, it typically gives poor results e.g. for the propagation
                            // of a laser pulse (the field is spuriously reduced on axis). For this reason
                            // a modified on-axis condition is used here: we use the fact that
                            // Etheta(r=0,m=1) should equal -iEr(r=0,m=1), for the fields Er and Et to be
                            // independent of theta at r=0. Now with linear interpolation:
                            // Er(r=0,m=1) = 0.5*[Er(r=dr/2,m=1) + Er(r=-dr/2,m=1)]
                            // And using the rule applying for the guards cells
                            // Er(r=-dr/2,m=1) = Er(r=dr/2,m=1). Thus: Et(i,j,m) = -i*Er(i,j,m)
                            Et(i,j,0,2*m-1) =  Er(i,j,0,2*m  );
                            Et(i,j,0,2*m  ) = -Er(i,j,0,2*m-1);
                        } else {
                            Et(i, j, 0, 2*m-1) = 0.;
                            Et(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + i*dr; // r on a nodal grid (Ez is nodal in r)
                if (r != 0) { // Off-axis, regular Maxwell equations
                    Ez(i, j, 0, 0) += c2 * dt*(
                       T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                        - PhysConst::mu0 * jz(i, j, 0, 0  ) ); // Mode m=0
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Ez(i, j, 0, 2*m-1) += c2 * dt *(
                            - m * Br(i, j, 0, 2*m  )/r
                            + T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            - PhysConst::mu0 * jz(i, j, 0, 2*m-1) ); // Real part
                        Ez(i, j, 0, 2*m  ) += c2 * dt *(
                            m * Br(i, j, 0, 2*m-1)/r
                            + T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            - PhysConst::mu0 * jz(i, j, 0, 2*m  ) ); // Imaginary part
                    }
                } else { // r==0: on-axis corrections
                    // For m==0, Bt is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Ez(i, j, 0, 0) += c2 * dt*(
                         4*Bt(i, j, 0, 0)/dr // regularization
                         - PhysConst::mu0 * jz(i, j, 0, 0  ) );
                    // Ensure that Ez remains 0 for higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        Ez(i, j, 0, 2*m-1) = 0.;
                        Ez(i, j, 0, 2*m  ) = 0.;
                    }
                }
            }

        ); // end of loop over cells

        // If F is not a null pointer, further update E using the grad(F) term
        // (hyperbolic correction for errors in charge conservation)
        if (Ffield) {

            // Extract field data for this grid/tile
            Array4<Real> F = Ffield->array(mfi);

            // Loop over the cells and update the fields
            amrex::ParallelFor(ter, tet, tez,

                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    Er(i, j, 0, 0) += c2 * dt * T_Algo::UpwardDr(F, coefs_r, n_coefs_r, i, j, 0, 0);
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        Er(i, j, 0, 2*m-1) += c2 * dt * T_Algo::UpwardDr(F, coefs_r, n_coefs_r, i, j, 0, 2*m-1); // Real part
                        Er(i, j, 0, 2*m  ) += c2 * dt * T_Algo::UpwardDr(F, coefs_r, n_coefs_r, i, j, 0, 2*m  ); // Imaginary part
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    // Mode m=0: no update
                    Real const r = rmin + i*dr; // r on a nodal grid (Et is nodal in r)
                    if (r != 0){ // Off-axis, regular Maxwell equations
                        for (int m=1; m<nmodes; m++) { // Higher-order modes
                            Et(i, j, 0, 2*m-1) += c2 * dt *  m * F(i, j, 0, 2*m  )/r; // Real part
                            Et(i, j, 0, 2*m  ) += c2 * dt * -m * F(i, j, 0, 2*m-1)/r; // Imaginary part
                        }
                    } else { // r==0: on-axis corrections
                        // For m==1, F is linear in r, for small r
                        // Therefore, the formula below regularizes the singularity
                        if (nmodes >= 2) { // needs to have at least m=0 and m=1
                            int const m=1;
                            Et(i, j, 0, 2*m-1) += c2 * dt *  m * F(i+1, j, 0, 2*m  )/dr; // Real part
                            Et(i, j, 0, 2*m  ) += c2 * dt * -m * F(i+1, j, 0, 2*m-1)/dr; // Imaginary part
                        }
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    Ez(i, j, 0, 0) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, 0, 0);
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        Ez(i, j, 0, 2*m-1) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, 0, 2*m-1); // Real part
                        Ez(i, j, 0, 2*m  ) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, 0, 2*m  ); // Imaginary part
                    }
                }

            ); // end of loop over cells

        } // end of if condition for F

    } // end of loop over grid/tiles

}

#endif // corresponds to ifndef WARPX_DIM_RZ
