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
#ifdef PULSAR
#    include "Particles/PulsarParameters.H"
#    include "WarpX.H"
#endif
#include <AMReX_Gpu.H>

using namespace amrex;

/**
 * \brief Update the B field, over one timestep
 */
void FiniteDifferenceSolver::EvolveB (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Efield,
    amrex::Real const dt ) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    if (m_fdtd_algo == MaxwellSolverAlgo::Yee){

        EvolveBCylindrical <CylindricalYeeAlgorithm> ( Bfield, Efield, dt );

#else
    if (m_do_nodal) {

        EvolveBCartesian <CartesianNodalAlgorithm> ( Bfield, Efield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveBCartesian <CartesianYeeAlgorithm> ( Bfield, Efield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveBCartesian <CartesianCKCAlgorithm> ( Bfield, Efield, dt );

#endif
    } else {
        amrex::Abort("EvolveB: Unknown algorithm");
    }

}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveBCartesian (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Efield,
    amrex::Real const dt ) {

#ifdef PULSAR
    auto & warpx = WarpX::GetInstance();
    amrex::IntVect bx_type = Bfield[0]->ixType().toIntVect();
    amrex::IntVect by_type = Bfield[1]->ixType().toIntVect();
    amrex::IntVect bz_type = Bfield[2]->ixType().toIntVect();
    const auto dx = warpx.Geom(0).CellSizeArray();
    const auto problo = warpx.Geom(0).ProbLoArray();
    const auto probhi = warpx.Geom(0).ProbHiArray();
    amrex::GpuArray<int, 3> Bx_stag;
    amrex::GpuArray<int, 3> By_stag;
    amrex::GpuArray<int, 3> Bz_stag;
    for (int idim = 0; idim < 3; ++idim) {
        Bx_stag[idim] = bx_type[idim];
        By_stag[idim] = by_type[idim];
        Bz_stag[idim] = bz_type[idim];
    }
    int turnOffMaxwell = PulsarParm::turnOffMaxwell;
    amrex::Real max_turnOffMaxwell_radius = PulsarParm::max_turnOffMaxwell_radius;
#endif

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tby  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef PULSAR
                int updateBx = 1;
                if (turnOffMaxwell == 1) {
                    amrex::Real x, y, z;
                    PulsarParm::ComputeCellCoordinates(i, j, k, Bx_stag, problo, dx,
                                                       x, y, z);
                    amrex::Real r, theta, phi;
                    PulsarParm::ConvertCartesianToSphericalCoord(x, y, z, problo, probhi,
                                                                 r, theta, phi);
                    if (r < max_turnOffMaxwell_radius) updateBx = 0;
                }
                if (updateBx == 1) {
                    Bx(i, j, k) += dt * T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                 - dt * T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k);
                }
#else
                Bx(i, j, k) += dt * T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                             - dt * T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k);
#endif
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef PULSAR
                int updateBy = 1;
                if (turnOffMaxwell == 1) {
                    amrex::Real x, y, z;
                    PulsarParm::ComputeCellCoordinates(i, j, k, By_stag, problo, dx,
                                                       x, y, z);
                    amrex::Real r, theta, phi;
                    PulsarParm::ConvertCartesianToSphericalCoord(x, y, z, problo, probhi,
                                                                 r, theta, phi);
                    if (r < max_turnOffMaxwell_radius) updateBy = 0;
                }
                if (updateBy == 1) {
                    By(i, j, k) += dt * T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                 - dt * T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k);
                }
#else
                By(i, j, k) += dt * T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                             - dt * T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k);
#endif
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef PULSAR
                int updateBz = 1;
                if (turnOffMaxwell == 1) {
                    amrex::Real x, y, z;
                    PulsarParm::ComputeCellCoordinates(i, j, k, Bz_stag, problo, dx,
                                                       x, y, z);
                    amrex::Real r, theta, phi;
                    PulsarParm::ConvertCartesianToSphericalCoord(x, y, z, problo, probhi,
                                                                 r, theta, phi);
                    if (r < max_turnOffMaxwell_radius) updateBz = 0;
                }
                if (updateBz == 1) {
                    Bz(i, j, k) += dt * T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                 - dt * T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k);
                }
#else
                Bz(i, j, k) += dt * T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                             - dt * T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k);
#endif
            }

        );

    }

}

#else // corresponds to ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveBCylindrical (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Efield,
    amrex::Real const dt ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Bt = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);

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
        Box const& tbr  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tbt  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbr, tbt, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + i*dr; // r on nodal point (Br is nodal in r)
                if (r != 0) { // Off-axis, regular Maxwell equations
                    Br(i, j, 0, 0) += dt * T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 0); // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        Br(i, j, 0, 2*m-1) += dt*(
                            T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                            - m * Ez(i, j, 0, 2*m  )/r );  // Real part
                        Br(i, j, 0, 2*m  ) += dt*(
                            T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                            + m * Ez(i, j, 0, 2*m-1)/r ); // Imaginary part
                    }
                } else { // r==0: On-axis corrections
                    // Ensure that Br remains 0 on axis (except for m=1)
                    Br(i, j, 0, 0) = 0.; // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        if (m == 1){
                            // For m==1, Ez is linear in r, for small r
                            // Therefore, the formula below regularizes the singularity
                            Br(i, j, 0, 2*m-1) += dt*(
                                T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                                - m * Ez(i+1, j, 0, 2*m  )/dr );  // Real part
                            Br(i, j, 0, 2*m  ) += dt*(
                                T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                                + m * Ez(i+1, j, 0, 2*m-1)/dr ); // Imaginary part
                        } else {
                            Br(i, j, 0, 2*m-1) = 0.;
                            Br(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Bt(i, j, 0, 0) += dt*(
                    T_Algo::UpwardDr(Ez, coefs_r, n_coefs_r, i, j, 0, 0)
                    - T_Algo::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 0)); // Mode m=0
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Bt(i, j, 0, 2*m-1) += dt*(
                        T_Algo::UpwardDr(Ez, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        - T_Algo::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 2*m-1)); // Real part
                    Bt(i, j, 0, 2*m  ) += dt*(
                        T_Algo::UpwardDr(Ez, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        - T_Algo::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 2*m  )); // Imaginary part
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + (i + 0.5)*dr; // r on a cell-centered grid (Bz is cell-centered in r)
                Bz(i, j, 0, 0) += dt*( - T_Algo::UpwardDrr_over_r(Et, r, dr, coefs_r, n_coefs_r, i, j, 0, 0));
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Bz(i, j, 0, 2*m-1) += dt*( m * Er(i, j, 0, 2*m  )/r
                        - T_Algo::UpwardDrr_over_r(Et, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)); // Real part
                    Bz(i, j, 0, 2*m  ) += dt*(-m * Er(i, j, 0, 2*m-1)/r
                        - T_Algo::UpwardDrr_over_r(Et, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )); // Imaginary part
                }
            }

        );

    }

}

#endif // corresponds to ifndef WARPX_DIM_RZ
