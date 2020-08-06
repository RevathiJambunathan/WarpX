#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
    // currently works only for 3D
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#endif
#include "Utils/WarpXConst.H"
#include "Utils/CoarsenIO.H"
#include "Utils/CoarsenMR.H"
#include <AMReX_Gpu.H>
#include <WarpX.H>

using namespace amrex;

void FiniteDifferenceSolver::MacroscopicEvolveE (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    amrex::Real const dt, std::unique_ptr<MacroscopicProperties> const& macroscopic_properties ) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    amrex::Abort("currently macro E-push does not work for RZ");
#else
    if (m_do_nodal) {
        amrex::Abort(" macro E-push does not work for nodal ");

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {

            MacroscopicEvolveECartesian <CartesianYeeAlgorithm, LaxWendroffAlgo>
                       ( Efield, Bfield, Jfield, dt, macroscopic_properties );

        }
        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {

            MacroscopicEvolveECartesian <CartesianYeeAlgorithm, BackwardEulerAlgo>
                       ( Efield, Bfield, Jfield, dt, macroscopic_properties );

        }

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        // Note : EvolveE is the same for CKC and Yee.
        // In the templated Yee and CKC calls, the core operations for EvolveE is tihe same.
        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {

            MacroscopicEvolveECartesian <CartesianCKCAlgorithm, LaxWendroffAlgo>
                       ( Efield, Bfield, Jfield, dt, macroscopic_properties );

        } else if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {

            MacroscopicEvolveECartesian <CartesianCKCAlgorithm, BackwardEulerAlgo>
                       ( Efield, Bfield, Jfield, dt, macroscopic_properties );

        }

    } else {
        amrex::Abort("Unknown algorithm");
    }
#endif

}


#ifndef WARPX_DIM_RZ

template<typename T_Algo, typename T_MacroAlgo>
void FiniteDifferenceSolver::MacroscopicEvolveECartesian (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    amrex::Real const dt, std::unique_ptr<MacroscopicProperties> const& macroscopic_properties ) {

    auto& sigma_mf = macroscopic_properties->getsigma_mf();
    auto& epsilon_mf = macroscopic_properties->getepsilon_mf();
    auto& mu_mf = macroscopic_properties->getmu_mf();

    // Index type required for calling CoarsenIO::Interp to interpolate macroscopic 
    // properties from their respective staggering to the Ex, Ey, Ez locations
    int const* const AMREX_RESTRICT
              sigma_stag   = macroscopic_properties->sigma_IndexType.data();
    int const* const AMREX_RESTRICT
              epsilon_stag = macroscopic_properties->epsilon_IndexType.data();
    int const* const AMREX_RESTRICT
              mu_stag      = macroscopic_properties->mu_IndexType.data();
    int const* const AMREX_RESTRICT
              Ex_stag      = macroscopic_properties->Ex_IndexType.data();
    int const* const AMREX_RESTRICT
              Ey_stag      = macroscopic_properties->Ey_IndexType.data();
    int const* const AMREX_RESTRICT
              Ez_stag      = macroscopic_properties->Ez_IndexType.data();
    int const* const AMREX_RESTRICT
              macro_cr     = macroscopic_properties->macro_cr_ratio.data();

    
    // Loop through the grids, and over the tiles within each grid
#ifdef _OPENMP
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

        // material prop //
        Array4<Real> const& sigma_arr = sigma_mf.array(mfi);
        Array4<Real> const& eps_arr = epsilon_mf.array(mfi);
        Array4<Real> const& mu_arr = mu_mf.array(mfi);

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

        // starting component to interpolate macro properties to Ex, Ey, Ez locations
        const int scomp = 0;
        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Interpolate conductivity, sigma, to Ex position on the grid
                amrex::Real const sigma_interp = CoarsenIO::Interp( sigma_arr, sigma_stag,
                                           Ex_stag, macro_cr, i, j, k, scomp);
                // Interpolated permittivity, epsilon, to Ex position on the grid
                amrex::Real const epsilon_interp = CoarsenIO::Interp( eps_arr, epsilon_stag, 
                                           Ex_stag, macro_cr, i, j, k, scomp);
                // Interpolated permeability, mu, to Ex position on the grid
                amrex::Real const mu = CoarsenIO::Interp( mu_arr, mu_stag,
                                           Ex_stag, macro_cr, i, j, k, scomp);
                amrex::Real alpha = T_MacroAlgo::alpha( sigma_interp, epsilon_interp, dt);
                amrex::Real beta = T_MacroAlgo::beta( sigma_interp, epsilon_interp, dt);

                Ex(i, j, k) = alpha * Ex(i, j, k) + (beta/mu)
                     * ( - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                         + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k))
                           - beta * jx(i, j, k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const sigma_interp = CoarsenIO::Interp( sigma_arr, sigma_stag,
                                           Ey_stag, macro_cr, i, j, k, scomp);
                amrex::Real const epsilon_interp = CoarsenIO::Interp( eps_arr, epsilon_stag, 
                                           Ey_stag, macro_cr, i, j, k, scomp);
                amrex::Real const mu = CoarsenIO::Interp( mu_arr, mu_stag,
                                           Ey_stag, macro_cr, i, j, k, scomp);
                amrex::Real alpha = T_MacroAlgo::alpha( sigma_interp, epsilon_interp, dt);
                amrex::Real beta = T_MacroAlgo::beta( sigma_interp, epsilon_interp, dt);

                Ey(i, j, k) = alpha * Ey(i, j, k) + (beta/mu)
                     * ( - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                         + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k))
                           - beta * jy(i, j, k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const sigma_interp = CoarsenIO::Interp( sigma_arr, sigma_stag,
                                           Ez_stag, macro_cr, i, j, k, scomp);
                amrex::Real const epsilon_interp = CoarsenIO::Interp( eps_arr, epsilon_stag, 
                                           Ez_stag, macro_cr, i, j, k, scomp);
                amrex::Real const mu = CoarsenIO::Interp( mu_arr, mu_stag,
                                           Ez_stag, macro_cr, i, j, k, scomp);
                amrex::Real alpha = T_MacroAlgo::alpha( sigma_interp, epsilon_interp, dt);
                amrex::Real beta = T_MacroAlgo::beta( sigma_interp, epsilon_interp, dt);

                Ez(i, j, k) = alpha * Ez(i, j, k) + (beta/mu)
                     * ( - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                         + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k))
                             - beta * jz(i, j, k);
            }

        );
    }
}

#endif // corresponds to ifndef WARPX_DIM_RZ
