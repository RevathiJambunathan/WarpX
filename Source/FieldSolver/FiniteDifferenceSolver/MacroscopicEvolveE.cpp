#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
    // currently works only for 3D
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#endif
#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>
#include <WarpX.H>

using namespace amrex;

void FiniteDifferenceSolver::MacroscopicEvolveE (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
#ifdef WARPX_MAG_LLG
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Mfield,
#endif
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
                       ( Efield, Bfield,
#ifdef WARPX_MAG_LLG
                         Mfield,
#endif
                         Jfield, dt, macroscopic_properties );
        }
        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {

            MacroscopicEvolveECartesian <CartesianYeeAlgorithm, BackwardEulerAlgo>
                       ( Efield, Bfield,
#ifdef WARPX_MAG_LLG
                         Mfield,
#endif
                         Jfield, dt, macroscopic_properties );

        }

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        // Note : EvolveE is the same for CKC and Yee.
        // In the templated Yee and CKC calls, the core operations for EvolveE is tihe same.
        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {

            MacroscopicEvolveECartesian <CartesianCKCAlgorithm, LaxWendroffAlgo>
                       ( Efield, Bfield,
#ifdef WARPX_MAG_LLG
                         Mfield,
#endif
                         Jfield, dt, macroscopic_properties );
        } else if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {

            MacroscopicEvolveECartesian <CartesianCKCAlgorithm, BackwardEulerAlgo>
                       ( Efield, Bfield,
#ifdef WARPX_MAG_LLG
                         Mfield,
#endif
                         Jfield, dt, macroscopic_properties );

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
#ifdef WARPX_MAG_LLG
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Mfield,
#endif
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    amrex::Real const dt, std::unique_ptr<MacroscopicProperties> const& macroscopic_properties ) {

	amrex::Print() << "macroscopic_evolvee " << std::endl;
	      amrex::Print() << std::endl;
	
    //const int &macroscopic_solver_algo = WarpX::macroscopic_solver_algo;
    //Real sigma = macroscopic_properties->sigma();
    //Real const mu = macroscopic_properties->mu();
    //Real const epsilon = macroscopic_properties->epsilon();

    auto& sigma_mf = macroscopic_properties->getsigma_mf();
    auto& epsilon_mf = macroscopic_properties->getepsilon_mf();
    auto& mu_mf = macroscopic_properties->getmu_mf();

    //Real alpha = 0._rt;
    //Real beta  = 0._rt;
    //Real fac1 = 0._rt;
    //Real inv_fac = 0._rt;
    //if (macroscopic_solver_algo == 0) {
    //    // sigma_method == 0 for Lax_Wendroff or semi-implicit approach
    //    fac1 = 0.5_rt * sigma * dt / epsilon;
    //    inv_fac = 1._rt / ( 1._rt + fac1);
    //    alpha = (1.0_rt - fac1) * inv_fac;
    //    beta  = dt * inv_fac / epsilon;
    //}
    //else if (macroscopic_solver_algo == 1) { // sigma method == 1
    //    // sigma_metha == 1 for Backward Euler
    //    fac1 = sigma * dt / epsilon;
    //    inv_fac = 1._rt / ( 1._rt + fac1);
    //    alpha = inv_fac;
    //    beta  = dt * inv_fac / epsilon;
    //}

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

#ifdef WARPX_MAG_LLG
        Array4<Real> const& M_xface = Mfield[0]->array(mfi); // note M_xface include x,y,z components at |_x faces
        Array4<Real> const& M_yface = Mfield[1]->array(mfi); // note M_yface include x,y,z components at |_y faces
        Array4<Real> const& M_zface = Mfield[2]->array(mfi); // note M_zface include x,y,z components at |_z faces
#endif

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


        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real alpha = T_MacroAlgo::alpha( sigma_arr, eps_arr, dt,
                                                 i, j, k, amrex::IntVect(1,0,0) );
                amrex::Real beta = T_MacroAlgo::beta(sigma_arr, eps_arr, dt,
                                                i, j, k, amrex::IntVect(1,0,0) );
                amrex::Real mu = T_MacroAlgo::macro_avg_to_edge(i, j, k, amrex::IntVect(2,1,1),
                                                    mu_arr);
#ifdef WARPX_MAG_LLG
                Ex(i, j, k) = alpha * Ex(i, j, k) + beta
                     * ((- T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k, 0)
                         + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k, 0))/PhysConst::mu0
                         + T_Algo::DownwardDz(M_yface, coefs_z, n_coefs_z, i, j, k, 1)
                         - T_Algo::DownwardDy(M_zface, coefs_y, n_coefs_y, i, j, k, 2));
#else
                Ex(i, j, k) = alpha * Ex(i, j, k) + (beta/mu)
                     * ( - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                         + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k));
#endif
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real alpha = T_MacroAlgo::alpha( sigma_arr, eps_arr, dt,
                                                 i, j, k, amrex::IntVect(0,1,0) );
                amrex::Real beta = T_MacroAlgo::beta(sigma_arr, eps_arr, dt,
                                                i, j, k, amrex::IntVect(0,1,0) );
                amrex::Real mu = T_MacroAlgo::macro_avg_to_edge(i, j, k, amrex::IntVect(1,2,1),
                                                    mu_arr);
#ifdef WARPX_MAG_LLG

              if( (i==0 && j==0 && k==0) || (i==1 && j==0 && k==0 ) ){
              amrex::Print() << "i,j,k " << i << " " << j << " " << k << std::endl;
	      amrex::Print() << "M_zface(i, j, k, 0) " << M_zface(i, j, k, 0) << std::endl;
	      amrex::Print() << "M_zface(i, j, k, 2) " << M_zface(i, j, k, 2) << std::endl;
	      amrex::Print() << "M_zface(i-1, j, k, 2) " << M_zface(i-1, j, k, 2) << std::endl;
	      amrex::Print() << "M_zface(i+7, j, k, 2) " << M_zface(i+7, j, k, 2) << std::endl;
	      amrex::Print() << std::endl;
	      }

		Ey(i, j, k) = alpha * Ey(i, j, k) + beta
                     * ((- T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k, 0)
                         + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k, 0))/PhysConst::mu0
                         + T_Algo::DownwardDx(M_zface, coefs_x, n_coefs_x, i, j, k, 2)
                         - T_Algo::DownwardDz(M_xface, coefs_z, n_coefs_z, i, j, k, 0));

              if( (i==0 && j==0 && k==0) || (i==1 && j==0 && k==0 ) ){
              amrex::Print() << "i,j,k " << i << " " << j << " " << k << std::endl;
//              amrex::Print() << "T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k, 0) " << T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k, 0) << std::endl;
//              amrex::Print() << "T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k, 0) " << T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k, 0) << std::endl;
              amrex::Print() << "T_Algo::DownwardDx(M_zface, coefs_x, n_coefs_x, i, j, k, 2) " << T_Algo::DownwardDx(M_zface, coefs_x, n_coefs_x, i, j, k, 2) << std::endl;
//              amrex::Print() << "T_Algo::DownwardDz(M_xface, coefs_z, n_coefs_z, i, j, k, 0) " << T_Algo::DownwardDz(M_xface, coefs_z, n_coefs_z, i, j, k, 0) << std::endl;
	      amrex::Print() << "Ey(i, j, k) " << Ey(i, j, k) << std::endl;
	      amrex::Print() << std::endl;
	      }
 
#else
                Ey(i, j, k) = alpha * Ey(i, j, k) + (beta/mu)
                     * ( - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                         + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k));
#endif
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real alpha = T_MacroAlgo::alpha( sigma_arr, eps_arr, dt,
                                                 i, j, k, amrex::IntVect(0,0,1) );
                amrex::Real beta = T_MacroAlgo::beta(sigma_arr, eps_arr, dt,
                                                i, j, k, amrex::IntVect(0,0,1) );
                amrex::Real mu = T_MacroAlgo::macro_avg_to_edge(i, j, k, amrex::IntVect(1,1,2),
                                                    mu_arr);
#ifdef WARPX_MAG_LLG
                Ez(i, j, k) = alpha * Ez(i, j, k) + beta
                     * ((- T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k, 0)
                         + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k, 0))/PhysConst::mu0
                         + T_Algo::DownwardDy(M_xface, coefs_y, n_coefs_y, i, j, k, 0)
                         - T_Algo::DownwardDx(M_yface, coefs_x, n_coefs_x, i, j, k, 1));
#else
                Ez(i, j, k) = alpha * Ez(i, j, k) + (beta/mu)
                     * ( - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                         + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k));
#endif
            }

        );

        // update E using J, if source currents are specified.
        if (Jfield[0]) {
            Array4<Real> const& jx = Jfield[0]->array(mfi);
            Array4<Real> const& jy = Jfield[1]->array(mfi);
            Array4<Real> const& jz = Jfield[2]->array(mfi);

            amrex::ParallelFor(tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Ex(i, j, k) += -T_MacroAlgo::beta(sigma_arr, eps_arr, dt,
                                                      i, j, k, amrex::IntVect(0, 0, 1) )
                                   * jx(i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Ey(i, j, k) += -T_MacroAlgo::beta(sigma_arr, eps_arr, dt,
                                                      i, j, k, amrex::IntVect(0, 1, 0) )
                                   * jy(i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Ez(i, j, k) += -T_MacroAlgo::beta(sigma_arr, eps_arr, dt,
                                                      i, j, k, amrex::IntVect(0, 0, 1)  )
                                    * jz(i, j, k);
                }
            );
        }
    }
}

#endif // corresponds to ifndef WARPX_DIM_RZ
