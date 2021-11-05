/* Copyright 2019-2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Utils/WarpXUtil.H"
#include "Initialization/ReconnectionPerturbation.H"
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cmath>


using namespace amrex;

void
Reconnection_Perturbation::AddBfieldPerturbation (amrex::MultiFab *Bx,
                              amrex::MultiFab *By,
                              amrex::MultiFab *Bz,
                              amrex::ParserExecutor<3> const& xfield_parser,
                              amrex::ParserExecutor<3> const& yfield_parser, 
                              amrex::ParserExecutor<3> const& zfield_parser, const int lev)
{
    auto &warpx = WarpX::GetInstance();
    const auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect x_nodal_flag = Bx->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = By->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = Bz->ixType().toIntVect();

    const auto problo = warpx.Geom(0).ProbLoArray();
    const auto probhi = warpx.Geom(0).ProbHiArray();
    amrex::Real pi_val = MathConst::pi;
    amrex::Real Lx = (real_box.hi(0) - real_box.lo(0) ) / 2.0;
#if (AMREX_SPACEDIM==2)
    amrex::Real Lz = ( real_box.hi(1) - real_box.lo(1) ) /2.0 ;
#else
    amrex::Real Lz = ( real_box.hi(2) - real_box.lo(2) ) /2.0;
#endif
    amrex::Print() << " Lx " << Lx << " " << Lz << "\n";
    amrex::Real xcs, B0, nd_ratio, delta;
    ParmParse pp_warpx("warpx");
    pp_warpx.get("xcs", xcs);
    pp_warpx.get("B0", B0);
    pp_warpx.get("nd_ratio", nd_ratio);
    pp_warpx.get("delta", delta);

    for ( MFIter mfi(*Bx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        auto const& Bx_array = Bx->array(mfi);
        auto const& By_array = By->array(mfi);
        auto const& Bz_array = Bz->array(mfi);

        const amrex::Box& tbx = mfi.tilebox( x_nodal_flag, Bx->nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox( z_nodal_flag, Bz->nGrowVect() );
        // Compute perturbation and add to Bx
        amrex::Print() << " tbx : " << tbx << " " << tbz << "\n";
        amrex::LoopOnCpu( tbx, [=] (int i, int j, int k)
        {
            amrex::Real fac_x = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
            amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2) 
            amrex::Real y = 0._rt;
            amrex::Real fac_z = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
            amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#else
            amrex::Real fac_y = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
            amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
            amrex::Real fac_z = (1._rt - x_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
            amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif

            amrex::Real magnitude_fac = -0.01_rt;
            amrex::Real prefactor = (pi_val / Lz) * std::cos(pi_val/Lz * z)
                                  * std::cos(pi_val/Lx * (x-xcs))
                                  * std::cos(pi_val/Lx * (x-xcs));
            amrex::Real IntegralBz_val = Reconnection_Perturbation::IntegralBz(
                                         x, z, pi_val, xcs, B0, nd_ratio, delta);
            Bx_array(i,j,k) += magnitude_fac * prefactor * IntegralBz_val;

        });
        // Compute perturbation and add to Bz
        amrex::LoopOnCpu( tbz, [=] (int i, int j, int k)
        {
            amrex::Real fac_x = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
            amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
            amrex::Real y = 0._rt;
            amrex::Real fac_z = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
            amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif (AMREX_SPACEDIM==3)
            amrex::Real fac_y = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
            amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
            amrex::Real fac_z = (1._rt - z_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
            amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
            amrex::Real magnitude_fac = 0.01_rt;
            amrex::Real prefactor_term1 = (-pi_val / Lx) * std::sin(pi_val/Lz * z)
                                        * std::sin(2.*pi_val/Lx * (x-xcs));
            amrex::Real prefactor_term2 = std::sin(pi_val/Lz * z)
                                        * std::cos(pi_val/Lx * (x-xcs))
                                        * std::cos(pi_val/Lx * (x-xcs)) ;
            amrex::Real IntegralBz_val = Reconnection_Perturbation::IntegralBz(
                                         x, z, pi_val, xcs, B0, nd_ratio, delta);
            Bz_array(i,j,k) += magnitude_fac * ( prefactor_term1 * IntegralBz_val
                                               + prefactor_term2 * zfield_parser(x,y,z)
                                               );
        });
    }
}

