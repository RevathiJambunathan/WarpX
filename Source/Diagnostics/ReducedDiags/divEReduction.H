/* Copyright 2021 RevathiJambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIVEREDUCTION_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIVEREDUCTION_H_

#include "ReducedDiags.H"
#include "Utils/CoarsenIO.H"
#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>


/**
 * This class contains a function that computes an arbitrary reduction of the fields. The function
 * used in the reduction is defined by an input file parser expression and the reduction operation
 * can be either Maximum, Minimum, or Integral (Sum multiplied by cell volume).
 */
class divEReduction : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    divEReduction(std::string rd_name);

    /**
     * This function is called at every time step, and if necessary calls the templated function
     * ComputeFieldReduction(), which does the actual reduction computation.
     *
     * @param[in] step the timestep
     */
    virtual void ComputeDiags(int step) override final;

private:
    /// Parser to read expression to be reduced from the input file.
    /// 9 elements are x, y, z, Ex, Ey, Ez, Bx, By, Bz
    static constexpr int m_nvars = 3;
    std::unique_ptr<amrex::Parser> m_parser;

    // Type of reduction (e.g. Maximum, Minimum or Sum)
    int m_reduction_type;
    int m_integral_type;

public:

    template<typename ReduceOp>
    void ComputedivEReduction ()
    {
        using namespace amrex::literals;
        // get a reference to WarpX instance
        auto & warpx = WarpX::GetInstance();

        // Guard cell is set to 1 for generality. However, for a cell-centered
        // output Multifab, mf_dst, the guard-cell data is not needed especially considering
        // the operations performend in the CoarsenAndInterpolate function.
        constexpr int ng = 1;
        const int lev = 0;
        // For staggered and nodal calculations, divE is computed on the nodes.
        // The temporary divE MultiFab is generated to comply with the location of divE.
        amrex::IntVect cell_type = amrex::IntVect::TheNodeVector();
        const amrex::BoxArray& ba = amrex::convert(warpx.boxArray(lev), cell_type);
        amrex::MultiFab divE(ba, warpx.DistributionMap(lev), 2*warpx.n_rz_azimuthal_modes-1, ng );
        warpx.ComputeDivE(divE, lev);

        amrex::Geometry const & geom = warpx.Geom(lev);
        const amrex::RealBox& real_box = geom.ProbDomain();
        const auto dx = geom.CellSizeArray();


        // General preparation of interpolation and reduction operations
        const amrex::GpuArray<int,3> nodeCenteredtype{1,1,1};
        const amrex::GpuArray<int,3> reduction_coarsening_ratio{1,1,1};
        constexpr int reduction_comp = 0;

        amrex::ReduceOps<ReduceOp> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        auto divEType = amrex::GpuArray<int, 3>{1,1,1};

        // get parser
        auto reduction_function_parser = m_parser->compile<m_nvars>();
        int integral_type = m_integral_type;
        amrex::IntVect iv = divE.ixType().toIntVect();

        // MFIter loop to interpolate fields to cell center and perform reduction
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(divE, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const amrex::Box& box = mfi.nodaltilebox();
            amrex::Array4<amrex::Real> const& divEarr = divE.array(mfi);
            reduce_op.eval(box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
            {
                // Shift x, y, z position based on index type
                amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
                amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
                return reduction_function_parser(x,y,z)*divEarr(i,j,k);
            });
        }

        amrex::Real reduce_value = amrex::get<0>(reduce_data.value());

        // MPI reduce
        if (std::is_same<ReduceOp, amrex::ReduceOpMax>::value)
        {
            amrex::ParallelDescriptor::ReduceRealMax(reduce_value);
        }
        if (std::is_same<ReduceOp, amrex::ReduceOpMin>::value)
        {
            amrex::ParallelDescriptor::ReduceRealMin(reduce_value);
        }
        if (std::is_same<ReduceOp, amrex::ReduceOpSum>::value)
        {
            amrex::ParallelDescriptor::ReduceRealSum(reduce_value);
        // If reduction operation is a sum, multiply the value by the cell volume so that the
        // result is the integral of the function over the simulation domain.
            if (integral_type == 0) {
#if (AMREX_SPACEDIM==2)
            reduce_value *= dx[0]*dx[1];
#else
            reduce_value *= dx[0]*dx[1]*dx[2];
#endif
            } else {
            // works only when dx = dy = dz
#if (AMREX_SPACEDIM==2)
            reduce_value *= dx[0];
#else
            reduce_value *= dx[0]*dx[1];
#endif
            }
        }

        // Fill output array
        m_data[0] = reduce_value;
    }

};

#endif
