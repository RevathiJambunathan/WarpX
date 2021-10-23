#ifndef WARPX_RECONNECTIONPERTURBATION_H_
#define WARPX_RECONNECTIONPERTURBATION_H_

#include "WarpX.H"
#include <AMReX.H>
#include <AMReX_Gpu.H>
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
#include <complex>
#include <cmath>
#include <gsl/gsl_sf_dilog.h>
#include <gsl/gsl_sf_result.h>

class
Reconnection_Perturbation {

public:

    Reconnection_Perturbation (){};

    static void AddBfieldPerturbation (amrex::MultiFab *Bx,
                              amrex::MultiFab *By,
                              amrex::MultiFab *Bz,
                              amrex::ParserExecutor<3> const& xfield_parser,
                              amrex::ParserExecutor<3> const& yfield_parser,
                              amrex::ParserExecutor<3> const& zfield_parser, const int lev);

    AMREX_FORCE_INLINE
    static std::complex<double> getComplexDilog ( const std::complex<double> z_arg) {
        gsl_sf_result z_dilog_re, z_dilog_im;
        amrex::Real magnitude = std::sqrt(z_arg.real()*z_arg.real() + z_arg.imag()*z_arg.imag());
        amrex::Real theta = std::atan2(z_arg.imag(), z_arg.real());
        int error = gsl_sf_complex_dilog_e( magnitude, theta,
                                            &z_dilog_re, &z_dilog_im );
        std::complex<double> complex_dilog_result;
        complex_dilog_result.real(z_dilog_re.val);
        complex_dilog_result.imag(z_dilog_im.val);
        return complex_dilog_result;
    }
    
    /**
     * Perform derivative along x on a cell-centered grid, from a nodal field `F` */
    AMREX_FORCE_INLINE
    static amrex::Real IntegralBz (
        const amrex::Real x, const amrex::Real z,
        const amrex::Real pi_val, const amrex::Real xcs, const amrex::Real B0,
        const amrex::Real nd_ratio, const amrex::Real delta)
    {
        using namespace amrex;
        std::complex<double> i(0.0, 1.);

        amrex::Real prefactor = (2._rt * B0) / ( (pi_val/2._rt) + nd_ratio - 1._rt);
        amrex::Real term1 = pi_val/4._rt * (x - 2._rt * xcs);
        amrex::Real term2 = 0.5_rt * (nd_ratio - 1._rt)
                          * ( delta * std::log( std::cosh ( (x-xcs) / delta) )
                            - delta * std::log( std::cosh ( (x+xcs) / delta) )
                            + x
                            );
        std::complex<double> term3 = delta * i / 2.0_rt
                                     * ( getComplexDilog( -i * std::exp( -(x-xcs)/delta) )
                                       - getComplexDilog(  i * std::exp( -(x-xcs)/delta) )
                                       );
        std::complex<double> term4 = - delta * i / 2.0_rt
                                     * ( getComplexDilog( -i * std::exp( -(x+xcs)/delta) )
                                       - getComplexDilog(  i * std::exp( -(x+xcs)/delta) )
                                       );
        std::complex<double> IntegralBz_val = prefactor * ( term1 + term2 + term3 + term4);
        return IntegralBz_val.real();
    }


};


#endif
