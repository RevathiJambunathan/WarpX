#include "NumberDensityFunctor.H"

#include "Utils/CoarsenIO.H"
#include "Utils/TextMsg.H"
#ifdef WARPX_DIM_RZ
#   include "WarpX.H"
#endif

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

NumberDensityFunctor::NumberDensityFunctor(amrex::MultiFab const * mf_src, int lev,
                                     amrex::IntVect crse_ratio,
                                     int species_index, int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_mf_src(mf_src), m_lev(lev),
      m_scomp(species_index)
{}

void
NumberDensityFunctor::operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
#ifndef WARPX_DIM_RZ
    amrex::Print() << " ndens : " << m_scomp << " ncomp " << nComp() << " dcomp " << dcomp << "\n";
    // In cartesian geometry, coarsen and interpolate from simulation MultiFab, m_mf_src,
    // to output diagnostic MultiFab, mf_dst.
    CoarsenIO::Coarsen( mf_dst, *m_mf_src, dcomp, m_scomp, nComp(), mf_dst.nGrowVect(), m_crse_ratio);
    amrex::ignore_unused(m_lev);
#endif
}
