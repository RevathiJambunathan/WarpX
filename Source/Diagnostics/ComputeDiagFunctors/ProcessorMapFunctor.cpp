#include "ProcessorMapFunctor.H"

#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

ProcessorMapFunctor::ProcessorMapFunctor(amrex::MultiFab const * mf_src, int lev,
                                     amrex::IntVect crse_ratio,
                                     bool convertRZmodes2cartesian, int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_mf_src(mf_src), m_lev(lev),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{}

void
ProcessorMapFunctor::operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    std::unique_ptr<amrex::MultiFab> tmp;
    tmp = std::make_unique<amrex::MultiFab>(m_mf_src->boxArray(), m_mf_src->DistributionMap(), 1, m_mf_src->nGrowVect());

    auto& warpx = WarpX::GetInstance();
    const amrex::DistributionMapping& dm = warpx.DistributionMap(m_lev);
    for (amrex::MFIter mfi(*tmp, false); mfi.isValid(); ++mfi)
    {
        auto bx = mfi.growntilebox(tmp->nGrowVect());
        auto arr = tmp->array(mfi);
        int iproc = dm[mfi.index()];
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                arr(i,j,k) = iproc;
            });
    }
    InterpolateMFForDiag(mf_dst, *tmp, dcomp, warpx.DistributionMap(m_lev),
                         m_convertRZmodes2cartesian);
}
