#include "jdotEFunctor.H"
#include "WarpX.H"
#include <ablastr/coarsen/sample.H>
#include <AMReX_IntVect.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX.H>

jdotEFunctor::jdotEFunctor (amrex::MultiFab const * Ex_src, amrex::MultiFab const * Ey_src,
                            amrex::MultiFab const * Ez_src, amrex::MultiFab const * jx_src,
                            amrex::MultiFab const * jy_src, amrex::MultiFab const * jz_src,
                            const int lev, amrex::IntVect crse_ratio, int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_Ex_src(Ex_src), m_Ey_src(Ey_src),
      m_Ez_src(Ez_src), m_jx_src(jx_src), m_jy_src(jy_src), m_jz_src(jz_src),
      m_lev(lev)
{
    amrex::ignore_unused(m_lev);
}

void
jdotEFunctor::operator ()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const
{
    using namespace amrex;
    const amrex::IntVect stag_dst = mf_dst.ixType().toIntVect();

    // convert boxarray of source MultiFab to staggering of dst Multifab
    // and coarsen it
    amrex::BoxArray ba_tmp = amrex::convert( m_Ex_src->boxArray(), stag_dst);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE (ba_tmp.coarsenable (m_crse_ratio),
        "source Multifab converted to staggering of dst Multifab is not coarsenable");
    ba_tmp.coarsen(m_crse_ratio);

    if (ba_tmp == mf_dst.boxArray() and m_Ex_src->DistributionMap() == mf_dst.DistributionMap()) {
        ComputejdotE(mf_dst, dcomp);
    } else {
        const int ncomp = 1;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_Ex_src->DistributionMap() == m_Ey_src->DistributionMap() and m_Ey_src->DistributionMap() == m_Ez_src->DistributionMap(),
            " all sources must have the same Distribution map");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_jx_src->DistributionMap() == m_jy_src->DistributionMap() and m_jy_src-> DistributionMap() == m_jz_src->DistributionMap(),
            " all sources must have the same Distribution map");
        amrex::MultiFab mf_tmp( ba_tmp, m_Ex_src->DistributionMap(), ncomp, 0);
        const int dcomp_tmp = 0;
        ComputejdotE(mf_tmp, dcomp_tmp);
        mf_dst.ParallelCopy( mf_tmp, 0, dcomp, ncomp);
    }
}

void
jdotEFunctor::ComputejdotE(amrex::MultiFab& mf_dst, int dcomp) const
{
    using namespace amrex;
    const amrex::IntVect stag_Exsrc = m_Ex_src->ixType().toIntVect();
    const amrex::IntVect stag_Eysrc = m_Ey_src->ixType().toIntVect();
    const amrex::IntVect stag_Ezsrc = m_Ez_src->ixType().toIntVect();
    const amrex::IntVect stag_jxsrc = m_jx_src->ixType().toIntVect();
    const amrex::IntVect stag_jysrc = m_jy_src->ixType().toIntVect();
    const amrex::IntVect stag_jzsrc = m_jz_src->ixType().toIntVect();
    const amrex::IntVect stag_dst = mf_dst.ixType().toIntVect();

    amrex::GpuArray<int,3> sf_Ex; // staggering of source xfield
    amrex::GpuArray<int,3> sf_Ey; // staggering of source yfield
    amrex::GpuArray<int,3> sf_Ez; // staggering of source zfield
    amrex::GpuArray<int,3> sf_jx; // staggering of source xfield
    amrex::GpuArray<int,3> sf_jy; // staggering of source yfield
    amrex::GpuArray<int,3> sf_jz; // staggering of source zfield
    amrex::GpuArray<int,3> s_dst;
    amrex::GpuArray<int,3> cr;

    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        sf_Ex[i] = stag_Exsrc[i];
        sf_Ey[i] = stag_Eysrc[i];
        sf_Ez[i] = stag_Ezsrc[i];
        sf_jx[i] = stag_jxsrc[i];
        sf_jy[i] = stag_jysrc[i];
        sf_jz[i] = stag_jzsrc[i];
        s_dst[i]  = stag_dst[i];
        cr[i] = m_crse_ratio[i];
    }


#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf_dst, TilingIfNotGPU() ); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(mf_dst.nGrowVect());
        amrex::Array4<amrex::Real> const& arr_dst = mf_dst.array(mfi);
        amrex::Array4<amrex::Real const> const & Ex_arr = m_Ex_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Ey_arr = m_Ey_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Ez_arr = m_Ez_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const & jx_arr = m_jx_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const & jy_arr = m_jy_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const & jz_arr = m_jz_src->const_array(mfi);
        const int ncomp = 1;
        amrex::ParallelFor (bx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::Real Ex_cc = ablastr::coarsen::sample::Interp(Ex_arr, sf_Ex, s_dst, cr,
                                            i, j, k, n);
                amrex::Real Ey_cc = ablastr::coarsen::sample::Interp(Ey_arr, sf_Ey, s_dst, cr,
                                            i, j, k, n);
                amrex::Real Ez_cc = ablastr::coarsen::sample::Interp(Ez_arr, sf_Ez, s_dst, cr,
                                            i, j, k, n);
                amrex::Real jx_cc = ablastr::coarsen::sample::Interp(jx_arr, sf_jx, s_dst, cr,
                                            i, j, k, n);
                amrex::Real jy_cc = ablastr::coarsen::sample::Interp(jy_arr, sf_jy, s_dst, cr,
                                            i, j, k, n);
                amrex::Real jz_cc = ablastr::coarsen::sample::Interp(jz_arr, sf_jz, s_dst, cr,
                                                      i, j, k, n);
                arr_dst(i,j,k,n+dcomp) = Ex_cc * jx_cc + Ey_cc * jy_cc + Ez_cc * jz_cc;
            });
    }
}
