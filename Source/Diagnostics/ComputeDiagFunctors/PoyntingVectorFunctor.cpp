#include "PoyntingVectorFunctor.H"
#include <ablastr/coarsen/sample.H>
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_IntVect.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX.H>

PoyntingVectorFunctor::PoyntingVectorFunctor (
                           amrex::MultiFab const * Ex_src, amrex::MultiFab const * Ey_src,
                           amrex::MultiFab const * Ez_src, amrex::MultiFab const * Bx_src,
                           amrex::MultiFab const * By_src, amrex::MultiFab const * Bz_src,
                           const int lev, amrex::IntVect crse_ratio,
                           int vectorcomp, int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_Ex_src(Ex_src), m_Ey_src(Ey_src),
      m_Ez_src(Ez_src), m_Bx_src(Bx_src), m_By_src(By_src), m_Bz_src(Bz_src),
      m_lev(lev), m_vectorcomp(vectorcomp)
{
    amrex::ignore_unused(m_lev);
}

void
PoyntingVectorFunctor::operator ()(amrex::MultiFab& mf_dst,
                                   int dcomp, const int /*i_buffer=0*/) const
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
        ComputePoyntingVector(mf_dst, dcomp);
    } else {
        const int ncomp = 1;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_Ex_src->DistributionMap() == m_Ey_src->DistributionMap() and m_Ey_src->DistributionMap() == m_Ez_src->DistributionMap(),
            " all sources must have the same Distribution map");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_Bx_src->DistributionMap() == m_By_src->DistributionMap() and m_By_src->DistributionMap() == m_Bz_src->DistributionMap(),
            " all sources must have the same Distribution map");
        amrex::MultiFab mf_tmp( ba_tmp, m_Ex_src->DistributionMap(), ncomp, 0 );
        const int dcomp_tmp = 0;
        ComputePoyntingVector(mf_tmp, dcomp_tmp);
        mf_dst.ParallelCopy( mf_tmp, 0, dcomp, ncomp);
    }

}

void
PoyntingVectorFunctor::ComputePoyntingVector(amrex::MultiFab& mf_dst, int dcomp) const
{
    using namespace amrex;
    const amrex::IntVect stag_Exsrc = m_Ex_src->ixType().toIntVect();
    const amrex::IntVect stag_Eysrc = m_Ey_src->ixType().toIntVect();
    const amrex::IntVect stag_Ezsrc = m_Ez_src->ixType().toIntVect();
    const amrex::IntVect stag_Bxsrc = m_Bx_src->ixType().toIntVect();
    const amrex::IntVect stag_Bysrc = m_By_src->ixType().toIntVect();
    const amrex::IntVect stag_Bzsrc = m_Bz_src->ixType().toIntVect();
    const amrex::IntVect stag_dst = mf_dst.ixType().toIntVect();
    auto & warpx = WarpX::GetInstance();
    auto dx = warpx.Geom(m_lev).CellSizeArray();
    const auto problo = warpx.Geom(m_lev).ProbLoArray();

    amrex::GpuArray<int,3> sf_Ex; // staggering of source xfield
    amrex::GpuArray<int,3> sf_Ey; // staggering of source yfield
    amrex::GpuArray<int,3> sf_Ez; // staggering of source zfield
    amrex::GpuArray<int,3> sf_Bx; // staggering of source xfield
    amrex::GpuArray<int,3> sf_By; // staggering of source yfield
    amrex::GpuArray<int,3> sf_Bz; // staggering of source zfield
    amrex::GpuArray<int,3> s_dst;
    amrex::GpuArray<int,3> cr;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star_arr;

    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        sf_Ex[i] = stag_Exsrc[i];
        sf_Ey[i] = stag_Eysrc[i];
        sf_Ez[i] = stag_Ezsrc[i];
        sf_Bx[i] = stag_Bxsrc[i];
        sf_By[i] = stag_Bysrc[i];
        sf_Bz[i] = stag_Bzsrc[i];
        s_dst[i]  = stag_dst[i];
        cr[i] = m_crse_ratio[i];
        dx[i] = dx[i] * cr[i];
        center_star_arr[i] = Pulsar::m_center_star[i];
    }
    const int vectorcomp = m_vectorcomp;



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
        amrex::Array4<amrex::Real const> const & Bx_arr = m_Bx_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const & By_arr = m_By_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Bz_arr = m_Bz_src->const_array(mfi);
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
                amrex::Real Bx_cc = ablastr::coarsen::sample::Interp(Bx_arr, sf_Bx, s_dst, cr,
                                                    i, j, k, n);
                amrex::Real By_cc = ablastr::coarsen::sample::Interp(By_arr, sf_By, s_dst, cr,
                                                    i, j, k, n);
                amrex::Real Bz_cc = ablastr::coarsen::sample::Interp(Bz_arr, sf_Bz, s_dst, cr,
                                                      i, j, k, n);
                amrex::Real Sx = Ey_cc * Bz_cc - Ez_cc * By_cc;
                amrex::Real Sy = Ez_cc * Bx_cc - Ex_cc * Bz_cc;
                amrex::Real Sz = Ex_cc * By_cc - Ey_cc * Bx_cc;
                if (vectorcomp == 0) {
                    arr_dst(i,j,k,n+dcomp) = Sx;
                } else if (vectorcomp == 1) {
                    arr_dst(i,j,k,n+dcomp) = Sy;
                } else if (vectorcomp == 2) {
                    arr_dst(i,j,k,n+dcomp) = Sz;
                } else if (vectorcomp > 2) {
                    // Compute spherical components of Poynting flux (ExB)
                    // compute cell coordinates
                    amrex::Real x, y, z;
                    Pulsar::ComputeCellCoordinates(i,j,k, s_dst, problo, dx, x, y, z);
                    // convert cartesian to spherical coordinates
                    amrex::Real r, theta, phi;
                    Pulsar::ConvertCartesianToSphericalCoord(x, y, z, center_star_arr,
                                                             r, theta, phi);
                    if (vectorcomp == 3) { // rcomponent
                        Pulsar::ConvertCartesianToSphericalRComponent(
                        Sx, Sy, Sz, theta, phi, arr_dst(i,j,k,n+dcomp));
                    } else if (vectorcomp == 4) { // theta component
                        Pulsar::ConvertCartesianToSphericalThetaComponent(
                        Sx, Sy, Sz, theta, phi, arr_dst(i,j,k,n+dcomp));
                    } else if (vectorcomp == 5) { // phi component
                        Pulsar::ConvertCartesianToSphericalPhiComponent(
                        Sx, Sy, Sz, theta, phi, arr_dst(i,j,k,n+dcomp));
                    } else if (vectorcomp == 6) { // vdrift phi
                        // vdrift_phi = (ExB)_phi_component / (B.B)
                        amrex::Real S_phi = 0.0;
                        Pulsar::ConvertCartesianToSphericalPhiComponent(
                        Sx, Sy, Sz, theta, phi, S_phi);
                        amrex::Real B2 = Bx_cc*Bx_cc + By_cc*By_cc + Bz_cc*Bz_cc;
                        arr_dst(i,j,k,n+dcomp) = S_phi/B2;
                    }
                }
            });
    }

}
