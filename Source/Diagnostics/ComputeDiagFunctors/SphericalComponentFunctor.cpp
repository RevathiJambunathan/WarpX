#include "SphericalComponentFunctor.H"
#include "Utils/CoarsenIO.H"
#ifdef PULSAR
    #include "Particles/PulsarParameters.H"
#endif
#include "WarpX.H"
#include <AMReX_IntVect.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX.H>

SphericalComponentFunctor::SphericalComponentFunctor (amrex::MultiFab const * mfx_src,
                                                      amrex::MultiFab const * mfy_src, 
                                                      amrex::MultiFab const * mfz_src,
                                                      int lev,
                                                      amrex::IntVect crse_ratio,
                                                      int sphericalcomp,
                                                      const int Efield,
                                                      int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_mfx_src(mfx_src),
      m_mfy_src(mfy_src), m_mfz_src(mfz_src), m_lev(lev), m_sphericalcomp(sphericalcomp),
      m_Efield(Efield)
{}


void 
SphericalComponentFunctor::operator ()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const
{
    using namespace amrex;
    auto & warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(m_lev).CellSizeArray();
    const auto problo = warpx.Geom(m_lev).ProbLoArray();
    const auto probhi = warpx.Geom(m_lev).ProbHiArray();
    const amrex::IntVect stag_dst = mf_dst.ixType().toIntVect();

    // convert boxarray of source MultiFab to staggering of dst Multifab
    // and coarsen it
    amrex::BoxArray ba_tmp = amrex::convert( m_mfx_src->boxArray(), stag_dst);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE (ba_tmp.coarsenable (m_crse_ratio),
        "source Multifab converted to staggering of dst Multifab is not coarsenable");
    ba_tmp.coarsen(m_crse_ratio);

    if (ba_tmp == mf_dst.boxArray() and m_mfx_src->DistributionMap() == mf_dst.DistributionMap()) {
        ComputeSphericalFieldComponent(mf_dst, dcomp);
    } else {
        const int ncomp = 1;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_mfx_src->DistributionMap() == m_mfy_src->DistributionMap() and m_mfy_src->DistributionMap() == m_mfz_src->DistributionMap(), 
            " all sources must have the same Distribution map");
        amrex::MultiFab mf_tmp( ba_tmp, m_mfx_src->DistributionMap(), ncomp, 0);
        const int dcomp_tmp = 0;
        ComputeSphericalFieldComponent(mf_tmp, dcomp_tmp);
        mf_dst.copy( mf_tmp, 0, dcomp, ncomp);
    }
}

void
SphericalComponentFunctor::ComputeSphericalFieldComponent( amrex::MultiFab& mf_dst, int dcomp) const
{
    using namespace amrex;
    auto & warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(m_lev).CellSizeArray();
    const auto problo = warpx.Geom(m_lev).ProbLoArray();
    const auto probhi = warpx.Geom(m_lev).ProbHiArray();
    const amrex::IntVect stag_xsrc = m_mfx_src->ixType().toIntVect();
    const amrex::IntVect stag_ysrc = m_mfy_src->ixType().toIntVect();
    const amrex::IntVect stag_zsrc = m_mfz_src->ixType().toIntVect();
    const amrex::IntVect stag_dst = mf_dst.ixType().toIntVect();

    amrex::GpuArray<int,3> sfx; // staggering of source xfield
    amrex::GpuArray<int,3> sfy; // staggering of source yfield
    amrex::GpuArray<int,3> sfz; // staggering of source zfield
    amrex::GpuArray<int,3> s_dst;
    amrex::GpuArray<int,3> cr;
    
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        sfx[i] = stag_xsrc[i];
        sfy[i] = stag_ysrc[i];
        sfz[i] = stag_zsrc[i];
        s_dst[i]  = stag_dst[i];
        cr[i] = m_crse_ratio[i];
    }
    const int sphericalcomp = m_sphericalcomp;
    amrex::Real cur_time = warpx.gett_new(0);
    int Efield = m_Efield;
#ifdef PULSAR

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf_dst, TilingIfNotGPU() ); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(mf_dst.nGrowVect());
        amrex::Array4<amrex::Real> const& arr_dst = mf_dst.array(mfi);
        amrex::Array4<amrex::Real const> const& xarr_src = m_mfx_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const& yarr_src = m_mfy_src->const_array(mfi);
        amrex::Array4<amrex::Real const> const& zarr_src = m_mfz_src->const_array(mfi);
        const int ncomp = 1;
        amrex::ParallelFor ( bx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::Real cc_xfield = CoarsenIO::Interp(xarr_src, sfx, s_dst, cr,
                                        i, j, k, n);
                amrex::Real cc_yfield = CoarsenIO::Interp(yarr_src, sfy, s_dst, cr,
                                        i, j, k, n);
                amrex::Real cc_zfield = CoarsenIO::Interp(zarr_src, sfz, s_dst, cr,
                                        i, j, k, n);
                // convert to spherical coordinates
                // compute cell coordinates
                amrex::Real x, y, z;
                PulsarParm::ComputeCellCoordinates(i,j,k, s_dst, problo, dx, x, y, z);
                // convert cartesian to spherical coordinates
                amrex::Real r, theta, phi;
                PulsarParm::ConvertCartesianToSphericalCoord(x, y, z, problo, probhi,
                                                             r, theta, phi);

                if (sphericalcomp == 0) { // rcomponent of field 
                    PulsarParm::ConvertCartesianToSphericalRComponent(
                        cc_xfield, cc_yfield, cc_zfield, theta, phi, arr_dst(i,j,k,n+dcomp));
                } else if (sphericalcomp == 1) { // theta component of field
                    PulsarParm::ConvertCartesianToSphericalThetaComponent(
                        cc_xfield, cc_yfield, cc_zfield, theta, phi, arr_dst(i,j,k,n+dcomp));
                } else if (sphericalcomp == 2) { // phi component of field
                    PulsarParm::ConvertCartesianToSphericalPhiComponent(
                        cc_xfield, cc_yfield, cc_zfield, theta, phi, arr_dst(i,j,k,n+dcomp));
                }
            });
    }

#endif
}
