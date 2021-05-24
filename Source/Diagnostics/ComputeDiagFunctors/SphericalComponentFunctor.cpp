#include "SphericalComponentFunctor.H"
#include "Utils/CoarsenIO.H"
#ifdef PULSAR
    #include "Particles/PulsarParameters.H"
#endif

#include <AMReX.H>

SphericalComponentFunctor::SphericalComponentFunctor (amrex::MultiFab const * mfx_src,
                                                      amrex::MultiFab const * mfy_src, 
                                                      amrex::MultiFab const * mfz_src,
                                                      int lev,
                                                      amrex::IntVect crse_ratio,
                                                      int sphericalcomp,
                                                      const bool Efield,
                                                      int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_mfx_src(mfx_src),
      m_mfy_src(mfy_src), m_mfz_src(mfz_src), m_lev(lev), m_sphericalcomp(sphericalcomp),
      m_Efield(Efield)
{}


void 
SphericalComponentFunctor::operator ()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const
{
    auto & warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(m_lev).CellSizeArray();
    const auto problo = warpx.Geom(m_lev).ProbLoArray();
    const auto probhi = warpx.Geom(m_lev).ProbHiArray();
    const IntVect stag_xsrc = m_mfx_src->ixType().toIntVect();
    const IntVect stag_ysrc = m_mfy_src->ixType().toIntVect();
    const IntVect stag_zsrc = m_mfz_src->ixType().toIntVect();
    const IntVect stag_dst = mf_dst.ixType().toIntVect();
    GpuArray<int,3> sfx; // staggering of source xfield
    GpuArray<int,3> sfy; // staggering of source yfield
    GpuArray<int,3> sfz; // staggering of source zfield
    GpuArray<int,3> s_dst;
    GpuArray<int,3> cr;
    
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        sfx[i] = stag_xsrc[i];
        sfy[i] = stag_ysrc[i];
        sfz[i] = stag_zsrc[i];
        s_dst[i]  = stag_dst[i];
        cr[i] = m_crse_ratio[i];
    }
    const int sphericalcomp = m_sphericalcomp;
    amrex::Real cur_time = warpx.gett_new(0);
    bool Efield = m_Efield;
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
                // compute theoretical E and B field
                amrex::Real Fr_theory, Ftheta_theory, Fphi_theory;
                if (Efield == true) {
                    PulsarParm::CorotatingEfieldSpherical(r, theta, phi, cur_time, Fr_theory,
                                             Ftheta_theory, Fphi_theory);
                } else {
                    PulsarParm::ExternalBFieldSpherical (r, theta, phi, cur_time, Fr_theory,
                                             Ftheta_theory, Fphi_theory);
                }              

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

                // print difference at pole
                amrex::Real theta_min, theta_max, phi_min, phi_max, r_min, r_max;
                theta_min = 0; theta_max = 0.08;
                phi_min = 1.48; phi_max = 1.6;
                r_min = PulsarParm::R_star - PulsarParm::dR_star;
                r_max = PulsarParm::R_star + PulsarParm::dR_star;

                if (r >= r_min and r <= r_max) {
                if (theta >= theta_min and theta <= theta_max) {
                if (phi >= phi_min and phi <= phi_max) {
                    if (sphericalcomp == 0) {
                        amrex::Print() << " Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Fr_theory " << Fr_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Fr_theory << "\n";                      
                    } else if (sphericalcomp == 1) {
                        amrex::Print() << " Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Ftheta_theory " << Ftheta_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Ftheta_theory << "\n";                      
                    } else if (sphericalcomp == 2) {
                        amrex::Print() << " Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Fphi_theory " << Fphi_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Fphi_theory << "\n";                      
                    }
                }}}

                // at the equator
                theta_min = 1.48; theta_max = 1.6;
                if (r >= r_min and r <= r_max) {
                if (theta >= theta_min and theta <= theta_max) {
                if (phi >= phi_min and phi <= phi_max) {
                    if (sphericalcomp == 0) {
                        amrex::Print() << " equator Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Fr_theory " << Fr_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Fr_theory << "\n";                      
                    } else if (sphericalcomp == 1) {
                        amrex::Print() << " equator Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Ftheta_theory " << Ftheta_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Ftheta_theory << "\n";                      
                    } else if (sphericalcomp == 2) {
                        amrex::Print() << " equator Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Fphi_theory " << Fphi_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Fphi_theory << "\n";                      
                    }
                }}}
               
                // at theta=55 
                theta_min = 0.96; theta_max = 1.05;
                if (r >= r_min and r <= r_max) {
                if (theta >= theta_min and theta <= theta_max) {
                if (phi >= phi_min and phi <= phi_max) {
                    if (sphericalcomp == 0) {
                        amrex::Print() << " theta55 Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Fr_theory " << Fr_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Fr_theory << "\n";                      
                    } else if (sphericalcomp == 1) {
                        amrex::Print() << " theta55 Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Ftheta_theory " << Ftheta_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Ftheta_theory << "\n";                      
                    } else if (sphericalcomp == 2) {
                        amrex::Print() << " theta55 Efield? " << Efield << " r " << r << " theta " << theta << " phi " << phi << " rcomp_sim : " << arr_dst(i,j,k,n+dcomp) << " Fphi_theory " << Fphi_theory << " diff : " << arr_dst(i,j,k,n+dcomp) - Fphi_theory << "\n";                      
                    }
                }}}

            });
    }

#endif
}
