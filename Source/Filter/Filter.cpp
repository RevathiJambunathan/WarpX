/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Filter.H"

#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <algorithm>

using namespace amrex;

#ifdef AMREX_USE_GPU

/* \brief Apply stencil on MultiFab (GPU version, 2D/3D).
 * \param dstmf Destination MultiFab
 * \param srcmf source MultiFab
 * \param[in] lev mesh refinement level
 * \param scomp first component of srcmf on which the filter is applied
 * \param dcomp first component of dstmf on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (MultiFab& dstmf, const MultiFab& srcmf, const int lev, int scomp, int dcomp, int ncomp)
{
    WARPX_PROFILE("Filter::ApplyStencil(MultiFab)");
    ncomp = std::min(ncomp, srcmf.nComp());

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    amrex::Print() << " in apply stencil for filter \n";

    for (MFIter mfi(dstmf); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        amrex::Real wt = amrex::second();

        const auto& src = srcmf.array(mfi);
        const auto& dst = dstmf.array(mfi);
        const Box& tbx = mfi.growntilebox();
        const Box& gbx = amrex::grow(tbx,stencil_length_each_dir-1);
        amrex::Print() << " tbx : " << tbx << " gbx : " << gbx << "\n";

        // tmpfab has enough ghost cells for the stencil
        FArrayBox tmp_fab(gbx,ncomp);
        Elixir tmp_eli = tmp_fab.elixir();  // Prevent the tmp data from being deleted too early
        auto const& tmp = tmp_fab.array();

        // Copy values in srcfab into tmpfab
        const Box& ibx = gbx & srcmf[mfi].box();
        AMREX_PARALLEL_FOR_4D ( gbx, ncomp, i, j, k, n,
        {
            if (ibx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                tmp(i,j,k,n) = src(i,j,k,n+scomp);
            } else {
                tmp(i,j,k,n) = 0.0;
            }
        });

        // Apply filter
        DoFilter(tbx, tmp, dst, 0, dcomp, ncomp);

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = amrex::second() - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

/* \brief Apply stencil on FArrayBox (GPU version, 2D/3D).
 * \param dstfab Destination FArrayBox
 * \param srcmf source FArrayBox
 * \param tbx Grown box on which srcfab is defined.
 * \param scomp first component of srcfab on which the filter is applied
 * \param dcomp first component of dstfab on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (FArrayBox& dstfab, const FArrayBox& srcfab,
                      const Box& tbx, int scomp, int dcomp, int ncomp)
{
    WARPX_PROFILE("Filter::ApplyStencil(FArrayBox)");
    ncomp = std::min(ncomp, srcfab.nComp());
    const auto& src = srcfab.array();
    const auto& dst = dstfab.array();
    const Box& gbx = amrex::grow(tbx,stencil_length_each_dir-1);

    // tmpfab has enough ghost cells for the stencil
    FArrayBox tmp_fab(gbx,ncomp);
    Elixir tmp_eli = tmp_fab.elixir();  // Prevent the tmp data from being deleted too early
    auto const& tmp = tmp_fab.array();

    // Copy values in srcfab into tmpfab
    const Box& ibx = gbx & srcfab.box();
    AMREX_PARALLEL_FOR_4D ( gbx, ncomp, i, j, k, n,
        {
            if (ibx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                tmp(i,j,k,n) = src(i,j,k,n+scomp);
            } else {
                tmp(i,j,k,n) = 0.0;
            }
        });

    // Apply filter
    DoFilter(tbx, tmp, dst, 0, dcomp, ncomp);
}

/* \brief Apply stencil (2D/3D, CPU/GPU)
 */
void Filter::DoFilter (const Box& tbx,
                       Array4<Real const> const& tmp,
                       Array4<Real      > const& dst,
                       int scomp, int dcomp, int ncomp)
{
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    Dim3 slen_local = slen;
#if (AMREX_SPACEDIM == 3)
    AMREX_PARALLEL_FOR_4D ( tbx, ncomp, i, j, k, n,
    {
        amrex::Print() << " i j k " << i << " " <<  j << " " << k << "\n";
        Real d = 0.0;
        amrex::Print() << " slen z " << slen_local.z;
        amrex::Print() << " slen y " << slen_local.y;
        amrex::Print() << " slen x " << slen_local.x << "\n";
        for         (int iz=0; iz < slen_local.z; ++iz){
            for     (int iy=0; iy < slen_local.y; ++iy){
                for (int ix=0; ix < slen_local.x; ++ix){
                    amrex::Print() << " sx " << sx[ix] << " "<< sx[iy] << " " << sx[iz] << "\n";
                    Real sss = sx[ix]*sy[iy]*sz[iz];
//                    amrex::Print() << " xcells : " << i-ix << " " << i+ix << "\n";
                    d += sss*( tmp(i-ix,j-iy,k-iz,scomp+n)
                              +tmp(i+ix,j-iy,k-iz,scomp+n)
                              +tmp(i-ix,j+iy,k-iz,scomp+n)
                              +tmp(i+ix,j+iy,k-iz,scomp+n)
                              +tmp(i-ix,j-iy,k+iz,scomp+n)
                              +tmp(i+ix,j-iy,k+iz,scomp+n)
                              +tmp(i-ix,j+iy,k+iz,scomp+n)
                              +tmp(i+ix,j+iy,k+iz,scomp+n));
                    amrex::Print() << " d : " << d << "\n";
                }
            }
        }

        dst(i,j,k,dcomp+n) = d;
    });
#else
    AMREX_PARALLEL_FOR_4D ( tbx, ncomp, i, j, k, n,
    {
        Real d = 0.0;

        for         (int iz=0; iz < slen_local.z; ++iz){
            for     (int iy=0; iy < slen_local.y; ++iy){
                for (int ix=0; ix < slen_local.x; ++ix){
                    Real sss = sx[ix]*sz[iy];
                    d += sss*( tmp(i-ix,j-iy,k,scomp+n)
                              +tmp(i+ix,j-iy,k,scomp+n)
                              +tmp(i-ix,j+iy,k,scomp+n)
                              +tmp(i+ix,j+iy,k,scomp+n));
                }
            }
        }

        dst(i,j,k,dcomp+n) = d;
    });
#endif
}

#else

/* \brief Apply stencil on MultiFab (CPU version, 2D/3D).
 * \param dstmf Destination MultiFab
 * \param srcmf source MultiFab
 * \param[in] lev mesh refinement level
 * \param scomp first component of srcmf on which the filter is applied
 * \param dcomp first component of dstmf on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (amrex::MultiFab& dstmf, const amrex::MultiFab& srcmf, const int lev, int scomp, int dcomp, int ncomp)
{
    WARPX_PROFILE("Filter::ApplyStencil(MultiFab)");
    ncomp = std::min(ncomp, srcmf.nComp());

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    amrex::Print() << " in apply stencil for filter \n";
#ifdef AMREX_USE_OMP
// never runs on GPU since in the else branch of AMREX_USE_GPU
#pragma omp parallel
#endif
    {
        FArrayBox tmpfab;
        for (MFIter mfi(dstmf,true); mfi.isValid(); ++mfi){

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            amrex::Real wt = amrex::second();

            const auto& srcfab = srcmf[mfi];
            auto& dstfab = dstmf[mfi];
            const Box& tbx = mfi.growntilebox();
            const Box& gbx = amrex::grow(tbx,stencil_length_each_dir-1);
            amrex::Print() << " tbx : " << tbx << " gbx : " << gbx << "\n";
            // tmpfab has enough ghost cells for the stencil
            tmpfab.resize(gbx,ncomp);
            tmpfab.setVal(0.0, gbx, 0, ncomp);
            // Copy values in srcfab into tmpfab
            const Box& ibx = gbx & srcfab.box();
            tmpfab.copy(srcfab, ibx, scomp, ibx, 0, ncomp);
            // Apply filter
            DoFilter(tbx, tmpfab.array(), dstfab.array(), 0, dcomp, ncomp);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = amrex::second() - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
            }
        }
    }
}

/* \brief Apply stencil on FArrayBox (CPU version, 2D/3D).
 * \param dstfab Destination FArrayBox
 * \param srcmf source FArrayBox
 * \param tbx Grown box on which srcfab is defined.
 * \param scomp first component of srcfab on which the filter is applied
 * \param dcomp first component of dstfab on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (amrex::FArrayBox& dstfab, const amrex::FArrayBox& srcfab,
                      const amrex::Box& tbx, int scomp, int dcomp, int ncomp)
{
    WARPX_PROFILE("Filter::ApplyStencil(FArrayBox)");
    ncomp = std::min(ncomp, srcfab.nComp());
    FArrayBox tmpfab;
    const Box& gbx = amrex::grow(tbx,stencil_length_each_dir-1);
    // tmpfab has enough ghost cells for the stencil
    tmpfab.resize(gbx,ncomp);
    tmpfab.setVal(0.0, gbx, 0, ncomp);
    // Copy values in srcfab into tmpfab
    const Box& ibx = gbx & srcfab.box();
    tmpfab.copy(srcfab, ibx, scomp, ibx, 0, ncomp);
    // Apply filter
    DoFilter(tbx, tmpfab.array(), dstfab.array(), 0, dcomp, ncomp);
}

void Filter::DoFilter (const Box& tbx,
                       Array4<Real const> const& tmp,
                       Array4<Real      > const& dst,
                       int scomp, int dcomp, int ncomp)
{
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n"; 
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                }
            }
        }
        // 3 nested loop on 3D stencil
        amrex::Print() << " slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int iz=0; iz < slen.z; ++iz){
            for     (int iy=0; iy < slen.y; ++iy){
                for (int ix=0; ix < slen.x; ++ix){
#if (AMREX_SPACEDIM == 3)
                    Real sss = sx[ix]*sy[iy]*sz[iz];
#else
                    Real sss = sx[ix]*sz[iy];
#endif
                    // 3 nested loop on 3D array
                    for         (int k = lo.z; k <= hi.z; ++k) {
                        for     (int j = lo.y; j <= hi.y; ++j) {
                            AMREX_PRAGMA_SIMD
                            for (int i = lo.x; i <= hi.x; ++i) {
#if (AMREX_SPACEDIM == 3)
                                dst(i,j,k,dcomp+n) += sss*(tmp(i-ix,j-iy,k-iz,scomp+n)
                                                          +tmp(i+ix,j-iy,k-iz,scomp+n)
                                                          +tmp(i-ix,j+iy,k-iz,scomp+n)
                                                          +tmp(i+ix,j+iy,k-iz,scomp+n)
                                                          +tmp(i-ix,j-iy,k+iz,scomp+n)
                                                          +tmp(i+ix,j-iy,k+iz,scomp+n)
                                                          +tmp(i-ix,j+iy,k+iz,scomp+n)
                                                          +tmp(i+ix,j+iy,k+iz,scomp+n));
                                //if ( dst(i,j,k,dcomp+n) > 0 || dst(i,j,k,dcomp+n) < 0) {
                                ////amrex::Print() << " ix : " << ix << " iy " << iy << " iz " << iz << "\n";
                                ////amrex::Print() << " sx : " << sx[ix] << " sy[iy] " << sy[iy] << " " <<sz[iz] << "\n";
                                ////amrex::Print() << " sss " << sss << "\n";
                                ////amrex::Print() << " i : " << i << " j " << j <<" k " << k << "\n"; 
                                ////amrex::Print() << " i-ix : " << i-ix << " " << i+ix << "\n";
                                ////amrex::Print() << "dst : " << dst(i,j,k,dcomp+n) << "\n";
                                ////amrex::Print() << " after update : " << dst(i,j,k,dcomp+n) << "\n";
                                //}                                
#else
                                dst(i,j,k,dcomp+n) += sss*(tmp(i-ix,j-iy,k,scomp+n)
                                                          +tmp(i+ix,j-iy,k,scomp+n)
                                                          +tmp(i-ix,j+iy,k,scomp+n)
                                                          +tmp(i+ix,j+iy,k,scomp+n));
#endif
                            }
                        }
                    }
                }
            }
        }
    }
}

#ifdef PULSAR
/* \brief Apply stencil on MultiFab (CPU version, 2D/3D).
 * \param dstmf Destination MultiFab
 * \param srcmf source MultiFab
 * \param[in] lev mesh refinement level
 * \param scomp first component of srcmf on which the filter is applied
 * \param dcomp first component of dstmf on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencilWithConductor (amrex::MultiFab& dstmf, const amrex::MultiFab& srcmf,
                                   const int lev, const amrex::MultiFab& conductormf,
                                   int scomp, int dcomp, int ncomp)
{
    WARPX_PROFILE("Filter::ApplyStencil(MultiFab)");
    ncomp = std::min(ncomp, srcmf.nComp());

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    amrex::Print() << " in apply stencil for filter with conductor \n";
    amrex::Print() << " dst bA : " << dstmf.boxArray() << "\n";
    amrex::Print() << " src bA : " << srcmf.boxArray() << "\n";
    amrex::Print() << " conductor bA : " << conductormf.boxArray() << "\n";
    amrex::Print() << " nodal flag : " << srcmf.ixType().toIntVect() << "\n";
    amrex::IntVect src_iv = srcmf.ixType().toIntVect();
    int comp_dir = 0;
    if (src_iv[0] == 0) {
        comp_dir = 0;
    } else if (src_iv[1] == 0) {
        comp_dir = 1;
    } else if (src_iv[2] == 0) {
        comp_dir = 2;
    }
    
#ifdef AMREX_USE_OMP
// never runs on GPU since in the else branch of AMREX_USE_GPU
#pragma omp parallel
#endif
    {
        FArrayBox tmpfab;
        for (MFIter mfi(dstmf,false); mfi.isValid(); ++mfi){

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            amrex::Real wt = amrex::second();

            auto& conductorfab = conductormf[mfi];
            const auto& srcfab = srcmf[mfi];
            auto& dstfab = dstmf[mfi];
            const Box& tbx = mfi.growntilebox();
            const Box& gbx = amrex::grow(tbx,stencil_length_each_dir-1);
            amrex::Print() << " tbx : " << tbx << " gbx : " << gbx << "\n";
            // tmpfab has enough ghost cells for the stencil
            tmpfab.resize(gbx,ncomp);
            tmpfab.setVal(0.0, gbx, 0, ncomp);
            // Copy values in srcfab into tmpfab
            const Box& ibx = gbx & srcfab.box();
            tmpfab.copy(srcfab, ibx, scomp, ibx, 0, ncomp);
            // Apply filter
//            DoFilter(tbx, tmpfab.array(), dstfab.array(), 0, dcomp, ncomp);
            for (int idim = 0; idim < 3; ++idim) {
                if (idim == 0) {
                    if (comp_dir == 0) {
                        amrex::Print() << " jx xpass \n";
                        DoJxFilterxpass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                    if (comp_dir == 1) {
                        amrex::Print() << " jy xpass \n";
                        DoJyFilterxpass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                    if (comp_dir == 2) {
                        amrex::Print() << " jz xpass \n";
                        DoJzFilterxpass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                } else if (idim == 1) {
                    if (comp_dir == 0) {
                        amrex::Print() << " jx ypass \n";
                        DoJxFilterypass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                    if (comp_dir == 1) {
                        amrex::Print() << " jy ypass \n";
                        DoJyFilterypass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                    if (comp_dir == 2) {
                        amrex::Print() << " jz ypass \n";
                        DoJzFilterypass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                } else if (idim == 2) {
                    if (comp_dir == 0) {
                        amrex::Print() << " jx zpass \n";
                        DoJxFilterzpass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                    if (comp_dir == 1) {
                        amrex::Print() << " jy zpass \n";
                        DoJyFilterzpass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                    if (comp_dir == 2) {
                        amrex::Print() << " jz zpass \n";
                        DoJzFilterzpass(tbx, tmpfab.array(), dstfab.array(),
                                        conductorfab.array(), 0, dcomp, ncomp);
                        const Box& jbx = gbx & dstfab.box();
                        tmpfab.copy(dstfab, jbx, dcomp, jbx, 0, ncomp);
                    }
                }
            }

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = amrex::second() - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
            }
        }
    }
}

void
Filter::DoJxFilterxpass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i+1,j,k) == 0 ) {
                        // check if lhs node is on conductor
                        if (conductor(i,j,k) == 1) {
                            // jx is normal to conductor plane. Therefore, mirror the current such that
                            // Jx(i-1,j,k) = Jx(i,j,k). Thus adding tmp(i,j,k) directly here, without modifying the value
                            // stored in tmp(i-1,j,k)
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i, j, k, scomp+n);
                            amrex::Print() << " normal to conductor on left " << i << "\n";
                        } else {
                            // not across a conductor. Therefore use deposited value 
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i-1, j, k, scomp+n);
                        }
                        // check if rhs node is on conductor
                        if (conductor(i+1,j,k) == 1) {
                            // jx is normal to conductor plane. Therefore mirror the current such that
                            // Jx(i+1,j,k) = Jx(i,j,k). Thus adding tmp(i,j,k) directly here without modifying the value
                            // stored in tmp(i+1,j,k)
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i, j, k, scomp+n);
                            amrex::Print() << " normal to conductor on right " << i << "\n";
                        } else {
                            // not across a conductor. Use deposited value as usual from rhs
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i+1, j, k, scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sx[0] * tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sx[0] * tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jx\n";
                    }                     
                }
            }
        }
            

    } // loop over ncomp
 
}

void
Filter::DoJxFilterypass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jx ypass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i+1,j,k) == 0 ) {
                        // Check if the edge neigbhor in the negative y-direction is in the conductor
                        if ( conductor(i, j-1, k) == 1 and conductor(i+1, j-1, k) == 1) {
                            // There is a conductor at (i, j-1, k), and therefore an image at j-2, resulting in
                            // zero current at (i, j-1, k)
                            dst(i,j,k,dcomp+n) += 0.0;
                            amrex::Print() << " tangential component jx is zero at j-1, jx-ypass (" << i << ", " << j-1 << ", " << k << ")\n"; 
                        } else {
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i, j-1, k, scomp+n);
                        }
                        // Check if the edge neigbhor in the positive y-direction is in the conductor
                        if ( conductor (i, j+1, k) == 1 and conductor(i+1, j+1, k) == 1) {
                            //There is a conductor at (i, j+1, k) and therefore an image at j+2, resulting in
                            // zero current at (i, j+1, k)
                            dst(i,j,k,dcomp+n) += 0.0;
                            amrex::Print() << " tangential component jx is zero at j+1, jx-ypass (" << i << ", " << j+1 << ", " << k << ")\n"; 
                        } else {
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i, j+1, k, scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sx[0] * tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sx[0] * tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jx ypass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}

void
Filter::DoJxFilterzpass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jx zpass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i+1,j,k) == 0 ) {
                        // Check if neighboring edge in negative z-direction is in a conductor
                        if ( conductor(i, j, k-1) == 1 and conductor(i+1, j, k-1) == 1 ) {
                            // There is a conductor at (i, j, k-1), and therefore an image at k-2, resulting in
                            // zero current at (i, j, k-1)
                            dst(i,j,k,dcomp+n) += 0.0;
                            amrex::Print() << " current i,j,k(" << i << "," << j  << "," << k << ") has k-1 neighbor on conductor" << k-1 << "\n"; 
                        } else {
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i, j, k-1, scomp+n);
                        }
                        // Check if neighboring edge in positive z-direction is in a conductor
                        if ( conductor(i, j, k+1) == 1 and conductor(i+1, j, k+1) == 1 ) {
                            // There is a conductor at (i, j, k+1), and therefore an image at k+2, resulting in
                            // zero current at (i, j, k+1)
                            dst(i,j,k,dcomp+n) += 0.0;
                            amrex::Print() << " current i,j,k(" << i << "," << j  << "," << k << ") has k+1 neighbor on conductor" << k+1 << "\n"; 
                        } else {
                            dst(i,j,k,dcomp+n) += sx[1] * tmp(i, j, k+1, scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sx[0] * tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sx[0] * tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jx zpass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}

// jy xpass
void
Filter::DoJyFilterxpass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jy xpass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx); 
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i,j+1,k) == 0 ) {
                        // Check if neighbor in the negative x-direction is in conductor
                        if ( conductor(i-1, j, k) == 1 and conductor(i-1, j+1, k) ==1 ) {
                            dst(i,j,k,dcomp+n) += 0.;
                            amrex::Print() << " jy is tangential on -x (" << i-1 << "," << j << "," << k << ") adding 0 to i,j,k" << i << " " << j << " " << k << "\n";
                        } else {
                            dst(i,j,k,dcomp+n) += sy[1] * tmp(i-1,j,k,scomp+n);
                        }
                        // Check if neighbor in the positive x-direction is in conductor
                        if ( conductor(i+1, j, k) ==1 and conductor(i+1, j+1, k) ) {
                            dst(i,j,k,dcomp+n) += 0.;
                            amrex::Print() << " jy is tangential on +x (" << i+1 << "," << j << "," << k << ") adding 0 to i,j,k" << i << " " << j << " " << k << "\n";
                        } else {
                            dst(i,j,k,dcomp+n) += sy[1] * tmp(i+1,j,k,scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sy[0] * tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sy[0] * tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jy xpass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}



void
Filter::DoJyFilterypass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jy ypass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx); 
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes in y-direction to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i,j+1,k) == 0 ) {
                        // check if lhs node is on conductor
                        if (conductor(i,j,k) == 1) {
                            // jy is normal to conductor plane. Therefore, mirror the current such that
                            // Jy(i,j-1,k) = Jx(i,j,k). Thus adding tmp(i,j,k) directly here, without modifying the value
                            // stored in tmp(i,j-1,k)
                            dst(i,j,k,dcomp+n) += sy[1] * tmp(i, j, k, scomp+n); 
                            amrex::Print() << " normal to conductor on left for " << j << " at j-1 << " << j-1 << "\n";
                        } else {
                            // not across a conductor. Therefore use deposited value 
                            dst(i,j,k,dcomp+n) += sy[1] * tmp(i, j-1, k, scomp+n);
                        }
                        // check if rhs node is on conductor
                        if (conductor(i,j+1,k) == 1) {
                            // jy is normal to conductor plane. Therefore mirror the current such that
                            // Jy(i,j+1,k) = Jy(i,j,k). Thus adding tmp(i,j,k) directly here without modifying the value
                            // stored in tmp(i,j+1,k)
                            dst(i,j,k,dcomp+n) += sy[1] * tmp(i, j, k, scomp+n);
                            amrex::Print() << " normal to conductor on right for " << j << "at " << j+1 << "\n";
                        } else {
                            // not across a conductor. Use deposited value as usual from rhs
                            dst(i,j,k,dcomp+n) += sy[1] * tmp(i, j+1, k, scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sy[0] * tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sy[0] * tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jy ypass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}

// jy zpass
void
Filter::DoJyFilterzpass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jy zpass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i,j+1,k) == 0 ) {
                        // Check if edge neighbor in negative z-direction is in conductor
                        if ( conductor(i,j,k-1) == 1 and conductor(i,j+1,k-1) == 1 ) {
                            dst(i,j,k,dcomp+n) += 0.;
                            amrex::Print() << " Jy is tangential onconductor at (" << i << "," << j << "," << k-1 << " so using 0 at ("<< i << "," << j << " " << k <<")\n";
                        } else {
                            dst(i,j,k,dcomp+n) += sy[1]*tmp(i,j,k-1,scomp+n);
                        }
                        if ( conductor(i,j,k+1) == 1 and conductor(i,j+1,k+1) == 1 ) {
                            dst(i,j,k,dcomp+n) += 0.;
                            amrex::Print() << " Jy is tangential onconductor at (" << i << "," << j << "," << k+1 << " so using 0 at ("<< i << "," << j << " " << k <<")\n";
                        } else {
                            dst(i,j,k,dcomp+n) += sy[1]*tmp(i,j,k+1,scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sy[0] * tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sy[0] * tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jy zpass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}

// jz xpass
void
Filter::DoJzFilterxpass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jz xpass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes in y-direction to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i,j,k+1) == 0 ) {
                        //Check if edge neighbor in negative x-direction is in conductor
                        if ( conductor(i-1,j,k) == 1 and conductor(i-1,j,k+1) == 1 ) {
                            // Jz is tangential on conductor at (i-1,j,k). Therefore there is an image current at i-2,
                            // resulting in zero value for Jz at (i-1,j,k). We use 0 here instead of modifying the value
                            // stored in tmp(i-1,j,k)
                            dst(i,j,k,dcomp+n) += 0.0;
                            amrex::Print() << " i-1 neighbor for Jz is on conductor there adding 0 to (" << i << "," << j << "," << k << ")\n";  
                        } else {
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i-1,j,k,scomp+n);
                        }
                        //Check if edge neighbor in positive x-direction is in conductor
                        if ( conductor(i+1,j,k) == 1 and conductor(i+1,j,k+1) == 1 ) {
                            // Jz is tangential on conductor at (i+1,j,k). Therefore there is an image current at i+2,
                            // resulting in zero value for Jz at (i+1,j,k). We use 0 here instead of modifying the value
                            // stored in tmp(i+1,j,k)
                            dst(i,j,k,dcomp+n) += 0.0;
                            amrex::Print() << " i+1 neighbor for Jz is on conductor there adding 0 to (" << i << "," << j << "," << k << ")\n";  
                        } else {
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i+1,j,k,scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sz[0]*tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sz[0]*tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jz xpass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}

// jz ypass
void
Filter::DoJzFilterypass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jz ypass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx); 
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes in y-direction to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i,j,k+1) == 0 ) {
                        // Check if edge neighbor in the negative y-direction is on conductor
                        if ( conductor(i,j-1,k) == 1 and conductor(i,j-1,k+1) == 1 ) {
                            dst(i,j,k,dcomp+n) += 0.;
                            amrex::Print() << " j-1 neighbor for Jz is on conductor there adding 0 to (" << i << "," << j << "," << k << ")\n";  
                        } else {
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i,j-1,k,scomp+n);
                        }
                        // Check if edge neighbor in the positive y-direction is on conductor
                        if ( conductor(i,j+1,k) == 1 and conductor(i,j+1,k+1) == 1 ) {
                            dst(i,j,k,dcomp+n) += 0.;
                            amrex::Print() << " j+1 neighbor for Jz is on conductor there adding 0 to (" << i << "," << j << "," << k << ")\n";  
                        } else {
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i,j+1,k,scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sz[0]*tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sz[0]*tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jz ypass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}



// jz zpass
void
Filter::DoJzFilterzpass (const Box& tbx,
                        amrex::Array4<amrex::Real const> const& tmp,
                        amrex::Array4<amrex::Real      > const& dst,
                        amrex::Array4<amrex::Real const> const& conductor,
                        int scomp, int dcomp, int ncomp)
{
    amrex::Print() << " in jz zpass \n";
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    amrex::Print() << " lo : " << lo << " hi " << hi << "\n";
    // tmp and dst are of type Array4 (Fortran ordering)
    amrex::Real const* AMREX_RESTRICT sx = stencil_x.data();
#if (AMREX_SPACEDIM == 3)
    amrex::Real const* AMREX_RESTRICT sy = stencil_y.data();
#endif
    amrex::Real const* AMREX_RESTRICT sz = stencil_z.data();
    amrex::Print() << " sx : " << sx[0] << " " << sx[1] << "\n";
    amrex::Print() << " sy : " << sy[0] << " " << sy[1] << "\n";
    amrex::Print() << " sz : " << sz[0] << " " << sz[1] << "\n";

    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                } // loop over i
            } // loop over j
        } // loop over k

        amrex::Print() << "slen : " << slen.x << " " << slen.y << " " << slen.z << "\n";
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Check if (i,j,k) edge center is NOT on conductor by checking the
                    // neighboring nodes in y-direction to see if atleast one node is not conductor.
                    // Since if edge center is on conductor, we do not modify the value.
                    if ( conductor(i,j,k) == 0 || conductor(i,j,k+1) == 0 ) {
                        // check if lhs node is on conductor
                        if ( conductor(i,j,k) == 1) {
                            // Jz is normal to conductor plane therefore using mirror current such that
                            // Jz(i,j,k-1) = Jz(i,j,k) without modifying the value stored in tmp(i,j,k-1)
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i,j,k,scomp+n);
                            amrex::Print() << " Jz is normal to conductor plane at (" << i << "," << j << "," << k << ") instead of using value at k-1 " << k-1 << "\n";
                        } else {
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i,j,k-1,scomp+n);
                        }
                        // check if rhs node is on conductor
                        if ( conductor(i,j,k+1) == 1) {
                            // Jz is normal to conductor plane therefore using mirror current such that
                            // Jz(i,j,k+11) = Jz(i,j,k) without modifying the value stored in tmp(i,j,k+1)
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i,j,k,scomp+n);
                            amrex::Print() << " Jz is normal to conductor plane at (" << i << "," << j << "," << k << ") instead of using value at k+1 " << k+1 << "\n";
                        } else {
                            dst(i,j,k,dcomp+n) += sz[1]*tmp(i,j,k+1,scomp+n);
                        }
                        // add 0.5* times the value at (i,j,k) based on bilinear stencil
                        // Since sx[0] = 0.25, adding twice here aligned with WarpX implementation
                        dst(i,j,k,dcomp+n) += sz[0] * tmp(i,j,k,scomp+n);
                        dst(i,j,k,dcomp+n) += sz[0] * tmp(i,j,k,scomp+n);
                    } else {
                        dst(i,j,k,dcomp+n) = tmp(i,j,k,scomp+n);
                        amrex::Print() << " ( " << i << "," << j << "," << k <<") is on conductor for jz zpass\n";
                    }
                }
            }
        }
    } // loop over ncomp
}


#endif

#endif // #ifdef AMREX_USE_CUDA
