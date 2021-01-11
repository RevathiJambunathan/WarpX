/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GATHEREXTERNALPULSARFIELDONGRID_H_
#define GATHEREXTERNALPULSARFIELDONGRID_H_

#ifdef PULSAR
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/ShapeFactors.H"
#include "Utils/WarpX_Complex.H"
#include "Particles/PulsarParameters.H"
#include <AMReX.H>


/**
 * Returns the x/y/z component of the E/B applied field for pulsar set-up.
 *
 * \param ii, jj, kk : indices in x, y, z
 * \param mf_type    : staggered location of the field
 * \param domain_lo  : physical coordinates of the lower corner of the domain
 * \param domain_hi  : physical coordinates of the upper corner of the domain
 * \param dx         : cell size in x, y, z
 * \param cur_time   : the current physical time of the simulation
 * \param Field     : 0 for Bfield, 1 for Efield
 * \param comp       : x(ncomp=0), y(ncomp=1), or z(ncomp=2) component
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real GetExternalFieldAtCellLocation(int ii, int jj, int kk,
        amrex::GpuArray<int, 3> const f_type,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_lo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_hi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
        amrex::Real cur_time, int Field, int comp)
{
    amrex::Real xcell, ycell, zcell;
    amrex::Real r, theta, phi;
    amrex::Real Fr, Ftheta, Fphi;
    amrex::Real Fext_comp;
    // compute cell coordinates, (x,y,z) corresponding to (ii,jj,kk)
    // based on the staggered Bx location on the yee-grid
    PulsarParm::ComputeCellCoordinates( ii, jj, kk, f_type, domain_lo,
                                        dx, xcell, ycell, zcell );
    // Convert (x,y,z) to (r,theta,phi)
    PulsarParm::ConvertCartesianToSphericalCoord( xcell, ycell, zcell,
                                                  domain_lo, domain_hi,
                                                  r, theta, phi);
    if (Field == 0) {
        // Compute External vacuum fields of the pulsar (Br, Btheta, Bphi)
        PulsarParm::ExternalBFieldSpherical( r, theta, phi, cur_time,
                                             Fr, Ftheta, Fphi);
    } else if (Field == 1) {
        // Compute External vacuum fields of the pulsar (Br, Btheta, Bphi)
        PulsarParm::ExternalEFieldSpherical( r, theta, phi, cur_time,
                                             Fr, Ftheta, Fphi);
    }
    
    if (comp == 0) {
        // Convert (Fr, Ftheta, Fphi) to Xcomponent
        PulsarParm::ConvertSphericalToCartesianXComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Fext_comp);
    } else if (comp == 1) {
        // Convert (Fr, Ftheta, Fphi) to YComponent
        PulsarParm::ConvertSphericalToCartesianYComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Fext_comp);
    } else if (comp == 2) {
        // Convert (Fr, Ftheta, Fphi) to ZComponent
        PulsarParm::ConvertSphericalToCartesianZComponent( Fr, Ftheta, Fphi,
                                                           r, theta, phi, Fext_comp);
    }
    
    return Fext_comp;
}


/**
 * \brief Field gather for a single particle
 *
 * \tparam depos_order              Particle shape order
 * \tparam galerkin_interpolation   Lower the order of the particle shape by
 *                                  this value (0/1) for the parallel field component
 * \param xp, yp, zp                Particle position coordinates
 * \param Exp, Eyp, Ezp             Electric field on particles.
 * \param Bxp, Byp, Bzp             Magnetic field on particles.
 * \param ex_arr ey_arr ez_arr      Array4 of the electric field, either full array or tile.
 * \param bx_arr by_arr bz_arr      Array4 of the magnetic field, either full array or tile.
 * \param ex_type, ey_type, ez_type IndexType of the electric field
 * \param bx_type, by_type, bz_type IndexType of the magnetic field
 * \param dx                        3D cell spacing
 * \param xyzmin                    Physical lower bounds of domain in x, y, z.
 * \param lo                        Index lower bounds of domain.
 * \param n_rz_azimuthal_modes       Number of azimuthal modes when using RZ geometry
 */
template <int depos_order, int galerkin_interpolation>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doPulsarFieldGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& ex_arr,
                     amrex::Array4<amrex::Real const> const& ey_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::IndexType ex_type,
                     const amrex::IndexType ey_type,
                     const amrex::IndexType ez_type,
                     const amrex::IndexType bx_type,
                     const amrex::IndexType by_type,
                     const amrex::IndexType bz_type,
                     const amrex::GpuArray<amrex::Real, 3>& dx,
                     const amrex::GpuArray<amrex::Real, 3>& xyzmin,
                     const amrex::Dim3& lo,
                     const long n_rz_azimuthal_modes,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dom_lo = {},
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dom_hi = {},
                     amrex::Real cur_time = 0._rt )
{
    using namespace amrex;

#if defined(WARPX_DIM_XZ)
    amrex::Abort("pulsar does not work in 2D");
#endif

#ifndef WARPX_DIM_RZ
    amrex::Abort("pulsar does not work in RZ");
#endif

    const amrex::Real dxi = 1.0_rt/dx[0];
    const amrex::Real dyi = 1.0_rt/dx[1];
    const amrex::Real dzi = 1.0_rt/dx[2];

    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];
    const amrex::Real zmin = xyzmin[2];

    constexpr int zdir = (AMREX_SPACEDIM - 1);
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // store staggering of ex-bz in GpuArray
    amrex::GpuArray<int, 3> Ex_stag, Ey_stag, Ez_stag, Bx_stag, By_stag, Bz_stag;
    for (int idim = 0; idim < 3; ++idim)
    {
        Ex_stag[idim] = ex_type[idim];
        Ey_stag[idim] = ey_type[idim];
        Ez_stag[idim] = ez_type[idim];
        Bx_stag[idim] = bx_type[idim];
        By_stag[idim] = by_type[idim];
        Bz_stag[idim] = bz_type[idim];
    }
    // --- Compute shape factors
    // x direction
    // Get particle position
    const amrex::Real x = (xp-xmin)*dxi;

    // j_[eb][xyz] leftmost grid point in x that the particle touches for the centering of each current
    // sx_[eb][xyz] shape factor along x for the centering of each current
    // There are only two possible centerings, node or cell centered, so at most only two shape factor
    // arrays will be needed.
    amrex::Real sx_node[depos_order + 1];
    amrex::Real sx_cell[depos_order + 1];
    amrex::Real sx_node_galerkin[depos_order + 1 - galerkin_interpolation] = {0._rt};
    amrex::Real sx_cell_galerkin[depos_order + 1 - galerkin_interpolation] = {0._rt};

    int j_node = 0;
    int j_cell = 0;
    int j_node_v = 0;
    int j_cell_v = 0;
    Compute_shape_factor< depos_order > const compute_shape_factor;
    Compute_shape_factor<depos_order - galerkin_interpolation > const compute_shape_factor_galerkin;
    if ((ey_type[0] == NODE) || (ez_type[0] == NODE) || (bx_type[0] == NODE)) {
        j_node = compute_shape_factor(sx_node, x);
    }
    if ((ey_type[0] == CELL) || (ez_type[0] == CELL) || (bx_type[0] == CELL)) {
        j_cell = compute_shape_factor(sx_cell, x - 0.5_rt);
    }
    if ((ex_type[0] == NODE) || (by_type[0] == NODE) || (bz_type[0] == NODE)) {
        j_node_v = compute_shape_factor_galerkin(sx_node_galerkin, x);
    }
    if ((ex_type[0] == CELL) || (by_type[0] == CELL) || (bz_type[0] == CELL)) {
        j_cell_v = compute_shape_factor_galerkin(sx_cell_galerkin, x - 0.5_rt);
    }
    const amrex::Real (&sx_ex)[depos_order + 1 - galerkin_interpolation] = ((ex_type[0] == NODE) ? sx_node_galerkin : sx_cell_galerkin);
    const amrex::Real (&sx_ey)[depos_order + 1             ] = ((ey_type[0] == NODE) ? sx_node   : sx_cell  );
    const amrex::Real (&sx_ez)[depos_order + 1             ] = ((ez_type[0] == NODE) ? sx_node   : sx_cell  );
    const amrex::Real (&sx_bx)[depos_order + 1             ] = ((bx_type[0] == NODE) ? sx_node   : sx_cell  );
    const amrex::Real (&sx_by)[depos_order + 1 - galerkin_interpolation] = ((by_type[0] == NODE) ? sx_node_galerkin : sx_cell_galerkin);
    const amrex::Real (&sx_bz)[depos_order + 1 - galerkin_interpolation] = ((bz_type[0] == NODE) ? sx_node_galerkin : sx_cell_galerkin);
    int const j_ex = ((ex_type[0] == NODE) ? j_node_v : j_cell_v);
    int const j_ey = ((ey_type[0] == NODE) ? j_node   : j_cell  );
    int const j_ez = ((ez_type[0] == NODE) ? j_node   : j_cell  );
    int const j_bx = ((bx_type[0] == NODE) ? j_node   : j_cell  );
    int const j_by = ((by_type[0] == NODE) ? j_node_v : j_cell_v);
    int const j_bz = ((bz_type[0] == NODE) ? j_node_v : j_cell_v);

    // y direction
    const amrex::Real y = (yp-ymin)*dyi;
    amrex::Real sy_node[depos_order + 1];
    amrex::Real sy_cell[depos_order + 1];
    amrex::Real sy_node_v[depos_order + 1 - galerkin_interpolation];
    amrex::Real sy_cell_v[depos_order + 1 - galerkin_interpolation];
    int k_node = 0;
    int k_cell = 0;
    int k_node_v = 0;
    int k_cell_v = 0;
    if ((ex_type[1] == NODE) || (ez_type[1] == NODE) || (by_type[1] == NODE)) {
        k_node = compute_shape_factor(sy_node, y);
    }
    if ((ex_type[1] == CELL) || (ez_type[1] == CELL) || (by_type[1] == CELL)) {
        k_cell = compute_shape_factor(sy_cell, y - 0.5_rt);
    }
    if ((ey_type[1] == NODE) || (bx_type[1] == NODE) || (bz_type[1] == NODE)) {
        k_node_v = compute_shape_factor_galerkin(sy_node_v, y);
    }
    if ((ey_type[1] == CELL) || (bx_type[1] == CELL) || (bz_type[1] == CELL)) {
        k_cell_v = compute_shape_factor_galerkin(sy_cell_v, y - 0.5_rt);
    }
    const amrex::Real (&sy_ex)[depos_order + 1             ] = ((ex_type[1] == NODE) ? sy_node   : sy_cell  );
    const amrex::Real (&sy_ey)[depos_order + 1 - galerkin_interpolation] = ((ey_type[1] == NODE) ? sy_node_v : sy_cell_v);
    const amrex::Real (&sy_ez)[depos_order + 1             ] = ((ez_type[1] == NODE) ? sy_node   : sy_cell  );
    const amrex::Real (&sy_bx)[depos_order + 1 - galerkin_interpolation] = ((bx_type[1] == NODE) ? sy_node_v : sy_cell_v);
    const amrex::Real (&sy_by)[depos_order + 1             ] = ((by_type[1] == NODE) ? sy_node   : sy_cell  );
    const amrex::Real (&sy_bz)[depos_order + 1 - galerkin_interpolation] = ((bz_type[1] == NODE) ? sy_node_v : sy_cell_v);
    int const k_ex = ((ex_type[1] == NODE) ? k_node   : k_cell  );
    int const k_ey = ((ey_type[1] == NODE) ? k_node_v : k_cell_v);
    int const k_ez = ((ez_type[1] == NODE) ? k_node   : k_cell  );
    int const k_bx = ((bx_type[1] == NODE) ? k_node_v : k_cell_v);
    int const k_by = ((by_type[1] == NODE) ? k_node   : k_cell  );
    int const k_bz = ((bz_type[1] == NODE) ? k_node_v : k_cell_v);

    // z direction
    const amrex::Real z = (zp-zmin)*dzi;
    amrex::Real sz_node[depos_order + 1];
    amrex::Real sz_cell[depos_order + 1];
    amrex::Real sz_node_v[depos_order + 1 - galerkin_interpolation];
    amrex::Real sz_cell_v[depos_order + 1 - galerkin_interpolation];
    int l_node = 0;
    int l_cell = 0;
    int l_node_v = 0;
    int l_cell_v = 0;
    if ((ex_type[zdir] == NODE) || (ey_type[zdir] == NODE) || (bz_type[zdir] == NODE)) {
        l_node = compute_shape_factor(sz_node, z);
    }
    if ((ex_type[zdir] == CELL) || (ey_type[zdir] == CELL) || (bz_type[zdir] == CELL)) {
        l_cell = compute_shape_factor(sz_cell, z - 0.5_rt);
    }
    if ((ez_type[zdir] == NODE) || (bx_type[zdir] == NODE) || (by_type[zdir] == NODE)) {
        l_node_v = compute_shape_factor_galerkin(sz_node_v, z);
    }
    if ((ez_type[zdir] == CELL) || (bx_type[zdir] == CELL) || (by_type[zdir] == CELL)) {
        l_cell_v = compute_shape_factor_galerkin(sz_cell_v, z - 0.5_rt);
    }
    const amrex::Real (&sz_ex)[depos_order + 1             ] = ((ex_type[zdir] == NODE) ? sz_node   : sz_cell  );
    const amrex::Real (&sz_ey)[depos_order + 1             ] = ((ey_type[zdir] == NODE) ? sz_node   : sz_cell  );
    const amrex::Real (&sz_ez)[depos_order + 1 - galerkin_interpolation] = ((ez_type[zdir] == NODE) ? sz_node_v : sz_cell_v);
    const amrex::Real (&sz_bx)[depos_order + 1 - galerkin_interpolation] = ((bx_type[zdir] == NODE) ? sz_node_v : sz_cell_v);
    const amrex::Real (&sz_by)[depos_order + 1 - galerkin_interpolation] = ((by_type[zdir] == NODE) ? sz_node_v : sz_cell_v);
    const amrex::Real (&sz_bz)[depos_order + 1             ] = ((bz_type[zdir] == NODE) ? sz_node   : sz_cell  );
    int const l_ex = ((ex_type[zdir] == NODE) ? l_node   : l_cell  );
    int const l_ey = ((ey_type[zdir] == NODE) ? l_node   : l_cell  );
    int const l_ez = ((ez_type[zdir] == NODE) ? l_node_v : l_cell_v);
    int const l_bx = ((bx_type[zdir] == NODE) ? l_node_v : l_cell_v);
    int const l_by = ((by_type[zdir] == NODE) ? l_node_v : l_cell_v);
    int const l_bz = ((bz_type[zdir] == NODE) ? l_node   : l_cell  );


    // Each field is gathered in a separate block of
    // AMREX_SPACEDIM nested loops because the deposition
    // order can differ for each component of each field
    // when galerkin_interpolation is set to 1

    // Gather Pulsar external field using grid resolution on particle Exp
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                // Add the external field contribution for pulsar
                int ii = lo.x + j_ex + ix;
                int jj = lo.y + k_ex + iy;
                int kk = lo.z + l_ex + iz;
                int Field = 1; // 1 for Efield, 0 for Bfield
                int comp = 0; // 0 for xcomponent
                amrex::Real ex_ext = GetExternalFieldAtCellLocation( ii, jj, kk,
                                       Ex_stag, dom_lo, dom_hi, dx, 
                                       cur_time, Field, comp);
                Exp += sx_ex[ix]*sy_ex[iy]*sz_ex[iz]*ex_ext;
            }
        }
    }
    // Gather Pulsar external field using grid resolution on particle Eyp
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                // Add the external field contribution for pulsar
                int ii = lo.x + j_ey + ix;
                int jj = lo.y + k_ey + iy;
                int kk = lo.z + l_ey + iz;
                int Field = 1; // 1 for Efield, 0 for Bfield
                int comp = 1; // 1 for ycomponent
                amrex::Real ey_ext = GetExternalFieldAtCellLocation( ii, jj, kk,
                                       Ey_stag, dom_lo, dom_hi, dx, 
                                       cur_time, Field, comp);
                Eyp += sx_ey[ix]*sy_ey[iy]*sz_ey[iz]*ey_ext;
            }
        }
    }
    // Gather Pulsar external field using grid resolution on particle Ezp
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                // Add the external field contribution for pulsar
                int ii = lo.x + j_ez + ix;
                int jj = lo.y + k_ez + iy;
                int kk = lo.z + l_ez + iz;
                int Field = 1; // 1 for Efield, 0 for Bfield
                int comp = 2; // 2 for zcomponent
                amrex::Real ez_ext = GetExternalFieldAtCellLocation( ii, jj, kk,
                                       Ez_stag, dom_lo, dom_hi, dx, 
                                       cur_time, Field, comp);
                Ezp += sx_ez[ix]*sy_ez[iy]*sz_ez[iz]*ez_ext;
            }
        }
    }
    // Gather Pulsar external field using grid resolution on particle Bzp
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                // Add the external field contribution for pulsar
                int ii = lo.x + j_bz + ix;
                int jj = lo.y + k_bz + iy;
                int kk = lo.z + l_bz + iz;
                int Field = 0; // 1 for Efield, 0 for Bfield
                int comp = 2; // 2 for zcomponent
                amrex::Real bz_ext = GetExternalFieldAtCellLocation( ii, jj, kk,
                                       Bz_stag, dom_lo, dom_hi, dx, 
                                       cur_time, Field, comp);
                Bzp += sx_bz[ix]*sy_bz[iy]*sz_bz[iz]*bz_ext;
            }
        }
    }
    // Gather Pulsar external field using grid resolution on particle Byp
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                // Add the external field contribution for pulsar
                int ii = lo.x + j_by + ix;
                int jj = lo.y + k_by + iy;
                int kk = lo.z + l_by + iz;
                int Field = 0; // 1 for Efield, 0 for Bfield
                int comp = 1; // 1 for ycomponent
                amrex::Real by_ext = GetExternalFieldAtCellLocation( ii, jj, kk,
                                       By_stag, dom_lo, dom_hi, dx, 
                                       cur_time, Field, comp);
                Byp += sx_by[ix]*sy_by[iy]*sz_by[iz]*by_ext;
            }
        }
    }
    // Gather Pulsar external field using grid resolution on particle Bxp
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                // Add the external field contribution for pulsar
                int ii = lo.x + j_bx + ix;
                int jj = lo.y + k_bx + iy;
                int kk = lo.z + l_bx + iz;
                int Field = 0; // 1 for Efield, 0 for Bfield
                int comp = 0; // 0 for xcomponent
                amrex::Real bx_ext = GetExternalFieldAtCellLocation( ii, jj, kk,
                                       Bx_stag, dom_lo, dom_hi, dx, 
                                       cur_time, Field, comp);
                Bxp += sx_bx[ix]*sy_bx[iy]*sz_bx[iz]*bx_ext;
            }
        }
    }
}




/**
 * \brief Field gather for particles
 *
 * /param getPosition          : A functor for returning the particle position.
 * /param getExternalEField    : A functor for assigning the external E field.
 * /param getExternalBField    : A functor for assigning the external B field.
 * \param Exp, Eyp, Ezp        : Pointer to array of electric field on particles.
 * \param Bxp, Byp, Bzp        : Pointer to array of magnetic field on particles.
 * \param exfab eyfab ezfab    : Array4 of the electric field, either full array or tile.
 * \param ezfab bxfab bzfab    : Array4 of the magnetic field, either full array or tile.
 * \param np_to_gather         : Number of particles for which field is gathered.
 * \param dx                   : 3D cell size
 * \param xyzmin               : Physical lower bounds of domain.
 * \param lo                   : Index lower bounds of domain.
 * \param n_rz_azimuthal_modes : Number of azimuthal modes when using RZ geometry
 */
template <int depos_order, int lower_in_v>
void doPulsarFieldGatherShapeN(const GetParticlePosition& getPosition,
                    const GetExternalEField& getExternalE, const GetExternalBField& getExternalB,
                    amrex::ParticleReal * const Exp, amrex::ParticleReal * const Eyp,
                    amrex::ParticleReal * const Ezp, amrex::ParticleReal * const Bxp,
                    amrex::ParticleReal * const Byp, amrex::ParticleReal * const Bzp,
                    amrex::FArrayBox const * const exfab,
                    amrex::FArrayBox const * const eyfab,
                    amrex::FArrayBox const * const ezfab,
                    amrex::FArrayBox const * const bxfab,
                    amrex::FArrayBox const * const byfab,
                    amrex::FArrayBox const * const bzfab,
                    const long np_to_gather,
                    const std::array<amrex::Real, 3>& dx,
                    const std::array<amrex::Real, 3>& xyzmin,
                    const amrex::Dim3 lo,
                    const long n_rz_azimuthal_modes,
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dom_lo = {},
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dom_hi = {},
                    amrex::Real cur_time = 0._rt )
{

    amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
    amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};

    amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    // Loop over particles and gather fields from
    // {e,b}{x,y,z}_arr to {E,B}{xyz}p.
    amrex::ParallelFor(
        np_to_gather,
        [=] AMREX_GPU_DEVICE (long ip) {

            amrex::ParticleReal xp, yp, zp;
            getPosition(ip, xp, yp, zp);
            getExternalE(ip, Exp[ip], Eyp[ip], Ezp[ip]);
            getExternalB(ip, Bxp[ip], Byp[ip], Bzp[ip]);

            doPulsarFieldGatherShapeN<depos_order, lower_in_v>(
                xp, yp, zp, Exp[ip], Eyp[ip], Ezp[ip], Bxp[ip], Byp[ip], Bzp[ip],
                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes,
                dom_lo, dom_hi, cur_time);
        });
}

/**
 * \brief Field gather for a single particle
 *
 * /param xp, yp, zp                : Particle position coordinates
 * \param Exp, Eyp, Ezp             : Electric field on particles.
 * \param Bxp, Byp, Bzp             : Magnetic field on particles.
 * \param ex_arr ey_arr ez_arr      : Array4 of the electric field, either full array or tile.
 * \param bx_arr by_arr bz_arr      : Array4 of the magnetic field, either full array or tile.
 * \param ex_type, ey_type, ez_type : IndexType of the electric field
 * \param bx_type, by_type, bz_type : IndexType of the magnetic field
 * \param dx                        : 3D cell spacing
 * \param xyzmin                    : Physical lower bounds of domain in x, y, z.
 * \param lo                        : Index lower bounds of domain.
 * \param n_rz_azimuthal_modes      : Number of azimuthal modes when using RZ geometry
 * \param nox                       : order of the particle shape function
 * \param galerkin_interpolation        : whether to use lower order in v
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doPulsarFieldGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& ex_arr,
                     amrex::Array4<amrex::Real const> const& ey_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::IndexType ex_type,
                     const amrex::IndexType ey_type,
                     const amrex::IndexType ez_type,
                     const amrex::IndexType bx_type,
                     const amrex::IndexType by_type,
                     const amrex::IndexType bz_type,
                     const amrex::GpuArray<amrex::Real, 3>& dx_arr,
                     const amrex::GpuArray<amrex::Real, 3>& xyzmin_arr,
                     const amrex::Dim3& lo,
                     const int n_rz_azimuthal_modes,
                     const int nox,
                     const bool galerkin_interpolation,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dom_lo = {},
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dom_hi = {},
                     amrex::Real cur_time = 0._rt )
{
    if (galerkin_interpolation) {
        if (nox == 1) {
            doPulsarFieldGatherShapeN<1,1>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes,
                                dom_lo, dom_hi, cur_time
                                );
        } else if (nox == 2) {
            doPulsarFieldGatherShapeN<2,1>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes,
                                dom_lo, dom_hi, cur_time
                                );
        } else if (nox == 3) {
            doPulsarFieldGatherShapeN<3,1>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes,
                                dom_lo, dom_hi, cur_time
                                );
        }
    } else {
        if (nox == 1) {
            doPulsarFieldGatherShapeN<1,0>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes,
                                dom_lo, dom_hi, cur_time
                                );
        } else if (nox == 2) {
            doPulsarFieldGatherShapeN<2,0>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes,
                                dom_lo, dom_hi, cur_time);
        } else if (nox == 3) {
            doPulsarFieldGatherShapeN<3,0>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes,
                                dom_lo, dom_hi, cur_time);
        }
    }
}

#endif // ifdef pulsar
#endif // GATHEREXTERNALPULSARFIELDONGRID_H_
