/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_For_PULSAR_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_For_PULSAR_H_

#include <AMReX_REAL.H>
#include "Particles/PulsarParameters.H"

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBorisForPulsar(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::Real q, const amrex::Real m, const amrex::Real dt,
    const amrex::ParticleReal& xp, const amrex::ParticleReal& yp, const amrex::ParticleReal& zp )
{
    using namespace amrex::literals;

    const amrex::Real econst = 0.5_rt*q*dt/m;

    // First half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;
    // Compute temporary gamma factor
    constexpr amrex::Real inv_c2 = 1._rt/(PhysConst::c*PhysConst::c);
    const amrex::Real inv_gamma = 1._rt/std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    // Magnetic rotation
    // - Compute temporary variables
    int computemomentum = 1;
    if (PulsarParm::noworkfunction_vcrossB_inRstar == 1) {
        const amrex::Real xc = PulsarParm::center_star[0];
        const amrex::Real yc = PulsarParm::center_star[1];
        const amrex::Real zc = PulsarParm::center_star[2];
        // find radius of the particle
        amrex::Real rp = std::sqrt( (xp-xc)*(xp-xc)
                       + (yp-yc)*(yp-yc) + (zp-zc)*(zp-zc) );
        if ( rp <= PulsarParm::R_star) {
            computemomentum = 0;
        }
    }
    if (computemomentum == 1) {
            const amrex::Real tx = econst*inv_gamma*Bx;
            const amrex::Real ty = econst*inv_gamma*By;
            const amrex::Real tz = econst*inv_gamma*Bz;
            const amrex::Real tsqi = 2._rt/(1._rt + tx*tx + ty*ty + tz*tz);
            const amrex::Real sx = tx*tsqi;
            const amrex::Real sy = ty*tsqi;
            const amrex::Real sz = tz*tsqi;
            const amrex::Real ux_p = ux + uy*tz - uz*ty;
            const amrex::Real uy_p = uy + uz*tx - ux*tz;
            const amrex::Real uz_p = uz + ux*ty - uy*tx;
    
            // - Update momentum
            ux += uy_p*sz - uz_p*sy;
            uy += uz_p*sx - ux_p*sz;
            uz += ux_p*sy - uy_p*sx;
    }
    // Second half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;
    if (PulsarParm::nosolidBodyRotationinRstar == 1) {
        amrex::GpuArray<amrex::Real, 3> const lo(AMREX_D_DECL(0., 0., 0.));
        amrex::GpuArray<amrex::Real, 3> const hi(AMREX_D_DECL(1., 1., 1.));
        amrex::Real r, theta, phi;
        PulsarParm::ConvertCartesianToSphericalCoord (xp, yp, zp, lo, hi, r, theta, phi);
        amrex::Real vphi = (-1.) * r * PulsarParm::omega(t) * std::sin(theta);
        amrex::Real tmpux= (-1.)*std::sin(phi)*vphi;
        amrex::Real tmpuy= std::cos(phi)*vphi;
        amrex::Real gamma = 1._rt/std::sqrt(1._rt - (tmpux*tmpux+tmpuy*tmpuy)*inv_c2);
        ux += gamma * tmpux;
        uy += gamma * tmpuy;
    }
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
