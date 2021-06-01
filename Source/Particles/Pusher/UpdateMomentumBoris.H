/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_

#include <AMReX_REAL.H>

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBoris(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::Real q, const amrex::Real m, const amrex::Real dt, amrex::Real * PulsarDiag = nullptr, const int i=0, const int compute_from_theory = 0 )
{
    using namespace amrex::literals;

    const amrex::Real econst = 0.5_rt*q*dt/m;
    amrex::Real ux_theory = ux;
    amrex::Real uy_theory = uy;
    amrex::Real uz_theory = uz;

    // First half-push for E
    if (compute_from_theory == 0) {
        ux += econst*Ex;
        uy += econst*Ey;
        uz += econst*Ez;
    } else {
        ux_theory += econst*Ex;
        uy_theory += econst*Ey;
        uz_theory += econst*Ez;
    }
#ifdef PULSAR
//    amrex::AllPrintToFile("PulsarParticle") << "half acceleration Ex : " << econst*Ex << " " << " Ey : " << econst*Ey << " Ez " << econst*Ez << "\n";
    if (i == 0) {
        if (compute_from_theory == 0) {
            PulsarDiag[31] = 2.*econst*Ex;
            PulsarDiag[32] = 2.*econst*Ey;
            PulsarDiag[33] = 2.*econst*Ez;
        } else {
            PulsarDiag[43] = 2.*econst*Ex;
            PulsarDiag[44] = 2.*econst*Ey;
            PulsarDiag[45] = 2.*econst*Ez;
        }
    }
#endif
    // Compute temporary gamma factor
    constexpr amrex::Real inv_c2 = 1._rt/(PhysConst::c*PhysConst::c);
    const amrex::Real inv_gamma = 1._rt/std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    // Magnetic rotation
    // - Compute temporary variables
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
    amrex::Real tmp_ux = uy_p*sz - uz_p*sy;
    amrex::Real tmp_uy = uz_p*sx - ux_p*sz;
    amrex::Real tmp_uz = ux_p*sy - uy_p*sx;
#ifdef PULSAR
//    amrex::AllPrintToFile("PulsarParticle") << " Momentum after rotation : " << ux << " " << " uy : " << uy << " uz " << uz << " rot ux : " << tmp_ux << " rot uy " << tmp_uy << " rot uz "  << tmp_uz<< "\n";
    if (i == 0) {
        if (compute_from_theory == 0) {
            PulsarDiag[34] = tmp_ux;
            PulsarDiag[35] = tmp_uy;
            PulsarDiag[36] = tmp_uz;
        } else {
            const amrex::Real ux_p_theory = ux_theory + uy_theory*tz - uz_theory*ty;
            const amrex::Real uy_p_theory = uy_theory + uz_theory*tx - ux_theory*tz;
            const amrex::Real uz_p_theory = uz_theory + ux_theory*ty - uy_theory*tx;
            amrex::Real tmp_ux_theory = uy_p_theory*sz - uz_p_theory*sy;
            amrex::Real tmp_uy_theory = uz_p_theory*sx - ux_p_theory*sz;
            amrex::Real tmp_uz_theory = ux_p_theory*sy - uy_p_theory*sx;
            PulsarDiag[46] = tmp_ux_theory;
            PulsarDiag[47] = tmp_uy_theory;
            PulsarDiag[48] = tmp_uz_theory;
        }
    }
#endif
    // Second half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;
#ifdef PULSAR
//    amrex::AllPrintToFile("PulsarParticle") << "half acceleration Ex : " << econst*Ex << " " << " Ey : " << econst*Ey << " Ez " << econst*Ez << "\n";
//    amrex::AllPrintToFile("PulsarParticle") << " Final ux : " << ux << " " << " uy : " << uy << " uz " << uz << "\n";
#endif
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
