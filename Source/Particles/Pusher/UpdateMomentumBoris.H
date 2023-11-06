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
    const amrex::ParticleReal q, const amrex::ParticleReal m,
#ifdef PULSAR
    const int do_zero_uperpB_driftframe,
    const amrex::ParticleReal r_p,
    const amrex::Real rmax_zero_uperpB_driftframe,
#endif
    const amrex::Real dt, amrex::Real * PulsarDiag = nullptr )
{
    using namespace amrex::literals;

    const amrex::ParticleReal econst = 0.5_prt*q*dt/m;

    // First half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;

//#ifdef PULSAR
//    PulsarDiag[25] = 2.*econst*Ex;
//    PulsarDiag[26] = 2.*econst*Ey;
//    PulsarDiag[27] = 2.*econst*Ez;
//#endif
    // Compute temporary gamma factor
    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    // Magnetic rotation
    // - Compute temporary variables
    const amrex::ParticleReal tx = econst*inv_gamma*Bx;
    const amrex::ParticleReal ty = econst*inv_gamma*By;
    const amrex::ParticleReal tz = econst*inv_gamma*Bz;
    const amrex::ParticleReal tsqi = 2._prt/(1._prt + tx*tx + ty*ty + tz*tz);
    const amrex::ParticleReal sx = tx*tsqi;
    const amrex::ParticleReal sy = ty*tsqi;
    const amrex::ParticleReal sz = tz*tsqi;
    const amrex::ParticleReal ux_p = ux + uy*tz - uz*ty;
    const amrex::ParticleReal uy_p = uy + uz*tx - ux*tz;
    const amrex::ParticleReal uz_p = uz + ux*ty - uy*tx;
    // - Update momentum
    ux += uy_p*sz - uz_p*sy;
    uy += uz_p*sx - ux_p*sz;
    uz += ux_p*sy - uy_p*sx;
//#ifdef PULSAR
//    PulsarDiag[28] = uy_p*sz - uz_p*sy;
//    PulsarDiag[29] = uz_p*sx - ux_p*sz;
//    PulsarDiag[30] = ux_p*sy - uy_p*sx;
//#endif
    // Second half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;


#ifdef PULSAR
    if (do_zero_uperpB_driftframe && (r_p < rmax_zero_uperpB_driftframe) ) {
        // 1(a) Find drift velocity
        amrex::Real B_sq = Bx*Bx + By*By + Bz*Bz;

        amrex::Real E_sq = Ex*Ex + Ey*Ey + Ez*Ez;

        amrex::Real EdotB = Ex*Bx + Ey*By + Ez*Bz;
        amrex::Real EdotB_sq = EdotB*EdotB;

        amrex::Real c2 = PhysConst::c*PhysConst::c;
        amrex::Real Eprime_sq = 2._prt * EdotB_sq * c2
                  / ( ( (B_sq * c2) - E_sq)
                    + std::sqrt( ( (B_sq*c2) - E_sq) * ( (B_sq*c2) - E_sq ) + 4._prt * EdotB_sq * c2 ) );

        // only if E << B -- else fix with thesis
        if ( (B_sq + Eprime_sq/c2) > 1.e-30 ) {
            amrex::Real vx_drift = (Ey*Bz - Ez*By)/(B_sq + Eprime_sq/c2);
            amrex::Real vy_drift = (Ez*Bx - Ex*Bz)/(B_sq + Eprime_sq/c2);
            amrex::Real vz_drift = (Ex*By - Ey*Bx)/(B_sq + Eprime_sq/c2);

            // 1(b) unit vector along drift velocity
            amrex::Real drift_mag = std::sqrt(vx_drift*vx_drift + vy_drift*vy_drift + vz_drift*vz_drift);
            if (drift_mag < (0.99 * PhysConst::c) ) {
	        amrex::Real vxd_hat = vx_drift/drift_mag;
                amrex::Real vyd_hat = vy_drift/drift_mag;
                amrex::Real vzd_hat = vz_drift/drift_mag;

                // 1(c) gamma_drift = gamma_o = 1 / (sqrt ( 1 - drift_mag^2/c^2) )
                amrex::Real gamma_drift = 1._prt/std::sqrt( 1._prt - drift_mag*drift_mag*inv_c2);

                // 2 Bfield transform
                // By construction, only Bperp to drifting frame is non-zero
                amrex::Real Bx_prime = gamma_drift * ( Bx - inv_c2 * (vy_drift*Ez - vz_drift*Ey) );
                amrex::Real By_prime = gamma_drift * ( By - inv_c2 * (vz_drift*Ex - vx_drift*Ez) );
                amrex::Real Bz_prime = gamma_drift * ( Bz - inv_c2 * (vx_drift*Ey - vy_drift*Ex) );

                amrex::Real Bprime_mag = std::sqrt(Bx_prime*Bx_prime + By_prime*By_prime + Bz_prime*Bz_prime);
                amrex::Real Bxprime_hat = Bx_prime/Bprime_mag;
                amrex::Real Byprime_hat = By_prime/Bprime_mag;
                amrex::Real Bzprime_hat = Bz_prime/Bprime_mag;

                // New u momentum in drift frame
                amrex::Real udotBprimehat = ux*Bxprime_hat + uy*Byprime_hat + uz*Bzprime_hat;
                amrex::Real ux_new_prime = udotBprimehat * Bxprime_hat;
                amrex::Real uy_new_prime = udotBprimehat * Byprime_hat;
                amrex::Real uz_new_prime = udotBprimehat * Bzprime_hat;

                // New u momentum in lab frame
                amrex::Real gamma_new_prime = std::sqrt( 1._prt + inv_c2 * ( ux_new_prime*ux_new_prime
                                                                           + uy_new_prime*uy_new_prime
                                                                           + uz_new_prime*uz_new_prime ) );
                ux = gamma_drift * ( ux_new_prime + gamma_new_prime * vx_drift);
                uy = gamma_drift * ( uy_new_prime + gamma_new_prime * vy_drift);
                uz = gamma_drift * ( uz_new_prime + gamma_new_prime * vz_drift);
            }
        }
    }
#endif

}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
