#ifndef PULSAR_PARAMETERS_H
#define PULSAR_PARAMETERS_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>

using namespace amrex;

namespace PulsarParm
{
    extern std::string pulsar_type;

    extern AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real ramp_omega_time;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real dR_star;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real damping_scale;
    extern AMREX_GPU_DEVICE_MANAGED int EB_external;
    extern AMREX_GPU_DEVICE_MANAGED int E_external_monopole;
    extern AMREX_GPU_DEVICE_MANAGED
           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real max_ndens;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real Ninj_fraction;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real rhoGJ_scale;
    extern AMREX_GPU_DEVICE_MANAGED int damp_EB_internal;

    extern AMREX_GPU_DEVICE_MANAGED int verbose;

    void ReadParameters();


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::Real Omega(const amrex::Real time)
    {
         amrex::Real omega = omega_star;
         if (ramp_omega_time > 0.0 && time < ramp_omega_time) {
             omega = omega_star * time / ramp_omega_time;
         }

         return omega;
    }


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ExternalEFieldSpherical (amrex::Real const r, amrex::Real const theta,
                                  amrex::Real const phi, amrex::Real const time,
                                  amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::Real omega = Omega(time);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real c_phi   = std::cos(phi);
        amrex::Real s_phi   = std::sin(phi);
        amrex::Real r_ratio = R_star/r;
        // inside pulsar
        if ( r < R_star -dR_star ) {
            amrex::Real r2 = r_ratio * r_ratio;
            // Michel and Li -- eq 14 , 15
            Er     =  B_star * omega * r2 * r * s_theta * s_theta;
            Etheta = -B_star * omega * r2 * r * 2.0 * s_theta * c_theta;
            Ephi = 0.0; // aligned magnetic and rotation axis
        }

        // outside pulsar
        if ( r >= (R_star - dR_star ) ) {
            amrex::Real r4 = r_ratio*r_ratio*r_ratio*r_ratio;
            // Taking derivative of phi given in eq 30 of Michel and Li
            Er = B_star * omega * R_star * r4 * (1.0-3.0*c_theta*c_theta);
            if (E_external_monopole == 1) {
                 Er += (2.0/3.0) * omega * B_star * R_star * r_ratio * r_ratio;
            }
            Etheta = (-1.0) * B_star * omega * R_star * r4 * (2.0*s_theta*c_theta);
            Ephi = 0.0;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ExternalBFieldSpherical (amrex::Real const r, amrex::Real const theta,
                                  amrex::Real const phi, amrex::Real const time,
                                  amrex::Real &Br, amrex::Real &Btheta, amrex::Real &Bphi)
    {
        amrex::Real omega = Omega(time);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real r_ratio = R_star/r;
        amrex::Real r3 = r_ratio*r_ratio*r_ratio;
        // Michel and Li -- eq 14 and 15 from michel and li
        // dipole B field inside and outside the pulsar
        Br     = 2.0*B_star*r3*c_theta;
        Btheta = B_star*r3*s_theta;
        Bphi   = 0.0;
    }

    namespace Spherical
    {
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        amrex::Real r(int i, int j, int k, amrex::GeometryData const& geom, amrex::GpuArray<int, 3> const mf_ixType)
        {
            const auto domain_box = geom.Domain();
            const auto domain_ilo = amrex::lbound(domain_box);
            const auto domain_xlo = geom.ProbLo();
            const auto domain_xhi = geom.ProbHi();
            const auto domain_dx  = geom.CellSize();

            const amrex::Real x = domain_xlo[0] + (i ) * domain_dx[0] + (1.0 - mf_ixType[0])*domain_dx[0]*0.5;
            const amrex::Real y = domain_xlo[1] + (j ) * domain_dx[1] + (1.0 - mf_ixType[1])*domain_dx[1]*0.5;
            const amrex::Real z = domain_xlo[2] + (k ) * domain_dx[2] + (1.0 - mf_ixType[2])*domain_dx[2]*0.5;

            const amrex::Real xc = 0.5 * (domain_xlo[0] + domain_xhi[0]);
            const amrex::Real yc = 0.5 * (domain_xlo[1] + domain_xhi[1]);
            const amrex::Real zc = 0.5 * (domain_xlo[2] + domain_xhi[2]);

            const amrex::Real r = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));

            return r;
        }
    }

    /** Compute Cartesian components corresponding to i, j, k based on the staggering,
        ixType, and the domain_lo and cell size, dx.
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ComputeCellCoordinates ( int i, int j, int k,
             amrex::GpuArray<int, 3> const mf_type,
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_lo,
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
             amrex::Real &x, amrex::Real &y, amrex::Real &z)
    {
        x = domain_lo[0] + i*dx[0] + (1.0_rt - mf_type[0]) * dx[0]*0.5;
        y = domain_lo[1] + j*dx[1] + (1.0_rt - mf_type[1]) * dx[1]*0.5;
        z = domain_lo[2] + k*dx[2] + (1.0_rt - mf_type[2]) * dx[2]*0.5;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertCartesianToSphericalCoord ( amrex::Real const x, amrex::Real const y,
             amrex::Real const z,
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_lo,
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_hi,
             amrex::Real &r, amrex::Real &theta, amrex::Real &phi)
    {
        amrex::Real xc = (domain_lo[0] + domain_hi[0])/2.0;
        amrex::Real yc = (domain_lo[1] + domain_hi[1])/2.0;
        amrex::Real zc = (domain_lo[2] + domain_hi[2])/2.0;

        r = std::sqrt( (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc) );
        theta = 0.0;
        if (r > 0) theta = std::acos( (z-zc) / r );
        phi = std::atan2( (y-yc), (x-xc) );
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertSphericalToCartesianXComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_x)
    {
        F_x = F_r     * std::sin(theta) * std::cos(phi)
            + F_theta * std::cos(theta) * std::cos(phi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertSphericalToCartesianYComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_y)
    {
        F_y = F_r     * std::sin(theta) * std::sin(phi)
            + F_theta * std::cos(theta) * std::sin(phi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertSphericalToCartesianZComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_z)
    {
        F_z = F_r * std::cos(theta) - F_theta * std::sin(theta);
    }


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void DampField(int i, int j, int k, amrex::GeometryData const& geom, amrex::Array4<amrex::Real> const& Efield, amrex::GpuArray<int, 3> const mf_ixtype)
    {
        amrex::Real r = Spherical::r(i, j, k, geom, mf_ixtype);
        if (r < R_star) {
            // Damping function: Fd = tanh(damping_scale * (r / R_star - 1)) + 1
            // for damping_scale >= 10 or so:
            // Fd(0) ~ 0
            // Fd(R_star) ~ 1
            const amrex::Real Fd = std::tanh(damping_scale * (r / R_star - 1.0)) + 1.0;
            Efield(i, j, k) = Efield(i, j, k) * Fd;
        }
    }
}

#endif
