#ifndef PULSAR_PARAMETERS_H
#define PULSAR_PARAMETERS_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>

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
    extern AMREX_GPU_DEVICE_MANAGED int ModifyParticleWtAtInjection;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real rhoGJ_scale;
    extern AMREX_GPU_DEVICE_MANAGED int damp_EB_internal;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real max_EBcorotating_radius;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real max_EBdamping_radius;
    extern AMREX_GPU_DEVICE_MANAGED int turnoffdeposition;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real max_nodepos_radius;
    extern AMREX_GPU_DEVICE_MANAGED int turnoff_plasmaEB_gather;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real max_nogather_radius;
    extern AMREX_GPU_DEVICE_MANAGED int verbose;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real max_particle_absorption_radius;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real particle_inject_rmin;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real particle_inject_rmax;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real corotatingE_maxradius;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real enforceDipoleB_maxradius;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real InitializeGrid_with_Pulsar_Bfield;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real InitializeGrid_with_Pulsar_Efield;
    extern AMREX_GPU_DEVICE_MANAGED int enforceCorotatingE;
    extern AMREX_GPU_DEVICE_MANAGED int enforceDipoleB;
    extern AMREX_GPU_DEVICE_MANAGED int singleParticleTest;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real Bdamping_scale;
    extern AMREX_GPU_DEVICE_MANAGED int DampBDipoleInRing;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real injection_time;
    extern AMREX_GPU_DEVICE_MANAGED int continuous_injection;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real removeparticle_theta_min;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real removeparticle_theta_max;
    extern AMREX_GPU_DEVICE_MANAGED int use_theoreticalEB;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real theory_max_rstar;
    extern AMREX_GPU_DEVICE_MANAGED int LimitDipoleBInit;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real DipoleB_init_maxradius;
    extern AMREX_GPU_DEVICE_MANAGED int AddExternalMonopoleOnly;
    extern AMREX_GPU_DEVICE_MANAGED int EnforceTheoreticalEBInGrid;

    void ReadParameters();
    void InitializeExternalPulsarFieldsOnGrid (amrex::MultiFab *mfx,
        amrex::MultiFab *mfy, amrex::MultiFab *mfz, const int lev, const bool init_Bfield);

    void ApplyCorotatingEfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Efield,
                                    const int lev, const amrex::Real a_dt);
    void ApplyDipoleBfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
                                    const int lev, const amrex::Real a_dt);


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
    void CorotatingEfieldSpherical (amrex::Real const r, amrex::Real const theta,
                                    amrex::Real const phi, amrex::Real const time,
                                    amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::Real omega = Omega(time);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real c_phi   = std::cos(phi);
        amrex::Real s_phi   = std::sin(phi);
        amrex::Real r_ratio;
        if (r > 0) {
            r_ratio = R_star/r;
        } else {
            r_ratio = R_star/(dR_star*0.5);
        }       
        amrex::Real r2 = r_ratio * r_ratio;
        // Michel and Li -- eq 14 , 15
        Er     =  B_star * omega * r2 * R_star * s_theta * s_theta;
        Etheta = -B_star * omega * r2 * R_star * 2.0 * s_theta * c_theta;
        Ephi   = 0.0; // aligned magnetic and rotation axis
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
        if (r <= corotatingE_maxradius) {
            amrex::Real r2 = r_ratio * r_ratio;
            // Michel and Li -- eq 14 , 15
            Er     =  B_star * omega * r2 * R_star * s_theta * s_theta;
            Etheta = -B_star * omega * r2 * R_star * 2.0 * s_theta * c_theta;
            Ephi = 0.0; // aligned magnetic and rotation axis
        }

        // outside pulsar
        //if ( r > max_EBcorotating_radius ) {
        if (r > corotatingE_maxradius) {
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
    void ExternalEMonopoleSpherical (amrex::Real const r, amrex::Real const theta,
                                     amrex::Real const phi, amrex::Real const time,
                                     amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::Real omega = Omega(time);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real c_phi   = std::cos(phi);
        amrex::Real s_phi   = std::sin(phi);
        amrex::Real r_ratio = R_star/r;

        Er = (2.0/3.0) * omega * B_star * R_star * r_ratio * r_ratio;
        Etheta = 0.;
        Ephi = 0.;

    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ExternalBFieldSpherical (amrex::Real const r, amrex::Real const theta,
                                  amrex::Real const phi, amrex::Real const time,
                                  amrex::Real &Br, amrex::Real &Btheta, amrex::Real &Bphi)
    {
        amrex::Real omega = Omega(time);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real r_ratio;
        if (r > 0) {
            r_ratio = R_star/r;
        } else {
            r_ratio = R_star/(dR_star*0.5);
        }
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

            const amrex::Real xc = PulsarParm::center_star[0];
            const amrex::Real yc = PulsarParm::center_star[1];
            const amrex::Real zc = PulsarParm::center_star[2];

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
        amrex::Real xc = PulsarParm::center_star[0];
        amrex::Real yc = PulsarParm::center_star[1];
        amrex::Real zc = PulsarParm::center_star[2];
        r = std::sqrt( (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc) );
        if (r > 0) {
            theta = std::acos( (z-zc) / r );
        } else {
            theta = std::acos( 0. );
        }
        phi = std::atan2( (y-yc), (x-xc) );
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertSphericalToCartesianXComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_x)
    {
        F_x = F_r     * std::sin(theta) * std::cos(phi)
            + F_theta * std::cos(theta) * std::cos(phi)
            - F_phi   * std::sin(phi);
        //F_x = F_r     * std::sin(theta) * std::cos(phi)
        //    + F_theta * std::cos(theta) * std::cos(phi);

    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertSphericalToCartesianYComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_y)
    {
        F_y = F_r     * std::sin(theta) * std::sin(phi)
            + F_theta * std::cos(theta) * std::sin(phi)
            + F_phi   * std::cos(phi);
        //F_y = F_r     * std::sin(theta) * std::sin(phi)
        //    + F_theta * std::cos(theta) * std::sin(phi);

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
    void ConvertCartesianToSphericalRComponent (amrex::Real const F_x,
            amrex::Real const F_y, amrex::Real const F_z,
            amrex::Real const theta, amrex::Real const phi, amrex::Real & F_r)
    {
        F_r = F_x * std::sin(theta) * std::cos(phi)
            + F_y * std::sin(theta) * std::sin(phi)
            + F_z * std::cos(theta);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertCartesianToSphericalThetaComponent (amrex::Real const F_x,
            amrex::Real const F_y, amrex::Real const F_z,
            amrex::Real const theta, amrex::Real const phi, amrex::Real & F_theta)
    {
        F_theta = F_x * std::cos(theta) * std::cos(phi)
                + F_y * std::cos(theta) * std::sin(phi)
                - F_z * std::sin(theta);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ConvertCartesianToSphericalPhiComponent (amrex::Real const F_x,
            amrex::Real const F_y, amrex::Real const F_z,
            amrex::Real const theta, amrex::Real const phi, amrex::Real &F_phi)
    {
        F_phi = - F_x * std::sin(phi) + F_y * std::cos(phi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void DampField(int i, int j, int k, amrex::GeometryData const& geom, amrex::Array4<amrex::Real> const& Efield, amrex::GpuArray<int, 3> const mf_ixtype)
    {
        amrex::Real r = Spherical::r(i, j, k, geom, mf_ixtype);
        if (r <= max_EBdamping_radius) {
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
