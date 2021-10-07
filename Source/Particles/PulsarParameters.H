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


class Pulsar {

public:
    //Constructor
    Pulsar ();    

    std::string pulsar_type () {return m_pulsar_type;}
    amrex::Real omega_star () {return m_omega_star;}
    amrex::Real R_star () {return m_R_star;}
    amrex::Real B_star () {return m_B_star;}
    amrex::Real dR_star () {return m_dR_star;}
    amrex::Real omega_ramp_time () {return m_omega_ramp_time;}
    amrex::Real field_damping_scale () {return m_field_damping_scale;}
    int do_EB_external () {return m_do_EB_external;}
    int do_E_external_monopole () {return m_do_E_external_monopole;}
    amrex::Real center_star (int i_dim) {return m_center_star[i_dim];}
    amrex::Real max_ndens () {return m_max_ndens;}
    amrex::Real Ninj_fraction () { return m_Ninj_fraction;}
    amrex::Real ModifyParticleWtAtInjection () { return m_ModifyParticleWtAtInjection;}
    amrex::Real rhoGJ_scale () {return m_rhoGJ_scale;}
    int do_damp_EB_internal () {return m_do_damp_EB_internal;}
    amrex::Real max_EBcorotating_radius () {return m_max_EBcorotating_radius;}
    amrex::Real max_EBdamping_radius () {return m_max_EBdamping_radius;}
    int do_turnoff_deposition () {return m_turnoffdeposition; }
    amrex::Real max_nodepos_radius () {return m_max_nodepos_radius;}
    int do_turnoff_plasmaEB_gather () {return m_turnoff_plasmaEB_gather;}
    amrex::Real max_nogather_radius () {return m_max_nogather_radius;}
    int verbose () {return m_verbose;}
    amrex::Real max_particle_absorption_radius () {return m_max_particle_absorption_radius;}
    amrex::Real particle_inject_rmin () {return m_particle_inject_rmin;}
    amrex::Real particle_inject_rmax () {return m_particle_inject_rmax;}
    amrex::Real corotatingE_maxradius () {return m_corotatingE_maxradius;}
    amrex::Real enforceDipoleB_maxradius () {return m_enforceDipoleB_maxradius;}
    int do_InitializeGrid_with_Pulsar_Bfield ()
                    {return m_do_InitializeGrid_with_Pulsar_Bfield;}
    int do_InitializeGrid_with_Pulsar_Efield () 
                    {return m_do_InitializeGrid_with_Pulsar_Efield;}
    int enforceCorotatingE () {return m_enforceCorotatingE;}
    int enforceDipoleB () {return m_enforceDipoleB;}
    int do_singleParticleTest () {return m_singleParticleTest;}
    amrex::Real Bdamping_scale () {return m_Bdamping_scale;}
    int do_DampBDipoleInRing () {return m_do_DampBDipoleInRing;}
    amrex::Real injection_time () {return m_injection_time;}
    int do_continuous_injection () {return m_continuous_injection;}
    amrex::Real removeparticle_theta_min () {return m_removeparticle_theta_min;}
    amrex::Real removeparticle_theta_max () {return m_removeparticle_theta_max;}
    int use_theoreticalEB () {return m_use_theoreticalEB;}
    amrex::Real theory_max_rstar () {return m_theory_max_rstar;}
    int LimitDipoleBInit () {return m_LimitDipoleBInit;}
    amrex::Real DipoleB_init_maxradius () {return m_DipoleB_init_maxradius;}
    int ApplyExternalMonopoleOnly () {return m_AddExternalMonopoleOnly;}
    int ApplyMonopoleInsideRstarOnGrid () {return m_AddMonopoleInsideRstarOnGrid;}
    int EnforceTheoreticalEBInGrid () {return m_EnforceTheoreticalEBInGrid;}

    void ReadParameters();
    void InitializeExternalPulsarFieldsOnGrid (amrex::MultiFab *mfx,
        amrex::MultiFab *mfy, amrex::MultiFab *mfz, const int lev, const bool init_Bfield);

    void ApplyCorotatingEfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Efield,
                                    const int lev, const amrex::Real a_dt);
    void ApplyDipoleBfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
                                    const int lev, const amrex::Real a_dt);


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real Omega(const amrex::Real omegastar, const amrex::Real cur_time, amrex::Real omega_ramp_time)
    {
         amrex::Real omega = omegastar;
         if (omega_ramp_time > 0.0 && cur_time < omega_ramp_time) {
             omega = omegastar * cur_time / omega_ramp_time;
         }

         return omega;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void CorotatingEfieldSpherical (amrex::Real const r, amrex::Real const theta,
                                    amrex::Real const phi, amrex::Real const time,
                                    amrex::Real const omega_star_data,
                                    amrex::Real const ramp_omega_time_data,
                                    amrex::Real const Bstar,
                                    amrex::Real const Rstar,
                                    amrex::Real const dRstar,
                                    amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::Real omega = Omega(omega_star_data, time, ramp_omega_time_data);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real c_phi   = std::cos(phi);
        amrex::Real s_phi   = std::sin(phi);
        amrex::Real r_ratio;
        if (r > 0) {
            r_ratio = Rstar/r;
        } else {
            r_ratio = Rstar/(dRstar*0.5);
        }       
        amrex::Real r2 = r_ratio * r_ratio;
        // Michel and Li -- eq 14 , 15
        Er     =  Bstar * omega * r2 * Rstar * s_theta * s_theta;
        Etheta = -Bstar * omega * r2 * Rstar * 2.0 * s_theta * c_theta;
        Ephi   = 0.0; // aligned magnetic and rotation axis
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ExternalEFieldSpherical (amrex::Real const r, amrex::Real const theta,
                                  amrex::Real const phi, amrex::Real const time,
                                  amrex::Real const omega_star_data,
                                  amrex::Real const ramp_omega_time_data,
                                  amrex::Real const Bstar, amrex::Real const Rstar,
                                  int const corotatingE_maxradius,
                                  int const Eexternal_monopole,
                                  amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::Real omega = Omega(omega_star_data, time, ramp_omega_time_data);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real c_phi   = std::cos(phi);
        amrex::Real s_phi   = std::sin(phi);
        amrex::Real r_ratio = Rstar/r;
        // inside pulsar
        //if ( r <= max_EBcorotating_radius ) {
        if (r <= corotatingE_maxradius) {
            amrex::Real r2 = r_ratio * r_ratio;
            // Michel and Li -- eq 14 , 15
            Er     =  Bstar * omega * r2 * Rstar * s_theta * s_theta;
            Etheta = -Bstar * omega * r2 * Rstar * 2.0 * s_theta * c_theta;
            Ephi = 0.0; // aligned magnetic and rotation axis
        }

        // outside pulsar
        //if ( r > max_EBcorotating_radius ) {
        if (r > corotatingE_maxradius) {
            amrex::Real r4 = r_ratio*r_ratio*r_ratio*r_ratio;
            // Taking derivative of phi given in eq 30 of Michel and Li
            Er = Bstar * omega * Rstar * r4 * (1.0-3.0*c_theta*c_theta);
            if (Eexternal_monopole == 1) {
                 Er += (2.0/3.0) * omega * Bstar * Rstar * r_ratio * r_ratio;
            }
            Etheta = (-1.0) * Bstar * omega * Rstar * r4 * (2.0*s_theta*c_theta);
            Ephi = 0.0;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ExternalEMonopoleSpherical (amrex::Real const r, amrex::Real const theta,
                                     amrex::Real const phi, amrex::Real const time,
                                     amrex::Real const omega_star_data,
                                     amrex::Real const ramp_omega_time_data,
                                     amrex::Real const Bstar, amrex::Real const Rstar,
                                     amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::Real omega = Omega(omega_star_data, time, ramp_omega_time_data);
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real c_phi   = std::cos(phi);
        amrex::Real s_phi   = std::sin(phi);
        amrex::Real r_ratio = Rstar/r;

        Er = (2.0/3.0) * omega * Bstar * Rstar * r_ratio * r_ratio;
        Etheta = 0.;
        Ephi = 0.;

    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ExternalBFieldSpherical (amrex::Real const r, amrex::Real const theta,
                                  amrex::Real const phi, amrex::Real const time,
                                  amrex::Real const Bstar, amrex::Real const Rstar,
                                  amrex::Real const dRstar,
                                  amrex::Real &Br, amrex::Real &Btheta, amrex::Real &Bphi)
    {
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        amrex::Real r_ratio;
        if (r > 0) {
            r_ratio = Rstar/r;
        } else {
            r_ratio = Rstar/(dRstar*0.5);
        }
        amrex::Real r3 = r_ratio*r_ratio*r_ratio;
        // Michel and Li -- eq 14 and 15 from michel and li
        // dipole B field inside and outside the pulsar
        Br     = 2.0*Bstar*r3*c_theta;
        Btheta = Bstar*r3*s_theta;
        Bphi   = 0.0;
    }

    /** Compute Cartesian components corresponding to i, j, k based on the staggering,
        ixType, and the domain_lo and cell size, dx.
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ComputeCellCoordinates ( int i, int j, int k,
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
    static void ConvertCartesianToSphericalCoord ( amrex::Real const x, amrex::Real const y,
             amrex::Real const z,
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const center_star,
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_lo,
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_hi,
             amrex::Real &r, amrex::Real &theta, amrex::Real &phi)
    {
        amrex::Real xc = center_star[0];
        amrex::Real yc = center_star[1];
        amrex::Real zc = center_star[2];

        r = std::sqrt( (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc) );
        theta = 0.0;
        if (r > 0) theta = std::acos( (z-zc) / r );
        phi = std::atan2( (y-yc), (x-xc) );
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ConvertSphericalToCartesianXComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_x)
    {
        F_x = F_r     * std::sin(theta) * std::cos(phi)
            + F_theta * std::cos(theta) * std::cos(phi)
            - F_phi   * std::sin(phi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ConvertSphericalToCartesianYComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_y)
    {
        F_y = F_r     * std::sin(theta) * std::sin(phi)
            + F_theta * std::cos(theta) * std::sin(phi)
            + F_phi   * std::cos(phi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ConvertSphericalToCartesianZComponent (amrex::Real const F_r,
             amrex::Real const F_theta, amrex::Real const F_phi,
             amrex::Real const r, amrex::Real const theta,
             amrex::Real const phi, amrex::Real & F_z)
    {
        F_z = F_r * std::cos(theta) - F_theta * std::sin(theta);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ConvertCartesianToSphericalRComponent (amrex::Real const F_x,
            amrex::Real const F_y, amrex::Real const F_z,
            amrex::Real const theta, amrex::Real const phi, amrex::Real & F_r)
    {
        F_r = F_x * std::sin(theta) * std::cos(phi)
            + F_y * std::sin(theta) * std::sin(phi)
            + F_z * std::cos(theta);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ConvertCartesianToSphericalThetaComponent (amrex::Real const F_x,
            amrex::Real const F_y, amrex::Real const F_z,
            amrex::Real const theta, amrex::Real const phi, amrex::Real & F_theta)
    {
        F_theta = F_x * std::cos(theta) * std::cos(phi)
                + F_y * std::cos(theta) * std::sin(phi)
                - F_z * std::sin(theta);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ConvertCartesianToSphericalPhiComponent (amrex::Real const F_x,
            amrex::Real const F_y, amrex::Real const F_z,
            amrex::Real const theta, amrex::Real const phi, amrex::Real &F_phi)
    {
        F_phi = - F_x * std::sin(phi) + F_y * std::cos(phi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void DampField(int i, int j, int k, amrex::GeometryData const& geom,
                          amrex::Array4<amrex::Real> const& Efield,
                          amrex::GpuArray<int, 3> const mf_ixtype,
                          amrex::GpuArray<amrex::Real, 3> const center_star_data,
                          amrex::GpuArray<amrex::Real, 3> const domain_xlo,
                          amrex::GpuArray<amrex::Real, 3> const domain_xhi,
                          amrex::GpuArray<amrex::Real, 3> const domain_dx,
                          amrex::Real max_EBdamping_radius_data,
                          amrex::Real damping_scale_data,
                          amrex::Real Rstar_data)
    {
        amrex::Real x, y, z;
        ComputeCellCoordinates(i, j, k, mf_ixtype, domain_xlo, domain_dx, x, y, z);
        amrex::Real r, theta, phi;
        ConvertCartesianToSphericalCoord (x, y, z, center_star_data, domain_xlo, domain_xhi,
                                          r, theta, phi);
        if (r <= max_EBdamping_radius_data) {
            // Damping function: Fd = tanh(damping_scale * (r / R_star - 1)) + 1
            // for damping_scale >= 10 or so:
            // Fd(0) ~ 0
            // Fd(R_star) ~ 1
            const amrex::Real Fd = std::tanh(damping_scale_data * (r / Rstar_data - 1.0)) + 1.0;
            Efield(i, j, k) = Efield(i, j, k) * Fd;
        }
    }


private:
    std::string m_pulsar_type;
    amrex::Real m_omega_star;
    amrex::Real m_R_star;
    amrex::Real m_B_star;
    amrex::Real m_dR_star;    
    amrex::Real m_omega_ramp_time;
    amrex::Real m_field_damping_scale;
    int m_do_EB_external;
    int m_do_E_external_monopole;
    amrex::Array<amrex::Real, 3> m_center_star = {{0.}};
    amrex::Real m_max_ndens;
    amrex::Real m_Ninj_fraction = 1.0;
    int m_ModifyParticleWtAtInjection = 1.0;
    amrex::Real m_rhoGJ_scale;
    int m_do_damp_EB_internal;
    amrex::Real m_max_EBcorotating_radius;
    amrex::Real m_max_EBdamping_radius;
    int m_turnoffdeposition;
    amrex::Real m_max_nodepos_radius;
    int m_turnoff_plasmaEB_gather;
    amrex::Real m_max_nogather_radius;
    int m_verbose;
    amrex::Real m_max_particle_absorption_radius;
    amrex::Real m_particle_inject_rmin;
    amrex::Real m_particle_inject_rmax;
    amrex::Real m_corotatingE_maxradius;
    amrex::Real m_enforceDipoleB_maxradius;
    int m_do_InitializeGrid_with_Pulsar_Bfield;
    int m_do_InitializeGrid_with_Pulsar_Efield;
    int m_enforceCorotatingE;
    int m_enforceDipoleB;
    int m_singleParticleTest;
    amrex::Real m_Bdamping_scale;
    int m_do_DampBDipoleInRing;
    amrex::Real m_injection_time;
    int m_continuous_injection;
    amrex::Real m_removeparticle_theta_min;
    amrex::Real m_removeparticle_theta_max;
    int m_use_theoreticalEB;
    amrex::Real m_theory_max_rstar;
    int m_LimitDipoleBInit;
    amrex::Real m_DipoleB_init_maxradius;
    int m_AddExternalMonopoleOnly;
    int m_AddMonopoleInsideRstarOnGrid;
    int m_EnforceTheoreticalEBInGrid;
};

#endif
