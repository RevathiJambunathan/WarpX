#ifndef PULSAR_PARAMETERS_H
#define PULSAR_PARAMETERS_H

#include "Utils/Parser/IntervalsParser.H"
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <list>

using namespace amrex::literals;


class Pulsar {

public:
    //Constructor
    Pulsar ();

    static void ReadParameters();
    void InitData ();
    void InitDataAtRestart ();
    void InitializeConductorMultifabUsingParser(amrex::MultiFab *mf,
         amrex::ParserExecutor<3> const& conductor_parser, const int lev);

    void InitializeExternalPulsarFieldsOnGrid (amrex::MultiFab *mfx,
        amrex::MultiFab *mfy, amrex::MultiFab *mfz, const int lev, const bool init_Bfield);

    void ApplyCorotatingEfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Efield,
                                    const int lev, const amrex::Real a_dt);
    void ApplyDipoleBfield_BC ( std::array< std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
                                    const int lev, const amrex::Real a_dt);
    void SetTangentialEforInternalConductor( std::array <std::unique_ptr<amrex::MultiFab>, 3> &Efield,
                                    const int lev, const amrex::Real a_dt);
    void InitializeConductorMultifabUsingParser(const int lev);
    void ComputePlasmaNumberDensity ();
    void ComputePlasmaMagnetization ();
    void TuneSigma0Threshold (const int step);
    void TotalParticles ();
    amrex::Real SumInjectionFlag ();
    amrex::Real SumSigmaRelDiff ();
    void PrintInjectedCellValues ();
    int TotalParticlesToBeInjected (amrex::Real scale_factor);
    void TotalParticlesInjected ();
    amrex::Real SumInjectedCells ();
    amrex::Real PcountSum ();
    void FlagCellsForInjectionWithPcounts ();
    amrex::Real MaxMagnetization ();
    amrex::Real MaxThresholdSigma();
    amrex::Real LeastSquareFitforTC ();
    void FlagCellsInPolarCap (const int lev);
    void FlagCellsInPolarCap (amrex::MultiFab *mf, const int lev, amrex::Real cell_inject_rmin, amrex::Real cell_inject_rmax);
    amrex::Real SumPolarCapFlag ();
    void FlagCellsInInjectionRing (const int lev);
    void FlagCellsInInjectionRing (amrex::MultiFab *mf, const int lev, amrex::Real cell_inject_rmin, amrex::Real cell_inject_rmax);
    amrex::Real SumMagnetizationinInjectionRing ();

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
                                    amrex::Real const phi, amrex::Real const chi,
                                    amrex::Real const time,
                                    amrex::Real const omega_star_data,
                                    amrex::Real const ramp_omega_time_data,
                                    amrex::Real const Bstar,
                                    amrex::Real const Rstar,
                                    amrex::Real const dRstar,
                                    amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::Real omega = Omega(omega_star_data, time, ramp_omega_time_data);
        // Polar angle
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        // Oblique angle
        amrex::Real c_chi = std::cos(chi);
        amrex::Real s_chi = std::sin(chi);
        // Instantaneous phase, psi = phi - Omega*t (phi is the azimuthal angle)
        // Refer to Pg. 6 of Jeromi Petri's 2016 paper
        amrex::Real omega_t_integral;
        if (time < ramp_omega_time_data) {
            // omega returned from function above is Omega_star * t / tramp
            omega_t_integral = omega * time / 2.;
        } else {
            omega_t_integral = omega*(time - ramp_omega_time_data) + omega*ramp_omega_time_data/2.0;
        }
        amrex::Real psi = phi - omega_t_integral;
        amrex::Real c_psi = std::cos(psi);
        // Ratio : Rstar/r
        amrex::Real r_ratio;
        if (r > 0) {
            r_ratio = Rstar/r;
        } else {
            r_ratio = Rstar/(dRstar*0.5);
        }
        amrex::Real r2 = r_ratio * r_ratio;
        // Corotating electric field inside the pulsar that corresponds to dipole magnetic field
        // Eq 2.9 (a-c) Jeromi Petri 2016
        Er = Bstar * omega * r2 * Rstar *
             ( c_chi * s_theta * s_theta
             - s_chi * c_theta * s_theta * c_psi);
        Etheta = -Bstar * omega * r2 * Rstar *
                 ( c_chi * 2._rt * s_theta * c_theta
                 + s_chi * 2._rt * s_theta * s_theta * c_psi);
        Ephi = 0._rt;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ExternalEFieldSpherical (amrex::Real const r, amrex::Real const theta,
                                  amrex::Real const phi, amrex::Real const chi,
                                  amrex::Real const time,
                                  amrex::Real const omega_star_data,
                                  amrex::Real const ramp_omega_time_data,
                                  amrex::Real const Bstar, amrex::Real const Rstar,
                                  amrex::Real const corotatingE_maxradius,
                                  int const Eexternal_monopole,
                                  int const ApplyCorotatingEField,
                                  amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::ignore_unused(corotatingE_maxradius);
        amrex::Real omega = Omega(omega_star_data, time, ramp_omega_time_data);
        // Polar angle
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        // Oblique angle
        amrex::Real c_chi = std::cos(chi);
        amrex::Real s_chi = std::sin(chi);
        // Instantaneous phase, psi = phi - Omega*t (phi is the azimuthal angle)
        // Refer to Pg. 6 of Jeromi Petri's 2016 paper
        amrex::Real omega_t_integral;
        if (time < ramp_omega_time_data) {
            // omega returned from function above is Omega_star * t / tramp
            omega_t_integral = omega * time / 2.;
        } else {
            omega_t_integral = omega*(time - ramp_omega_time_data) + omega*ramp_omega_time_data/2.0;
        }
        amrex::Real psi = phi - omega_t_integral;
        amrex::Real c_psi = std::cos(psi);
        amrex::Real s_psi = std::sin(psi);

        amrex::Real r_ratio = Rstar/r;
        // inside pulsar
        if (ApplyCorotatingEField == 1) {
            amrex::Real r2 = r_ratio * r_ratio;
            // Corotating electric field inside the pulsar that corresponds to dipole magnetic field
            // Eq 2.9 (a-c) Jeromi Petri 2016
            Er = Bstar * omega * r2 * Rstar *
                 ( c_chi * s_theta * s_theta
                 - s_chi * c_theta * s_theta * c_psi);
            Etheta = -Bstar * omega * r2 * Rstar *
                     ( c_chi * 2._rt * s_theta * c_theta
                     + s_chi * 2._rt * s_theta * s_theta * c_psi);
            Ephi = 0._rt;
        }

        // outside pulsar
        if (ApplyCorotatingEField == 0) {
            amrex::Real r2 = r_ratio * r_ratio;
            amrex::Real r4 = r2 * r2;
            // Equations 2.8(a) - 2.8(c)
            Er = Bstar * omega * Rstar * r4 *
                 ( c_chi * (1._rt - 3._rt * c_theta * c_theta)
                 - 3._rt * s_chi * c_theta * s_theta * c_psi );
            if (Eexternal_monopole == 1) {
                // Monopole term from central charge = Qc/(4*pi*eps0*r^2)
                // For dipolar magnetic field, Qc = 8pi/3 * eps0*Omega*Bstar*Rstar^3*c_chi
                Er += (2._rt/3._rt) * omega * Bstar * Rstar * r2 * c_chi;
            }
            Etheta = Bstar * omega * Rstar *
                     ( r2 * s_chi * (r2*std::cos(2._rt*theta) - 1._rt) * c_psi
                     - r4 * c_chi * std::sin(2._rt*theta) );
            Ephi = Bstar * omega * Rstar * r2 * (1._rt - r2) * s_chi * c_theta * s_psi;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ExternalEMonopoleSpherical (amrex::Real const r, amrex::Real const theta,
                                     amrex::Real const phi, amrex::Real const chi,
                                     amrex::Real const time,
                                     amrex::Real const omega_star_data,
                                     amrex::Real const ramp_omega_time_data,
                                     amrex::Real const Bstar, amrex::Real const Rstar,
                                     amrex::Real &Er, amrex::Real &Etheta, amrex::Real &Ephi)
    {
        amrex::ignore_unused(phi, theta);
        amrex::Real omega = Omega(omega_star_data, time, ramp_omega_time_data);
        amrex::Real r_ratio = Rstar/r;
        amrex::Real r2 = r_ratio * r_ratio;
        amrex::Real c_chi = std::cos(chi);

        Er = (2._rt/3._rt) * omega * Bstar * Rstar * r2 * c_chi;
        Etheta = 0.;
        Ephi = 0.;

    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void ExternalBFieldSpherical (amrex::Real const r, amrex::Real const theta,
                                  amrex::Real const phi, amrex::Real const chi,
                                  amrex::Real const time,
                                  amrex::Real const omega_star_data,
                                  amrex::Real const ramp_omega_time_data,
                                  amrex::Real const Bstar, amrex::Real const Rstar,
                                  amrex::Real const dRstar,
                                  amrex::Real &Br, amrex::Real &Btheta, amrex::Real &Bphi)
    {
        // Polar angle
        amrex::Real c_theta = std::cos(theta);
        amrex::Real s_theta = std::sin(theta);
        // Oblique angle
        amrex::Real c_chi = std::cos(chi);
        amrex::Real s_chi = std::sin(chi);
        // Instantaneous phase, psi = phi - Omega*t (phi is the azimuthal angle)
        // Refer to Pg. 6 of Jeromi Petri's 2016 paper
        amrex::Real omega = Omega(omega_star_data, time, ramp_omega_time_data);
        amrex::Real omega_t_integral;
        if (time < ramp_omega_time_data) {
            // omega returned from function above is Omega_star * t / tramp
            omega_t_integral = omega * time / 2.;
        } else {
            omega_t_integral = omega*(time - ramp_omega_time_data) + omega*ramp_omega_time_data/2.0;
        }
        amrex::Real psi = phi - omega_t_integral;
        amrex::Real c_psi = std::cos(psi);
        amrex::Real s_psi = std::sin(psi);

        amrex::Real r_ratio;
        if (r > 0) {
            r_ratio = Rstar/r;
        } else {
            r_ratio = Rstar/(dRstar*0.5);
        }
        amrex::Real r3 = r_ratio*r_ratio*r_ratio;
        // The full dipole magnetic field as f(theta, phi, omega, t, and Chi)
        // Eqs 2.7(a) - 2.7(c) from Jeromi Petri 2016 paper
        Br = 2._rt * Bstar * r3 * ( c_chi * c_theta + s_chi * s_theta * c_psi);
        Btheta = Bstar * r3 * ( c_chi * s_theta - s_chi * c_theta * c_psi);
        Bphi = Bstar * r3 * s_chi * s_psi;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void getExternalEBOnParticle( amrex::Real const r, amrex::Real const theta,
                                         amrex::Real const phi, amrex::Real const chi,
                                         const int AddExternalMonopoleOnly, const int AddPulsarVacuumEFields,
                                         const int AddBDipoleOutsideRstar, const int AddPulsarVacuumBFields,
                                         amrex::Real const corotatingE_maxradius,
                                         const int E_external_monopole,
                                         amrex::Real const cur_time, amrex::Real const omega_star,
                                         amrex::Real const ramp_omega_time, amrex::Real const Bstar,
                                         amrex::Real const Rstar, amrex::Real dR_star,
                                         amrex::ParticleReal &Exp, amrex::ParticleReal &Eyp, amrex::ParticleReal &Ezp,
                                         amrex::ParticleReal &Bxp, amrex::ParticleReal &Byp, amrex::ParticleReal &Bzp)
    {
        // Adding external electric fields
        if (AddExternalMonopoleOnly == 1) {
            // only add the monopole electric field external to the conductive boundary surface
            if (r > corotatingE_maxradius) {
                amrex::Real Er = 0.0_rt;
                amrex::Real Etheta = 0.0_rt;
                amrex::Real Ephi = 0.0_rt;
                ExternalEMonopoleSpherical(r, theta, phi, chi, cur_time, omega_star,
                                           ramp_omega_time, Bstar, Rstar,
                                           Er, Etheta, Ephi);
                amrex::Real Ex_monopole = 0._rt;
                amrex::Real Ey_monopole = 0._rt;
                amrex::Real Ez_monopole = 0._rt;
                ConvertSphericalToCartesianXComponent(Er, Etheta, Ephi, r, theta, phi, Ex_monopole);
                ConvertSphericalToCartesianYComponent(Er, Etheta, Ephi, r, theta, phi, Ey_monopole);
                ConvertSphericalToCartesianZComponent(Er, Etheta, Ephi, r, theta, phi, Ez_monopole);
                Exp += Ex_monopole;
                Eyp += Ey_monopole;
                Ezp += Ez_monopole;
            } // if r > corotatingE_radius
        } else if (AddPulsarVacuumEFields == 1) {
            // Add the full pulsar vacuum fields at all locations
            // corotating inside the conducting boundary and
            // quadrapolar + monopole outside the conducting boundary
            amrex::Real Er = 0.0_rt;
            amrex::Real Etheta = 0.0_rt;
            amrex::Real Ephi = 0.0_rt;
            int ApplyCorotatingEField = 0;
            if (r <= corotatingE_maxradius) ApplyCorotatingEField = 1;
            ExternalEFieldSpherical(r, theta, phi, chi, cur_time, omega_star,
                                    ramp_omega_time, Bstar, Rstar,
                                    corotatingE_maxradius,
                                    E_external_monopole,
                                    ApplyCorotatingEField,
                                    Er, Etheta, Ephi);
            amrex::Real Ex_pulsar_vacuum = 0._rt;
            amrex::Real Ey_pulsar_vacuum = 0._rt;
            amrex::Real Ez_pulsar_vacuum = 0._rt;
            ConvertSphericalToCartesianXComponent(Er, Etheta, Ephi, r, theta, phi, Ex_pulsar_vacuum);
            ConvertSphericalToCartesianYComponent(Er, Etheta, Ephi, r, theta, phi, Ey_pulsar_vacuum);
            ConvertSphericalToCartesianZComponent(Er, Etheta, Ephi, r, theta, phi, Ez_pulsar_vacuum);
            Exp += Ex_pulsar_vacuum;
            Eyp += Ey_pulsar_vacuum;
            Ezp += Ez_pulsar_vacuum;
        }
        // Adding external magnetic fields
        if (AddBDipoleOutsideRstar == 1) {
            if (r > corotatingE_maxradius) {
                // Add dipole magnetic field outside the conducting boundary
                amrex::Real Br = 0._rt;
                amrex::Real Btheta = 0._rt;
                amrex::Real Bphi = 0._rt;
                ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                        omega_star, ramp_omega_time,
                                        Bstar, Rstar, dR_star,
                                        Br, Btheta, Bphi);
                amrex::Real Bx_dipole = 0._rt;
                amrex::Real By_dipole = 0._rt;
                amrex::Real Bz_dipole = 0._rt;
                ConvertSphericalToCartesianXComponent(Br, Btheta, Bphi, r, theta, phi, Bx_dipole);
                ConvertSphericalToCartesianYComponent(Br, Btheta, Bphi, r, theta, phi, By_dipole);
                ConvertSphericalToCartesianZComponent(Br, Btheta, Bphi, r, theta, phi, Bz_dipole);
                Bxp += Bx_dipole;
                Byp += By_dipole;
                Bzp += Bz_dipole;
            }
        } else if (AddPulsarVacuumBFields == 1) {
            // Add dipole magnetic field at all locations in the domain
            amrex::Real Br = 0._rt;
            amrex::Real Btheta = 0._rt;
            amrex::Real Bphi = 0._rt;
            ExternalBFieldSpherical(r, theta, phi, chi, cur_time,
                                    omega_star, ramp_omega_time,
                                    Bstar, Rstar, dR_star,
                                    Br, Btheta, Bphi);
            amrex::Real Bx_dipole = 0._rt;
            amrex::Real By_dipole = 0._rt;
            amrex::Real Bz_dipole = 0._rt;
            ConvertSphericalToCartesianXComponent(Br, Btheta, Bphi, r, theta, phi, Bx_dipole);
            ConvertSphericalToCartesianYComponent(Br, Btheta, Bphi, r, theta, phi, By_dipole);
            ConvertSphericalToCartesianZComponent(Br, Btheta, Bphi, r, theta, phi, Bz_dipole);
            Bxp += Bx_dipole;
            Byp += By_dipole;
            Bzp += Bz_dipole;
        }
    }


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void EnforceTheoreticalEBOnParticle( amrex::Real const r, amrex::Real const theta,
                                         amrex::Real const phi, amrex::Real const chi,
                                         amrex::Real const theory_max_radius,
                                         amrex::Real const corotatingE_maxradius,
                                         const int E_external_monopole,
                                         amrex::Real const cur_time, amrex::Real const omega_star,
                                         amrex::Real const ramp_omega_time, amrex::Real const Bstar,
                                         amrex::Real const Rstar, amrex::Real const dR_star,
                                         amrex::ParticleReal &Exp, amrex::ParticleReal &Eyp,
                                         amrex::ParticleReal &Ezp, amrex::ParticleReal &Bxp,
                                         amrex::ParticleReal &Byp, amrex::ParticleReal &Bzp)
    {
        // enfore theory on particle for r <= theory_max_radius
        if (r <= theory_max_radius) {
            int ApplyCorotatingEField = 0;
            // for r inside conducting boundary, apply corotating efield
            if (r <= corotatingE_maxradius) ApplyCorotatingEField = 1;
            // Compute theoretical Efield in spherical coordinates
            amrex::Real Er = 0._rt;
            amrex::Real Etheta = 0._rt;
            amrex::Real Ephi = 0._rt;
            amrex::Real Ex_theory = 0._rt;
            amrex::Real Ey_theory = 0._rt;
            amrex::Real Ez_theory = 0._rt;
            ExternalEFieldSpherical(r, theta, phi, chi, cur_time, omega_star, ramp_omega_time,
                                    Bstar, Rstar, corotatingE_maxradius, E_external_monopole,
                                    ApplyCorotatingEField, Er, Etheta, Ephi);
            // Compute theoretical Bfield in spherical coordinates
            amrex::Real Br = 0._rt;
            amrex::Real Btheta = 0._rt;
            amrex::Real Bphi = 0._rt;
            amrex::Real Bx_theory = 0._rt;
            amrex::Real By_theory = 0._rt;
            amrex::Real Bz_theory = 0._rt;
            ExternalBFieldSpherical(r, theta, phi, chi, cur_time, omega_star, ramp_omega_time,
                                    Bstar, Rstar, dR_star, Br, Btheta, Bphi);
            // Convert Efield from spherical to Cartesian
            ConvertSphericalToCartesianXComponent(Er, Etheta, Ephi, r, theta, phi, Ex_theory);
            ConvertSphericalToCartesianYComponent(Er, Etheta, Ephi, r, theta, phi, Ey_theory);
            ConvertSphericalToCartesianZComponent(Er, Etheta, Ephi, r, theta, phi, Ez_theory);
            // Convert Bfield from spherical to Cartesian
            ConvertSphericalToCartesianXComponent(Br, Btheta, Bphi, r, theta, phi, Bx_theory);
            ConvertSphericalToCartesianYComponent(Br, Btheta, Bphi, r, theta, phi, By_theory);
            ConvertSphericalToCartesianZComponent(Br, Btheta, Bphi, r, theta, phi, Bz_theory);
            // enforce electric field
            Exp = Ex_theory;
            Eyp = Ey_theory;
            Ezp = Ez_theory;
            // enforce magnetic field
            Bxp = Bx_theory;
            Byp = By_theory;
            Bzp = Bz_theory;
        }
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
        amrex::ignore_unused(r);
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
        amrex::ignore_unused(r);
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
        amrex::ignore_unused(r, phi, F_phi);
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

        amrex::ignore_unused(theta, F_z);
        F_phi = - F_x * std::sin(phi) + F_y * std::cos(phi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static void DampField(int i, int j, int k,
                          amrex::Array4<amrex::Real> const& Efield,
                          amrex::GpuArray<int, 3> const mf_ixtype,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const center_star_data,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_xlo,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_dx,
                          amrex::Real max_EBdamping_radius_data,
                          amrex::Real damping_scale_data,
                          amrex::Real Rstar_data)
    {
        amrex::Real x, y, z;
        ComputeCellCoordinates(i, j, k, mf_ixtype, domain_xlo, domain_dx, x, y, z);
        amrex::Real r, theta, phi;
        ConvertCartesianToSphericalCoord (x, y, z, center_star_data,
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


    static std::string m_pulsar_type;
    static amrex::Real m_omega_star;
    static amrex::Real m_R_star;
    static amrex::Real m_B_star;
    static amrex::Real m_dR_star;
    static amrex::Real m_omega_ramp_time;
    static amrex::Real m_field_damping_scale;
    static int m_do_EB_external;
    static int m_do_E_external_monopole;
    static amrex::Array<amrex::Real, 3> m_center_star;
    static amrex::Real m_max_ndens;
    static amrex::Real m_Ninj_fraction;
    static amrex::Real m_injection_rate;
    static amrex::Real m_GJ_injection_rate;
    static amrex::Real m_Sigma0_threshold;
    static amrex::Real m_Sigma0_baseline;
    static amrex::Real m_min_Sigma0;
    static amrex::Real m_max_Sigma0;
    static int m_ModifyParticleWtAtInjection;
    static amrex::Real m_rhoGJ_scale;
    static int m_do_damp_EB_internal;
    static amrex::Real m_max_EBcorotating_radius;
    static amrex::Real m_max_EBdamping_radius;
    static int m_turnoffdeposition;
    static amrex::Real m_max_nodepos_radius;
    static int m_turnoff_plasmaEB_gather;
    static amrex::Real m_max_nogather_radius;
    static int m_verbose;
    static amrex::Real m_max_particle_absorption_radius;
    static amrex::Real m_particle_inject_rmin;
    static amrex::Real m_particle_inject_rmax;
    static amrex::Real m_cell_inject_rmin;
    static amrex::Real m_cell_inject_rmax;
    static amrex::Real m_corotatingE_maxradius;
    static amrex::Real m_enforceDipoleB_maxradius;
    static int m_do_InitializeGrid_with_Pulsar_Bfield;
    static int m_do_InitializeGrid_with_Pulsar_Efield;
    static int m_enforceCorotatingE;
    static int m_enforceDipoleB;
    static int m_singleParticleTest;
    static amrex::Real m_Bdamping_scale;
    static int m_do_DampBDipoleInRing;
    static amrex::Real m_injection_time;
    static int m_continuous_injection;
    static amrex::Real m_removeparticle_theta_min;
    static amrex::Real m_removeparticle_theta_max;
    static int m_use_theoreticalEB;
    static amrex::Real m_theory_max_rstar;
    static int m_LimitDipoleBInit;
    static amrex::Real m_DipoleB_init_maxradius;
    static int m_AddExternalMonopoleOnly;
    static int m_AddMonopoleInsideRstarOnGrid;
    static int m_EnforceTheoreticalEBInGrid;
    static std::string m_str_conductor_function;
    static std::unique_ptr<amrex::Parser> m_conductor_parser;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_conductor_fp;
    static bool m_do_conductor;
    static int m_ApplyEfieldBCusingConductor;
    static bool m_do_FilterWithConductor;
    static int m_do_InitializeGridWithCorotatingAndExternalEField;
    static int m_AddBdipoleExternal;
    static int m_AddVacuumEFieldsIntAndExt;
    static int m_AddVacuumBFieldsIntAndExt;
    static amrex::Real m_injection_endtime;
    static utils::parser::IntervalsParser m_injection_tuning_interval;
    static amrex::Real m_sum_injection_rate;
    /** Number density of the species, one for electrons and one for positrons for pair-plasma simulations*/
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_plasma_number_density;
    /** Magnetization of plasma, sigma = B^2/mu_0m_e(ne+ni)*c^2 */
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_magnetization;
    /** Flagging 1 where plasma is injected */
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_injection_flag;
    /** Flagging 1 where plasma is injected */
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_injected_cell;
    /** Relative difference between sigma and sigma threshold (sigma-sigma_threshold)/sigma_threshold */
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_sigma_reldiff;
    /** Number of particles to be injected per cell per species */
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_pcount;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_injection_ring;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_injection_ringCC;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_sigma_inj_ring;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_sigma_threshold;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_PC_flag;
    amrex::MultiFab* get_pointer_sigma_threshold (int lev) const {return m_sigma_threshold[lev].get();}
    amrex::MultiFab& get_sigma_threshold (int lev) const {return *m_sigma_threshold[lev];}
    /** get pointer to number density */
    amrex::MultiFab* get_pointer_ndens (int lev) const{return m_plasma_number_density[lev].get();}
    /** get pointer to number magnetization */
    amrex::MultiFab* get_pointer_magnetization (int lev) const{return m_magnetization[lev].get();}
    amrex::MultiFab& get_magnetization  (int lev) {return *m_magnetization[lev];}
    amrex::MultiFab* get_pointer_injection_flag (int lev) const{return m_injection_flag[lev].get();}
    amrex::MultiFab& get_injection_flag (int lev) {return *m_injection_flag[lev];}
    amrex::MultiFab* get_pointer_injected_cell (int lev) const {return m_injected_cell[lev].get();}
    amrex::MultiFab& get_injected_cell (int lev) {return *m_injected_cell[lev];}
    amrex::MultiFab* get_pointer_sigma_reldiff (int lev) const { return m_sigma_reldiff[lev].get(); }
    amrex::MultiFab& get_sigma_reldiff (int lev) { return *m_sigma_reldiff[lev].get(); }
    amrex::MultiFab* get_pointer_pcount (int lev) const {return m_pcount[lev].get();}
    amrex::MultiFab& get_pcount (int lev) {return *m_pcount[lev];}
    amrex::MultiFab* get_pointer_injection_ring (int lev) const {return m_injection_ring[lev].get();}
    amrex::MultiFab& get_injection_ring (int lev) {return *m_injection_ring[lev];}
    amrex::MultiFab* get_pointer_PC_flag (int lev) const {return m_PC_flag[lev].get();}
    amrex::MultiFab& get_PC_flag (int lev) {return *m_PC_flag[lev];}
    static std::string m_sigma_tune_method;
    std::list<amrex::Real> ROI_list;
    int list_size = 0;
    static int ROI_avg_window_size;
    static int modify_sigma_threshold;
    static int m_print_injected_celldata;
    static int m_print_celldata_starttime;
    std::list<amrex::Real> sigma_list;
    int sigma_list_size = 0;
    static int sigma_ref_avg_window_size;
    static amrex::Real m_sigma_threshold_sum;
    static int use_single_sigma_ref;
    static amrex::Real sigma_ref;
    /** Lower bound for number density to compute magnetization (Default is 1.e-16)*/
    static amrex::Real m_lbound_ndens_magnetization;
    /** Upeer bound for relative difference sigma tune method, such that,
     ** if relative_difference > upper bound, then sigma is modified by upper bound.
     ** Upper bound specified must be less than 1, and default is 0.1 (10%)
     */
    static amrex::Real m_ubound_reldiff_sigma0;
    /** Oblique angle between magnetic field axis and pulsar rotation axis.
     ** (Read in degrees and converted to radians)
     */
    static amrex::Real m_Chi;
    static int EnforceParticleInjection;
    // upper bound in sigma rell diff to inject particles
    static amrex::Real m_injection_sigma_reldiff;
    static int WeightedParticleInjection;
    static amrex::Real m_bufferdR_forCCBounds;
    static int TotalParticlesIsSumOfSpecies;
    static int sigma_tune_method_TCTP;
    std::list<amrex::Real> InjCell_list;
    int InjCell_list_size = 0;
    static int m_InjCell_avg_window_size;
    static amrex::Real m_InjCell_sum;
    static amrex::Real m_particle_wt;
    static amrex::Real m_particle_scale_fac;
    static int m_use_Sigma0_avg;
    static amrex::Real m_particle_weight_scaling;
    static int m_use_LstSqFit_TC;
    static int m_use_maxsigma_for_Sigma0;
    static amrex::Real m_min_TCTP_ratio;
    static amrex::Real m_maxsigma_fraction;
    static amrex::Real m_injRing_radius;
    static amrex::Real m_part_bulkVelocity;
    static int m_pair_injection_flag;
    static amrex::Real m_RLC;
    static amrex::Real m_PC_radius;
    static amrex::Real m_PC_theta;
    static int m_flag_polarcap;
    static int m_use_FixedSigmaInput;
    static int m_onlyPCinjection;
    static int m_usePCflagcount_minInjCell;
    static amrex::Real m_PCInjectionCellFraction;
    static amrex::Real m_totalpolarcap_cells;
    static int m_use_injection_rate;
    static int m_GJdensity_limitinjection;
    static amrex::Real m_GJdensity_thresholdfactor;
    amrex::Real m_sum_inj_magnetization;
    static amrex::Real m_limit_GJfactor;
    static int injectiontype;
    static amrex::Real m_injection_GJdensitythreshold;
    static int m_use_BC_smoothening;
    static amrex::Real m_min_BC_radius;
    static amrex::Real m_BC_width;
    static amrex::Real m_gatherbuffer_min;
    static amrex::Real m_gatherbuffer_max;
    static amrex::Real m_depositbuffer_min;
    static amrex::Real m_depositbuffer_max;
    static int m_pml_cubic_sigma;
    static amrex::Real m_gammarad_real;
    static amrex::Real m_gammarad_scaled;
    static amrex::Real m_damping_strength;
    static amrex::Real m_totalcells_injectionring;
    static int m_do_scale_re_RR;
    static amrex::Real m_re_scaled;
    static amrex::Real m_gammarad_RR;
    static amrex::Real m_BLC;
    static amrex::Real m_beta_rec_RR;
    static amrex::Real m_re_scaledratio;
    static int m_do_zero_uperpB_driftframe;
    static amrex::Real m_rmax_zero_uperpB_driftframe;
private:
};

#endif
