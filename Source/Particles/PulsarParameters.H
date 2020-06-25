#ifndef PULSAR_PARAMETERS_H
#define PULSAR_PARAMETERS_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>

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
   extern AMREX_GPU_DEVICE_MANAGED int damp_E_internal;

   extern AMREX_GPU_DEVICE_MANAGED int verbose;

   void ReadParameters();

   AMREX_GPU_HOST_DEVICE AMREX_INLINE
   void PulsarEField(const amrex::ParticleReal xp, const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal &Exp, amrex::ParticleReal &Eyp,
                     amrex::ParticleReal &Ezp,
                     amrex::Real time)
   {
        // spherical r, theta, phi
        const amrex::Real xc = center_star[0];
        const amrex::Real yc = center_star[1];
        const amrex::Real zc = center_star[2];
        const amrex::Real r = std::sqrt( (xp-xc)*(xp-xc) + (yp-yc)*(yp-yc) + (zp-zc)*(zp-zc) );
        const amrex::Real phi = std::atan2((yp-yc),(xp-xc));
        amrex::Real theta = 0.0;
        if (r > 0) {
           theta = std::acos((zp-zc)/r);
        }
        const amrex::Real c_theta = std::cos(theta);
        const amrex::Real s_theta = std::sin(theta);
        const amrex::Real c_phi = std::cos(phi);
        const amrex::Real s_phi = std::sin(phi);
        amrex::Real omega = omega_star;
        // ramping up omega
        if (time < 2.0e-4) {
           omega = omega_star*time/2.0e-4;
        }

        if (r<R_star) {
           amrex::Real r_ratio = R_star/r;
           amrex::Real r3 = r_ratio*r_ratio*r_ratio;
           // Michel and Li -- eq 14 and 15 from michel and li
           amrex::Real Er = B_star*omega*r_ratio*r_ratio*r*s_theta*s_theta;
           amrex::Real Etheta = -B_star*omega*r_ratio*r_ratio*r*2.0*s_theta*c_theta;
           Exp = Er*s_theta*c_phi + Etheta*c_theta*c_phi;
           Eyp = Er*s_theta*s_phi + Etheta*c_theta*s_phi;
           Ezp = Er*c_theta - Etheta*s_theta;
        }

        // On and outside star surface -- dipole B and E with monopole
        if (r >= R_star ) {
           amrex::Real r_ratio = R_star/r;
           amrex::Real r3 = r_ratio*r_ratio*r_ratio;
           // Taking derivative of phi given in eq 30 of Michel and Li
           amrex::Real Er = B_star*omega*R_star*r_ratio*r3*(1.0-3.0*c_theta*c_theta);
           if (E_external_monopole == 1) {
                Er += (2.0/3.0)*omega*B_star*R_star*r_ratio*r_ratio;
           }
           amrex::Real Etheta = (-1.0)*B_star*omega*R_star*r_ratio*r3*(2.0*s_theta*c_theta);

           Exp = Er*s_theta*c_phi + Etheta*c_theta*c_phi;
           Eyp = Er*s_theta*s_phi + Etheta*c_theta*s_phi;
           Ezp = Er*c_theta - Etheta*s_theta;
        }
   }


   AMREX_GPU_HOST_DEVICE AMREX_INLINE
   void PulsarBField(const amrex::ParticleReal xp, const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal &Bxp, amrex::ParticleReal &Byp,
                     amrex::ParticleReal &Bzp,
                     amrex::Real time)
   {
        // spherical r, theta, phi
        const amrex::Real xc = center_star[0];
        const amrex::Real yc = center_star[1];
        const amrex::Real zc = center_star[2];
        const amrex::Real r = std::sqrt( (xp-xc)*(xp-xc) + (yp-yc)*(yp-yc) + (zp-zc)*(zp-zc) );
        const amrex::Real phi = std::atan2((yp-yc),(xp-xc));
        amrex::Real theta = 0.0;
        if (r > 0) {
           theta = std::acos((zp-zc)/r);
        }
        const amrex::Real c_theta = std::cos(theta);
        const amrex::Real s_theta = std::sin(theta);
        const amrex::Real c_phi = std::cos(phi);
        const amrex::Real s_phi = std::sin(phi);
        amrex::Real omega = omega_star;
        // ramping up omega
        if (time < 2.0e-4) {
           omega = omega_star*time/2.0e-4;
        }

        if (r<R_star) {
           amrex::Real r_ratio = R_star/r;
           amrex::Real r3 = r_ratio*r_ratio*r_ratio;
           // Michel and Li -- eq 14 and 15 from michel and li
           // dipole B field
           amrex::Real Br = 2.0*B_star*r3*c_theta;
           amrex::Real Btheta = B_star*r3*s_theta;

           Bxp = Br*s_theta*c_phi + Btheta*c_theta*c_phi;
           Byp = Br*s_theta*s_phi + Btheta*c_theta*s_phi;
           Bzp = Br*c_theta - Btheta*s_theta;
        }

        // On and outside star surface -- dipole B and E with monopole
        if (r >= R_star ) {
           amrex::Real r_ratio = R_star/r;
           amrex::Real r3 = r_ratio*r_ratio*r_ratio;
           // Taking derivative of phi given in eq 30 of Michel and Li
           amrex::Real Br = 2.0*B_star*r3*c_theta;
           amrex::Real Btheta = B_star*r3*s_theta;

           Bxp = Br*s_theta*c_phi + Btheta*c_theta*c_phi;
           Byp = Br*s_theta*s_phi + Btheta*c_theta*s_phi;
           Bzp = Br*c_theta - Btheta*s_theta;
        }
   }

//   AMREX_GPU_HOST_DEVICE AMREX_INLINE
//   void PulsarEBField(amrex::Real xp, amrex::Real yp, amrex::Real zp,
//                      amrex::Real &Exp, amrex::Real &Eyp, amrex::Real &Ezp,
//                      amrex::Real &Bxp, amrex::Real &Byp, amrex::Real &Bzp,
//                      amrex::Real time)
//   {
//        // spherical r, theta, phi
//        const amrex::Real xc = center_star[0];
//        const amrex::Real yc = center_star[1];
//        const amrex::Real zc = center_star[2];
//        const amrex::Real r = std::sqrt( (xp-xc)*(xp-xc) + (yp-yc)*(yp-yc) + (zp-zc)*(zp-zc) );
//        const amrex::Real phi = std::atan2((yp-yc),(xp-xc));
//        amrex::Real theta = 0.0;
//        if (r > 0) {
//           theta = std::acos((zp-zc)/r);
//        }
//        const amrex::Real c_theta = std::cos(theta);
//        const amrex::Real s_theta = std::sin(theta);
//        const amrex::Real c_phi = std::cos(phi);
//        const amrex::Real s_phi = std::sin(phi);
//        amrex::Real omega = omega_star;
//        // ramping up omega
//        if (time < 2.0e-4) {
//           omega = omega_star*time/2.0e-4;
//        }
//
//        if (r<R_star) {
//           amrex::Real r_ratio = R_star/r;
//           amrex::Real r3 = r_ratio*r_ratio*r_ratio;
//           // Michel and Li -- eq 14 and 15 from michel and li
//           amrex::Real Er = B_star*omega*r_ratio*r_ratio*r*s_theta*s_theta;
//           amrex::Real Etheta = -B_star*omega*r_ratio*r_ratio*r*2.0*s_theta*c_theta;
//           Exp = Er*s_theta*c_phi + Etheta*c_theta*c_phi;
//           Eyp = Er*s_theta*s_phi + Etheta*c_theta*s_phi;
//           Ezp = Er*c_theta - Etheta*s_theta;
//
//           // dipole B field
//           amrex::Real Br = 2.0*B_star*r3*c_theta;
//           amrex::Real Btheta = B_star*r3*s_theta;
//
//           Bxp = Br*s_theta*c_phi + Btheta*c_theta*c_phi;
//           Byp = Br*s_theta*s_phi + Btheta*c_theta*s_phi;
//           Bzp = Br*c_theta - Btheta*s_theta;
//        }
//
//        // On and outside star surface -- dipole B and E with monopole
//        if (r >= R_star ) {
//           amrex::Real r_ratio = R_star/r;
//           amrex::Real r3 = r_ratio*r_ratio*r_ratio;
//           // Taking derivative of phi given in eq 30 of Michel and Li
//           amrex::Real Er = B_star*omega*R_star*r_ratio*r3*(1.0-3.0*c_theta*c_theta);
//           if (E_external_monopole == 1) {
//                Er += (2.0/3.0)*omega*B_star*R_star*r_ratio*r_ratio;
//           }
//           amrex::Real Etheta = (-1.0)*B_star*omega*R_star*r_ratio*r3*(2.0*s_theta*c_theta);
//
//           Exp = Er*s_theta*c_phi + Etheta*c_theta*c_phi;
//           Eyp = Er*s_theta*s_phi + Etheta*c_theta*s_phi;
//           Ezp = Er*c_theta - Etheta*s_theta;
//
//           amrex::Real Br = 2.0*B_star*r3*c_theta;
//           amrex::Real Btheta = B_star*r3*s_theta;
//
//           Bxp = Br*s_theta*c_phi + Btheta*c_theta*c_phi;
//           Byp = Br*s_theta*s_phi + Btheta*c_theta*s_phi;
//           Bzp = Br*c_theta - Btheta*s_theta;
//        }
//   }

   namespace Spherical
   {
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        amrex::Real r(int i, int j, int k, amrex::GeometryData const& geom)
        {
            const auto domain_box = geom.Domain();
            const auto domain_ilo = amrex::lbound(domain_box);
            const auto domain_xlo = geom.ProbLo();
            const auto domain_xhi = geom.ProbHi();
            const auto domain_dx  = geom.CellSize();

            const amrex::Real x = domain_xlo[0] + (i - domain_ilo.x + 0.5) * domain_dx[0];
            const amrex::Real y = domain_xlo[1] + (j - domain_ilo.y + 0.5) * domain_dx[1];
            const amrex::Real z = domain_xlo[2] + (k - domain_ilo.z + 0.5) * domain_dx[2];

            const amrex::Real xc = 0.5 * (domain_xlo[0] + domain_xhi[0]);
            const amrex::Real yc = 0.5 * (domain_xlo[1] + domain_xhi[1]);
            const amrex::Real zc = 0.5 * (domain_xlo[2] + domain_xhi[2]);

            const amrex::Real r = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));

            return r;
        }
   }

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
   void DampEField(int i, int j, int k, amrex::GeometryData const& geom, amrex::Array4<amrex::Real> const& Efield)
   {
       const amrex::Real r = Spherical::r(i, j, k, geom);

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
