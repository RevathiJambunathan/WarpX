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
   extern AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;
   extern AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
   extern AMREX_GPU_DEVICE_MANAGED amrex::Real dR_star;
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
   void PulsarEBField(amrex::Real xp, amrex::Real yp, amrex::Real zp,
                      amrex::Real &Exp, amrex::Real &Eyp, amrex::Real &Ezp,
                      amrex::Real &Bxp, amrex::Real &Byp, amrex::Real &Bzp,
                      amrex::Real time);

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
   void PulsarDampEField(int i, int j, int k, amrex::GeometryData const& geom, amrex::Array4<amrex::Real> const& Efield)
   {
       const amrex::Real r = Spherical::r(i, j, k, geom);

       if (r < PulsarParm::R_star) {
           // Damping function: Fd = 0.5*(tanh(r-0.5*R_star)+1)
           // Fd(0) ~ 0
           // Fd(R_star) ~ 1
           const amrex::Real Fd = 0.5 * (std::tanh(r - 0.5 * PulsarParm::R_star) + 1.0);
           Efield(i, j, k) = Efield(i, j, k) * Fd;
       }
   }
}

#endif
