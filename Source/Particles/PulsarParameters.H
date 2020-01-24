#ifndef PULSAR_PARAMETERS_H
#define PULSAR_PARAMETERS_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

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

   extern AMREX_GPU_DEVICE_MANAGED int verbose;

   void ReadParameters();
   AMREX_GPU_HOST_DEVICE AMREX_INLINE
   void PulsarEBField(amrex::Real xp, amrex::Real yp, amrex::Real zp,
                      amrex::Real &Exp, amrex::Real &Eyp, amrex::Real &Ezp,
                      amrex::Real &Bxp, amrex::Real &Byp, amrex::Real &Bzp,
                      amrex::Real time);

}

#endif
