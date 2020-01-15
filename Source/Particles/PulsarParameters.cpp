#include <PulsarParameters.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

namespace PulsarParm
{
   std::string pulsar_type;

   AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real dR_star;

   AMREX_GPU_DEVICE_MANAGED int verbose = 0;

   void ReadParameters() {
      amrex::ParmParse pp("pulsar");
      pp.query("pulsarType",pulsar_type);
      pp.query("omega_star",omega_star);
      pp.query("R_star",R_star);
      pp.query("B_star",B_star);
      pp.query("dR",dR_star);
      pp.query("verbose",verbose);
   }
  
   
}
