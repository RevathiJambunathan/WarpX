#include <PulsarParameters.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

namespace PulsarParm
{
   std::string pulsar_type;

   AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real dR_star;
   AMREX_GPU_DEVICE_MANAGED int EB_external = 0;
   AMREX_GPU_DEVICE_MANAGED int E_external_monopole = 0;
   AMREX_GPU_DEVICE_MANAGED 
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star(3,0.0);
   AMREX_GPU_DEVICE_MANAGED int verbose = 0;

   void ReadParameters() {
      amrex::ParmParse pp("pulsar");
      pp.query("pulsarType",pulsar_type);
      pp.query("omega_star",omega_star);
      amrex::Vector<amrex::Real> center_star_v(AMREX_SPACEDIM);
      pp.queryarr("center_star",center_star_v);
      std::copy(center_star_v.begin(),center_star_v.end(),center_star.begin());
      pp.query("R_star",R_star);
      pp.query("B_star",B_star);
      pp.query("dR",dR_star);
      pp.query("verbose",verbose);
      pp.query("EB_external",EB_external);
      pp.query("E_external_monopole",E_external_monopole);
      amrex::Print() << " Pulsar center: " << center_star[0] << " " << center_star[1] << " " << center_star[2] << "\n";
      amrex::Print() << " Pulsar omega: " << omega_star << "\n";
      amrex::Print() << " Pulsar B_star : " << B_star << "\n";
   }

   AMREX_GPU_HOST_DEVICE AMREX_INLINE 
   void PulsarEBField(amrex::Real xp, amrex::Real yp, amrex::Real zp,
                      amrex::Real &Exp, amrex::Real &Eyp, amrex::Real &Ezp,
                      amrex::Real &Bxp, amrex::Real &Byp, amrex::Real &Bzp,
                      amrex::Real time) {
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
           amrex::Real Er = B_star*omega*R_star*r_ratio*r3*(1.0-3.0*c_theta*c_theta);
           if (E_external_monopole == 1) {
                Er += (2.0/3.0)*omega*B_star*R_star*r_ratio*r_ratio;
           }
           amrex::Real Etheta = (-1.0)*B_star*omega*R_star*r_ratio*r3*(2.0*s_theta*c_theta);

           Exp = Er*s_theta*c_phi + Etheta*c_theta*c_phi;
           Eyp = Er*s_theta*s_phi + Etheta*c_theta*s_phi;
           Ezp = Er*c_theta - Etheta*s_theta;

           amrex::Real Br = 2.0*B_star*r3*c_theta;
           amrex::Real Btheta = B_star*r3*s_theta;

           Bxp = Br*s_theta*c_phi + Btheta*c_theta*c_phi;
           Byp = Br*s_theta*s_phi + Btheta*c_theta*s_phi;
           Bzp = Br*c_theta - Btheta*s_theta;
        }
   }

}
