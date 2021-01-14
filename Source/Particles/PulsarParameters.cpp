#include "PulsarParameters.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

namespace PulsarParm
{
    std::string pulsar_type;

    AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real ramp_omega_time = -1.0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real dR_star;
    AMREX_GPU_DEVICE_MANAGED amrex::Real damping_scale = 10.0;
    AMREX_GPU_DEVICE_MANAGED int EB_external = 0;
    AMREX_GPU_DEVICE_MANAGED int E_external_monopole = 0;
    AMREX_GPU_DEVICE_MANAGED
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star;
    AMREX_GPU_DEVICE_MANAGED int damp_EB_internal = 0;
    AMREX_GPU_DEVICE_MANAGED int verbose = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_ndens;
    AMREX_GPU_DEVICE_MANAGED amrex::Real Ninj_fraction;
    AMREX_GPU_DEVICE_MANAGED int ModifyParticleWtAtInjection = 1;
    AMREX_GPU_DEVICE_MANAGED amrex::Real rhoGJ_scale;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_EBcorotating_radius;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_EBdamping_radius;
    AMREX_GPU_DEVICE_MANAGED int turnoffdeposition = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_nodepos_radius;
    AMREX_GPU_DEVICE_MANAGED int turnoff_plasmaEB_gather = 0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_nogather_radius;
    AMREX_GPU_DEVICE_MANAGED amrex::Real max_particle_absorption_radius;

    void ReadParameters() {
        amrex::ParmParse pp("pulsar");
        pp.query("pulsarType",pulsar_type);
        pp.get("omega_star",omega_star);
        amrex::Vector<amrex::Real> center_star_v(AMREX_SPACEDIM);
        pp.getarr("center_star",center_star_v);
        std::copy(center_star_v.begin(),center_star_v.end(),center_star.begin());
        pp.get("R_star",R_star);
        pp.get("B_star",B_star);
        pp.get("dR",dR_star);
        pp.query("verbose",verbose);
        pp.query("EB_external",EB_external);
        pp.query("E_external_monopole",E_external_monopole);
        pp.query("damp_EB_internal",damp_EB_internal);
        pp.query("damping_scale",damping_scale);
        pp.query("ramp_omega_time",ramp_omega_time);
        amrex::Print() << " Pulsar center: " << center_star[0] << " " << center_star[1] << " " << center_star[2] << "\n";
        amrex::Print() << " Pulsar omega: " << omega_star << "\n";
        amrex::Print() << " Pulsar B_star : " << B_star << "\n";
        pp.get("max_ndens", max_ndens);
        pp.get("Ninj_fraction",Ninj_fraction);
        pp.query("ModifyParticleWeight", ModifyParticleWtAtInjection);
        // The maximum radius within which particles are absorbed/deleted every timestep.
        max_particle_absorption_radius = R_star;
        pp.query("max_particle_absorption_radius", max_particle_absorption_radius);
        pp.get("rhoGJ_scale",rhoGJ_scale);
        amrex::Print() << " pulsar max ndens " << max_ndens << "\n";
        amrex::Print() << " pulsar ninj fraction " << Ninj_fraction << "\n";
        amrex::Print() << " pulsar modify particle wt " << ModifyParticleWtAtInjection << "\n";
        amrex::Print() << " pulsar rhoGJ scaling " << rhoGJ_scale << "\n";
        amrex::Print() << " EB_external : " << EB_external << "\n";
        if (EB_external == 1) {
            // Max corotating radius defines the region where the EB field shifts from
            // corotating (v X B) to quadrapole. default is R_star.
            max_EBcorotating_radius = R_star;
            pp.query("EB_corotating_maxradius", max_EBcorotating_radius);
            amrex::Print() << " EB coratating maxradius : " << max_EBcorotating_radius << "\n";
        }
        if (damp_EB_internal == 1) {
            // Radius of the region within which the EB fields are damped
            max_EBdamping_radius = R_star;
            pp.query("damp_EB_radius", max_EBdamping_radius);
            amrex::Print() << " max EB damping radius : " << max_EBdamping_radius << "\n";
        }
        // query to see if particle j,rho deposition should be turned off
        // within some region interior to the star
        // default is 0
        pp.query("turnoffdeposition", turnoffdeposition);
        amrex::Print() << " is deposition off ? " << turnoffdeposition << "\n";
        if (turnoffdeposition == 1) {
            max_nodepos_radius = R_star;
            pp.query("max_nodepos_radius", max_nodepos_radius);
            amrex::Print() << " deposition turned off within radius : " << max_nodepos_radius << "\n";
        }
        pp.query("turnoff_plamsaEB_gather", turnoff_plasmaEB_gather);
        amrex::Print() << " is plasma EB gather off ? " << turnoff_plasmaEB_gather << "\n";
        if (turnoff_plasmaEB_gather == 1) {
            max_nogather_radius = R_star;
            pp.query("max_nogather_radius", max_nogather_radius);
            amrex::Print() << " gather off within radius : " << max_nogather_radius << "\n";
        }
    }
}
