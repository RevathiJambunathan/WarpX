#ifndef WARPX_PARTICLES_ADDPULSARPLASMA_H_
#define WARPX_PARTICLES_ADDPULSARPLASMA_H_

#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"
#include "Utils/WarpXConst.H"
#ifdef PULSAR
    #include "Particles/PulsarParameters.H"
#endif // end if pulsar
#include <AMReX_REAL.H>
#include <limits>
#include <AMReX_Particles.H>

using namespace amrex;
using ParticleType = WarpXParticleContainer::ParticleType;
struct AddPulsarPlasmaCondition
{

    AddPulsarPlasmaCondition () = default;

    /** This operator determines if a particle p, with charge q_pm, should be
     *  introduced in the cell that intersects with the surface of the pulsar.
     *  Particles are introduced by computing the jump in the E-field, Er_cell-Er_cor
     *  and obtaining the corresponding surface charge density.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator () ( const GpuArray<Real,AMREX_SPACEDIM> overlap_corner,
                       int i, int j, int k, const GpuArray<Real,AMREX_SPACEDIM> dx,
                       amrex::Real t,
                       amrex::Array4<const amrex::Real> const& ex_arr,
                       amrex::Array4<const amrex::Real> const& ey_arr,
                       amrex::Array4<const amrex::Real> const& ez_arr,
                       amrex::Array4<const amrex::Real> const& rho_arr,
                       int icell, int jcell, int kcell, ParticleType& p,
                       const amrex::Real q_pm, int i_part, int num_ppc,
                       int include_part_sampling = 1) const noexcept
    {
        amrex::Real scale_fac;
#if AMREX_SPACEDIM==3
        scale_fac = dx[0]*dx[1]*dx[2]/num_ppc;
#elif AMREX_SPACEDIM==2
        scale_fac = dx[0]*dx[1]/num_ppc;
#endif
        // get cell center
        amrex::Real cc_x = overlap_corner[0] + i*dx[0] + 0.5*dx[0];
        amrex::Real cc_y = overlap_corner[1] + j*dx[1] + 0.5*dx[1];
        amrex::Real cc_z = overlap_corner[2] + k*dx[2] + 0.5*dx[2];
        // Pulsar center_star
        amrex::Real xc = PulsarParm::center_star[0];
        amrex::Real yc = PulsarParm::center_star[1];
        amrex::Real zc = PulsarParm::center_star[2];
        // Convert cartesian cell-center co-ordinates to spherical co-ordinates
        // spherical radius
        amrex::Real cc_rad = std::sqrt( (cc_x-xc)*(cc_x-xc)
                                      + (cc_y-yc)*(cc_y-yc)
                                      + (cc_z-zc)*(cc_z-zc) );
        // cylindrical radius
        amrex::Real r_cl = std::sqrt( (cc_x-xc)*(cc_x-xc)
                                    + (cc_y-yc)*(cc_y-yc) );
        // theta
        amrex::Real cc_theta = 0.0;
        if (cc_rad > 0.0 ) cc_theta = std::acos( (cc_z-zc) / cc_rad );
        // phi
        amrex::Real cc_phi = std::atan2( (cc_y-yc), (cc_x-xc) );
        // cos_theta
        const amrex::Real c_theta = (cc_z - zc)/cc_rad;
        // sin theta
        const amrex::Real s_theta = r_cl/cc_rad;
        // cos_phi and sin_phi
        amrex::Real c_phi = 0.0;
        amrex::Real s_phi = 0.0;
        if ( r_cl > 0.0 ) {
            c_phi = (cc_x - xc)/r_cl;
            s_phi = (cc_y - yc)/r_cl;
        }
        // Compute the cororating radial electric field
        amrex::Real omega = PulsarParm::Omega(t);
        amrex::Real R_ratio = PulsarParm::R_star/cc_rad;
        amrex::Real r3 = R_ratio*R_ratio*R_ratio;
        amrex::Real Er_cor = PulsarParm::B_star * omega
                             * cc_rad * s_theta * s_theta;

        // Theoretical Goldreich-Julian surface charge-density
        amrex::Real rho_GJ = 2. * PhysConst::ep0
                           * PulsarParm::B_star * omega
                           * (1. - 3.*c_theta * c_theta) * PulsarParm::rhoGJ_scale;

        amrex::Real sigma_GJ = PhysConst::ep0 * omega * cc_rad * PulsarParm::B_star*
                               (3.0 - 5.*c_theta*c_theta);

        // Average the edge-centered E-field to cell-center
        amrex::Real ex_avg = 0.25*( ex_arr(icell  , jcell  , kcell  )
                                  + ex_arr(icell  , jcell+1, kcell  )
                                  + ex_arr(icell  , jcell  , kcell+1)
                                  + ex_arr(icell  , jcell+1, kcell+1) );

        amrex::Real ey_avg = 0.25*( ey_arr(icell  , jcell  , kcell  )
                                  + ey_arr(icell+1, jcell  , kcell  )
                                  + ey_arr(icell  , jcell  , kcell+1)
                                  + ey_arr(icell+1, jcell  , kcell+1) );

        amrex::Real ez_avg = 0.25*( ez_arr(icell  , jcell  , kcell  )
                                  + ez_arr(icell+1, jcell  , kcell  )
                                  + ez_arr(icell  , jcell+1, kcell  )
                                  + ez_arr(icell+1, jcell+1, kcell  ) );
        // convert cell-centered cartesian components of E-field to spherical co-ord.
        amrex::Real Er_cell = ex_avg*s_theta*c_phi + ey_avg*s_theta*s_phi
                                  + ez_avg*c_theta;

        //amrex::Real Er_ext = PulsarParm::B_star * omega * PulsarParm::R_star
        //                    * R_ratio*r3*(1.0-3.0*c_theta*c_theta);
        //Er_ext += (2./3.)*omega*PulsarParm::B_star*PulsarParm::R_star*R_ratio*R_ratio;

        // The following steps determine the total number of particles to be
        // introduced on the surface based on the difference between the Er_cell
        // and Er_corotating.
        // 1. Surface_charge_density = (Er_cell-Er_cor)*epsilon
        // 2. Only a fraction of this surface charge density is actually introduced
        //    Surface_charge_density_intro = surface_inj_fraction*Surface_charge_density
        // 3. Number_of_particles = surface_charge_density_intro * area
        //                        / ( charge * particle_wt=scale_fac)
        amrex::Real sigma_inj = Er_cell - Er_cor;
        amrex::Real sigma_diff = sigma_GJ/PhysConst::ep0;
        amrex::Real max_dens = PulsarParm::max_ndens;
        // Only a fraction of the surface charge density is introduced per timestep.
        // This value is defined by the user.
        amrex::Real surface_inj_fraction = PulsarParm::Ninj_fraction;
        // Total number of particles injected (float)
        //amrex::Real N_inj = surface_inj_fraction * amrex::Math::abs(sigma_inj)
        //amrex::Real N_inj = surface_inj_fraction * amrex::Math::abs(sigma_inj)
        //                        * PhysConst::ep0 * dx[0]*dx[0]
        //                        / (PhysConst::q_e * max_dens * scale_fac);
        amrex::Real N_inj = num_ppc ;
        // Apply conditions to check if particles needs to be introduced.
        if ( N_inj > 1) {
            if ( N_inj < (num_ppc) ) {
                // if number of particles to be introduced is less than num_ppc
                // then to ensure spatial regularity, introduce every num_ppc/N_inj particle
                int part_freq = floor ( num_ppc / N_inj );
                if ( i_part % part_freq !=0 ) {
            //        if (include_part_sampling == 1 ) p.id() = -1;
                }
            }
        }
        else { // N_inj < 1 (no particles to be injected)
            p.id() = -1;
        } // end if N_inj >=1

        // For an inactive vacuum pulsar, only electron or only ions are introduced
        // if surface change density < 0, then do not introduce ions
        // Benoit Cerutti uses this approach - currently comment out.
//        if ( sigma_inj < 0 and q_pm > 0) {
//            p.id() = -1;
//            amrex::Print() << " ion not intro\n";
//        }
//        // if surface change density > 0, then do not introduce electrons
//        if ( sigma_inj > 0 and q_pm < 0) {
//            p.id() = -1;
//            amrex::Print() << " el not introd \n";
//        }
        // if rho is too small, we do not inject particles
        if (std::abs(rho_GJ) < 1.E-20) {
            p.id() = -1;
        } else {
            amrex::Real rel_rho_err = ( amrex::Math::abs(rho_arr(icell, jcell, kcell))
                                        - amrex::Math::abs(rho_GJ))
                                      / ( amrex::Math::abs(rho_GJ) );
            // if current rho is close to the rho_GJ, then particles are not introduced.
            // This is the stopping condition for particle supply.
            // But the difference could be high if for some reason rho has overshot rho_GJ.
            // So first the amplitude of rho_cell and rho_GJ is compared.
//            if ( rho_arr(icell,jcell,kcell) >0 and rho_GJ <0) p.id() = -1;
//            if ( rho_arr(icell,jcell,kcell) <0 and rho_GJ >0) p.id() = -1;
            if ( amrex::Math::abs(rho_arr(icell,jcell,kcell)) / amrex::Math::abs(rho_GJ) < 0.8) {
                if (amrex::Math::abs(rel_rho_err) < 0.05) {
//                    p.id() = -1;
                }
            } else {
//                p.id() = -1; // The absolute value of rho > rho_GJ
            }
        }
    } // end of operator function.


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator () ( const GpuArray<Real,AMREX_SPACEDIM> overlap_corner,
                       int i, int j, int k, const GpuArray<Real,AMREX_SPACEDIM> dx,
                       amrex::Real t,
                       amrex::Array4<const amrex::Real> const& ex_arr,
                       amrex::Array4<const amrex::Real> const& ey_arr,
                       amrex::Array4<const amrex::Real> const& ez_arr,
                       amrex::Array4<const amrex::Real> const& rho_arr,
                       int icell, int jcell, int kcell,
                       const amrex::Real q_pm, int num_ppc) const noexcept
    {
        int part_counter = 0;
        for (int i_part = 0; i_part < num_ppc; ++i_part) {
            ParticleType dummy_p;
            dummy_p.id() = 1;
            this->operator()( overlap_corner, i, j, k, dx, t, ex_arr, ey_arr, ez_arr, rho_arr,
                  icell, jcell, kcell, dummy_p, q_pm, i_part, num_ppc, 0);
            if ( dummy_p.id() > 0) ++part_counter;
        }
//        amrex::Print() << " partcounter : " << part_counter << "\n";
        return part_counter;
    }
};

#endif

