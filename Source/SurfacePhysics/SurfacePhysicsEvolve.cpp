#include "SurfacePhysicsBase.H"
#include "WarpX.H"

void
SurfacePhysicsBase::EvolveSurfacePhysics (amrex::Real cur_time)
{
    if (cur_time < m_start_time) return;
    amrex::Print() << " in evolve surface physics \n";
   // surface density = Sites = 1e19/m^2 - input at initialization
   // loop over surface species
   //     parallelize over all mesh elements
   //         loop over all reactions
   //            is surface species a reactant and there is no surface species in prodcut, no change
   //                                              there is a different surface species in the product (loss) 
   //            else if surface species is not a reactant but only in product (gain) 
   //            else if neither a reactant nor a product, skip to next reaction
   //            loop over reactants
   //                *=Gamma gos gas reactant
   //                *=N for surface reactant
   //            * reacttions probability / Sites
   //            prefactor = 1 for gain, -1 for loss, 0 for no surface species in product
   //            dN += prefactor * reac_term
   //         for this species, update N = dt * reac_term + old_N
   //
   //
   //
   //
    int num_rxns          = static_cast<int>(reactions.size());
    int max_r             = m_max_reactants_per_rxn;
    const amrex::Real* rxn_P0       = m_rxn_P0.data();
    const amrex::Real* rxn_E_ref    = m_rxn_E_ref.data();
    const amrex::Real* rxn_E_th     = m_rxn_E_th.data();
    const amrex::Real* rxn_exp_arr  = m_rxn_exp.data();
    const int* rxn_num_react        = m_rxn_num_reactants.data();
    const int* react_is_gas         = m_reactant_is_gas.data();
    const int* react_sp_val         = m_reactant_sp_val.data();
    int* rxn_has_surface_products   = reaction_has_surface_products.data();
    int* surf_sp_is_reactant        = surface_sp_is_reactant.data();
    int* surf_sp_is_product         = surface_sp_is_product.data();
    int* rxn_has_gas_prod           = reaction_has_gas_products.data();
    int* gas_is_prod                = gas_sp_is_product.data();
    computeInflux();
    const amrex::Geometry& geom = WarpX::GetInstance().Geom(0);
    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();

    amrex::Print() << " start time " << m_start_time << " chem dt " << m_chem_dt << " end time " << m_end_time << "\n"; 
    m_cur_time = m_start_time;
    for (int istep = m_start_time/m_chem_dt; istep < m_end_time/m_chem_dt; istep ++ ) {
    const amrex::Real site_density = m_surface_site_density;
    const amrex::Real flux = m_plasma_influx;
    const amrex::Real E_in = m_plasma_Ein ; //* 1.6022e-19; // eV to Joules
    amrex::Real dt = m_chem_dt;
    int num_surf_elements = static_cast<int>(surf_ijk.size());

    // Surface species evolution
    for ( int isp = 0; isp < static_cast<int>(surface_species_vec.size()); ++isp) {
        const auto& [s_sp, val] = surface_species_vec[isp];
        amrex::Real* sp_surf_density = m_surface_density_fraction.data();
        amrex::Real* sp_influx = m_incoming_flux.data();
        amrex::ParallelFor(surf_ijk.size(),
        [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            amrex::Real dN = 0;
            for (int irxn = 0; irxn < num_rxns ; ++irxn) {
                amrex::Real react_term = 1.;
// to delete                const Reaction& rxn = reactions[irxn];
                amrex::Real prefactor = 0.; // no surface species in product, no change to surface density
                if (rxn_has_surface_products[irxn] == 1) {
                    if (surf_sp_is_reactant[isp * num_rxns + irxn] == 0 &&
                        surf_sp_is_product[isp * num_rxns + irxn] == 0) {
                        continue;
                    }
                    if (surf_sp_is_reactant[isp * num_rxns + irxn] == 1) {
                        if (surf_sp_is_product[isp * num_rxns + irxn] == 0) {
                            prefactor = -1.; // loss
                        }
                        // if reactant and product, prefactor set to 0 meaning no change
                        // so even though we have not explicitly handled this case, its inherently accounted for
                    } else {
                        if (surf_sp_is_product[isp * num_rxns + irxn] == 1) {
                            prefactor = 1.; // gain
                        }
                    }
                }
                amrex::Real exp_val = rxn_exp_arr[irxn];
                amrex::Real reaction_prob = rxn_P0[irxn]
                                * (std::pow(E_in,exp_val) - std::pow(rxn_E_th[irxn],exp_val))
                                / (std::pow(rxn_E_ref[irxn],exp_val) - std::pow(rxn_E_th[irxn],exp_val));
                if (reaction_prob > 0) {
                    for (int ir = 0; ir < rxn_num_react[irxn]; ir++) {
                        int index = react_sp_val[irxn * max_r + ir] * num_surf_elements + i;
                        react_term *= (react_is_gas[irxn * max_r + ir] == 1) ?
                                          sp_influx[index] : sp_surf_density[index];
                    }
                    react_term *= prefactor * reaction_prob / site_density;
                } else {
                    dN = 0.;
                }
                dN += react_term;
            }
            sp_surf_density[isp*num_surf_elements + i] += dN * dt;
        });
    }
   
    {
        amrex::Vector<amrex::Real> h_surf_dens(m_surface_density_fraction.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                         m_surface_density_fraction.begin(),
                         m_surface_density_fraction.end(),
                         h_surf_dens.begin());
//        for (int is = 0; is < num_surf_elements; ++is ) {
//            amrex::PrintToFile("surface_evolution.txt") << istep << " " << m_cur_time << " " ;
//            for (int s_sp = 0; s_sp < static_cast<int>(surface_species_vec.size()); s_sp++) {
//                amrex::PrintToFile("surface_evolution.txt") << surface_species_vec[s_sp].first << " ";
//                amrex::PrintToFile("surface_evolution.txt") << h_surf_dens[s_sp*surf_ijk.size()+is] << " ";
//            }
//            amrex::PrintToFile("surface_evolution.txt") << "\n";
//        }
        for (int is = 0; is < num_surf_elements; ++is ) {
            const amrex::IntVect& iv = surf_ijk[is];
#if defined(WARPX_DIM_3D)
            const amrex::Real x = plo[0] + (iv[0] + 0.5) * dx[0];
            const amrex::Real y = plo[1] + (iv[1] + 0.5) * dx[1];
            const amrex::Real z = plo[2] + (iv[2] + 0.5) * dx[2];
            amrex::PrintToFile("surface_evolution.txt")
                << is << " " << istep << " " << m_cur_time << " "
                << iv[0] << " " << iv[1] << " " << iv[2] << " "
                << x << " " << y << " " << z << " ";
#elif defined(WARPX_DIM_XZ)
            const amrex::Real x = plo[0] + (iv[0] + 0.5) * dx[0];
            const amrex::Real z = plo[1] + (iv[1] + 0.5) * dx[1];
            amrex::PrintToFile("surface_evolution.txt")
                << is << " " << istep << " " << m_cur_time << " "
                << iv[0] << " " << iv[1] << " "
                << x << " " << z << " ";
#endif
            for (int s_sp = 0; s_sp < static_cast<int>(surface_species_vec.size()); s_sp++) {
                amrex::PrintToFile("surface_evolution.txt") << surface_species_vec[s_sp].first << " ";
                amrex::PrintToFile("surface_evolution.txt") << h_surf_dens[s_sp*surf_ijk.size()+is] << " ";
            }
            amrex::PrintToFile("surface_evolution.txt") << "\n";
        }
    }

   // now compute the returning Gammma for each gas species (ion and neutral)
   //
   // loop over all the gas species
   //     parallelize over all mesh elements
   //         loop over all the reactions
   //             is gas species a product
   //                 loop over reactants
   //                 if reactant is a gas *=Gamma
   //                                  species *=N
   //                  *=p(E) / Sites
   //             ignore gas species as reactant - i.e., returning flux for gas species not participating in a reaction i.e, 1-p(E)
   //             is gas species a reactant
   //                 *= (1-p(E)) / Sites
   //             dGammaR += 
  

    for (int isp = 0; isp < static_cast<int>(gas_species_vec.size()); ++isp) {
        const auto& [s_sp, val] = gas_species_vec[isp];
//        amrex::Print() << "isp " << isp << " g_sp " << s_sp << "val " << val<< "\n";
// to delete        int num_surf_elements = surf_ijk.size();
        amrex::Real* sp_surf_density = m_surface_density_fraction.data();
        amrex::Real* returning_flux = m_returning_gas_flux.data();
        amrex::Real* sp_influx = m_incoming_flux.data();
        amrex::ParallelFor(num_surf_elements,
        [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            amrex::Real dgamma = 0.;
            for (int irxn = 0; irxn < num_rxns; ++irxn) {
                amrex::Real prefactor = 0.;
// to delete                const Reaction& rxn = reactions[irxn];
                if (rxn_has_gas_prod[irxn] == 1) {
                    amrex::Real react_term = 0.;
                    if (gas_is_prod[isp * num_rxns + irxn] == 1) {
                        react_term = 1.;
                        prefactor = 1.;
                        amrex::Real exp_val = rxn_exp_arr[irxn];
                        amrex::Real reaction_prob = rxn_P0[irxn]
                                        * (std::pow(E_in,exp_val) - std::pow(rxn_E_th[irxn],exp_val))
                                        / (std::pow(rxn_E_ref[irxn],exp_val) - std::pow(rxn_E_th[irxn],exp_val));
                        if (reaction_prob > 0.) {
                            for (int ir = 0; ir < rxn_num_react[irxn] ; ++ir) {
                                int index = react_sp_val[irxn * max_r + ir] * num_surf_elements + i;
                                react_term *= (react_is_gas[irxn * max_r + ir] == 1) ? sp_influx[index] : sp_surf_density[index];
                            }
                            react_term *= prefactor * reaction_prob;
                        }
                    }
                    dgamma += react_term;
                }
            }
            returning_flux[isp * num_surf_elements + i] = dgamma;
        });
    }

    {
        amrex::Vector<amrex::Real> h_gas_flux(m_returning_gas_flux.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                         m_returning_gas_flux.begin(),
                         m_returning_gas_flux.end(),
                         h_gas_flux.begin());
        amrex::Vector<amrex::Real> h_gas_influx(m_incoming_flux.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                         m_incoming_flux.begin(),
                         m_incoming_flux.end(),
                         h_gas_influx.begin());
        for (int is = 0; is < num_surf_elements; ++is ) {
            const amrex::IntVect& iv = surf_ijk[is];
#if defined(WARPX_DIM_3D)
            const amrex::Real x = plo[0] + (iv[0] + 0.5) * dx[0];
            const amrex::Real y = plo[1] + (iv[1] + 0.5) * dx[1];
            const amrex::Real z = plo[2] + (iv[2] + 0.5) * dx[2];
            amrex::PrintToFile("surface_flux_evolution.txt")
                << is << " " << istep << " " << m_cur_time << " "
                << iv[0] << " " << iv[1] << " " << iv[2] << " "
                << x << " " << y << " " << z << " ";
#elif defined(WARPX_DIM_XZ)
            const amrex::Real x = plo[0] + (iv[0] + 0.5) * dx[0];
            const amrex::Real z = plo[1] + (iv[1] + 0.5) * dx[1];
            amrex::PrintToFile("surface_flux_evolution.txt")
                << is << " " << istep << " " << m_cur_time << " "
                << iv[0] << " " << iv[1] << " "
                << x << " " << z << " ";
#endif
            for (int g_sp = 0; g_sp < static_cast<int>(gas_species_vec.size()); ++g_sp) {
                amrex::PrintToFile("surface_flux_evolution.txt") << gas_species_vec[g_sp].first << " ";
                amrex::PrintToFile("surface_flux_evolution.txt") << h_gas_flux[g_sp*surf_ijk.size()+is] << " ";
            }
            amrex::PrintToFile("surface_flux_evolution.txt")  << "\n";
        }
    }
    m_cur_time += m_chem_dt;
    }  // time loop
    // Here we can reset influx collection window - nullify Influx ParticleCounter and reset influx window start time
}
