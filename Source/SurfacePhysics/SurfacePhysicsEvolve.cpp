#include "SurfacePhysicsBase.H"

void
SurfacePhysicsBase::EvolveSurfacePhysics ()
{
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
    computeInflux();
//    for (int istep = m_start_time/m_chem_dt; istep < m_end_time/m_chem_dt; istep ++ ) {
    amrex::Print() << " start time " << m_start_time << " chem dt " << m_chem_dt << " end time " << m_end_time << "\n"; 
    for (int istep = m_start_time/m_chem_dt; istep < m_end_time/m_chem_dt; istep ++ ) {
    const amrex::Real site_density = m_surface_site_density;
    const amrex::Real flux = m_plasma_influx;
    const amrex::Real E_in = m_plasma_Ein ; //* 1.6022e-19; // eV to Joules
    int* rxn_has_surface_products = reaction_has_surface_products.data();
    int* surf_sp_is_reactant = surface_sp_is_reactant.data(); 
    int* surf_sp_is_product = surface_sp_is_product.data(); 
    // initialize surface density for the two surface species and change them
    amrex::Real dt = m_chem_dt;
    for ( int isp = 0; isp < surface_species_vec.size(); ++isp) {
        const auto& [s_sp, val] = surface_species_vec[isp];
        amrex::Real* sp_surf_density = m_surface_density_fraction.data();
//        amrex::Real* sp_influx = bnd_influx[isp].data();
        amrex::Real* sp_influx = m_incoming_flux.data();
        int num_surf_elements = surf_ijk.size();
        amrex::ParallelFor(surf_ijk.size(),
        [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            amrex::Real dN = 0;
            for (int irxn = 0; irxn < reactions.size(); ++irxn) {
                amrex::Real react_term = 1.;
                const Reaction& rxn = reactions[irxn];
                amrex::Real prefactor = 0.; // no surface species in product, no change to surface density
                if (rxn_has_surface_products[irxn] == 1) {
                    if (surface_sp_is_reactant[isp * reactions.size() + irxn] == 0 && 
                        surface_sp_is_product[isp * reactions.size() + irxn] == 0) {
                        continue;
                    }
                    if (surface_sp_is_reactant[isp * reactions.size() + irxn] == 1) {
                        if (surface_sp_is_product[isp * reactions.size() + irxn] == 0) {
                            prefactor = -1.; // loss
                        }
                        // if reactant and product, prefactor set to 0 meaning no change
                        // so even though we have not explicitly handled this case, its inherently accounted for
                    } else {
                        if (surface_sp_is_product[isp * reactions.size() + irxn] == 1) {
                            prefactor = 1.; // gain
                        }
                    }
                }
                amrex::Real exp = rxn.exp;
                amrex::Real reaction_prob = rxn.P0
                                          * (std::pow(E_in,exp) - std::pow(rxn.E_th,exp))
                                          / (std::pow(rxn.E_ref,exp) - std::pow(rxn.E_th,exp)); 
                if (reaction_prob > 0) {
                    for (int ir = 0; ir < rxn.reactants.size(); ir++) {
                        const std::string& reactant = rxn.reactants[ir];
                        const std::string& reactant_type = rxn.reactant_type[ir];
                        int index = -1;
                        //if (reactant_type == "surface") {
                        index = rxn.reactant_sp_val[ir]*num_surf_elements + i;
                        //}
                        //react_term *= (reactant_type == "gas") ? flux : sp_surf_density[index];
                        react_term *= (reactant_type == "gas") ? sp_influx[index] : sp_surf_density[index];
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
   
    for (int is = 0; is < surf_ijk.size(); ++is ) {
        auto& [s_sp, val] = surface_species_vec[0];
        auto& [as_sp, aval] = surface_species_vec[1];
        amrex::PrintToFile("surface_evolution.txt") << istep << " " << m_cur_time << " " << m_surface_density_fraction[0*surf_ijk.size()+is] << " " << m_surface_density_fraction[1*surf_ijk.size()+is] << "\n";
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
  

    for (int isp = 0; isp < gas_species_vec.size(); ++isp) {
        const auto& [s_sp, val] = gas_species_vec[isp];
//        amrex::Print() << "isp " << isp << " g_sp " << s_sp << "val " << val<< "\n";
        int num_surf_elements = surf_ijk.size();
        amrex::Real* sp_surf_density = m_surface_density_fraction.data();
        amrex::Real* returning_flux = m_returning_gas_flux.data();
        amrex::Real* sp_influx = m_incoming_flux.data();
        amrex::ParallelFor(surf_ijk.size(),
        [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            amrex::Real dgamma = 0.;
            for (int irxn = 0; irxn < reactions.size(); ++irxn) {
                amrex::Real prefactor = 0.;
                const Reaction& rxn = reactions[irxn];
//                if (i == 0) amrex::Print() << " rxn : " << rxn.equation << "\n";
//                if (i == 0) amrex::Print() << " rxn has gas prod " << reaction_has_gas_products[irxn] << "\n";
                if (reaction_has_gas_products[irxn] == 1) {
//                    if (i == 0) amrex::Print() << " rxn has gas prod " << reaction_has_gas_products[irxn] << "\n";
                    amrex::Real react_term = 0.;
//                    if (i == 0) amrex::Print() << "gas sp is prod : " << gas_sp_is_product[isp * reactions.size() + irxn] << "\n";
                    if (gas_sp_is_product[isp * reactions.size() + irxn] == 1) {
                        react_term = 1.;
                        prefactor = 1.;
//                        if (i == 0) amrex::Print() << " prefactor " << prefactor << "\n"; 
                        amrex::Real exp = rxn.exp;
                        amrex::Real reaction_prob = rxn.P0
                                                  * (std::pow(E_in,exp) - std::pow(rxn.E_th,exp))
                                                  / (std::pow(rxn.E_ref,exp) - std::pow(rxn.E_th,exp));
//                        if (i == 0) amrex::Print() << " reaction prob " << reaction_prob << "\n";
                        if (reaction_prob > 0.) {
                            for (int ir = 0; ir < rxn.reactants.size() ; ++ir) {
                                const std::string& reactant = rxn.reactants[ir];
                                const std::string& reactant_type = rxn.reactant_type[ir];
//                                if (i == 0) amrex::Print() << " reactant " << reactant << " type " << reactant_type << "\n";
                                int index = -1;
                                index = rxn.reactant_sp_val[ir]*num_surf_elements + i;
//                                react_term *= (reactant_type == "gas") ? flux : sp_surf_density[index];
                                react_term *= (reactant_type == "gas") ? sp_influx[index] : sp_surf_density[index];
//                                if (reactant_type == "gas" ) {
//                                    if (i == 0) amrex::Print() << " flux " << flux << " react " << react_term << "\n";
//                                } else {
//                                    if (i == 0) amrex::Print() << " sp surf density " << sp_surf_density[index] << " reac " << react_term << "\n";
//                                }
                            }
//                            if (i == 0) amrex::Print() << " react term / site density * pe " << react_term << "\n";
                            react_term *= prefactor * reaction_prob;
//                            if (i == 0) amrex::Print() << " react term " << react_term << "\n";
                        }
//                        if (i == 0) amrex::Print() << " irxn : " << irxn << " reac term : " << react_term << "\n";
                    }
                    dgamma += react_term;
//                    if (i == 0) amrex::Print() << " dgamma : " << dgamma << "\n";
                } else {
//                    if (i == 0) amrex::Print() << "next reaction \n";
                }
            }
            //if (i == 0) amrex::Print() << " dgamma " << dgamma << "\n";
            returning_flux[isp * num_surf_elements + i] = dgamma;
            //if (i == 0) amrex::Print() << " isp : " << isp << " " << returning_flux[isp * num_surf_elements + i] << "\n";
        });
    }

    for (int is = 0; is < surf_ijk.size(); ++is ) {
        amrex::PrintToFile("surface_flux_evolution.txt") << istep << " " << m_cur_time << " " << m_returning_gas_flux[0*surf_ijk.size()+is] << " " << m_returning_gas_flux[1*surf_ijk.size()+is] << "\n";
    }
    m_cur_time += m_chem_dt;
    }  // time loop
}
