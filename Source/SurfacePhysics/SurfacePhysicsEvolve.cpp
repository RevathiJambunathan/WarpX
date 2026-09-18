#include "SurfacePhysicsBase.H"
#include "WarpX.H"

void
SurfacePhysicsBase::computeFluxWeightedReactionRates ()
{
    int const num_rxns          = static_cast<int>(reactions.size());
    int const num_surf_elements = static_cast<int>(surf_ijk.size());
    const amrex::Real* rxn_P0       = m_rxn_P0.data();
    const amrex::Real* rxn_E_ref    = m_rxn_E_ref.data();
    const amrex::Real* rxn_E_th     = m_rxn_E_th.data();
    const amrex::Real* rxn_exp_arr  = m_rxn_exp.data();
    const int* rxn_gas_reactant_sp_idx = m_rxn_gas_reactant_sp_idx.data();

    m_rxn_flux_weighted_rate.resize(num_rxns * num_surf_elements);
    amrex::Real* rate = m_rxn_flux_weighted_rate.data();

    bool const use_ebin = m_use_energy_binned_flux;
    int const num_ebin  = m_num_energy_bins;
    const amrex::Real* sp_influx_ebin = m_incoming_flux_ebin.data();
    amrex::Real const energy_bin_min  = m_energy_bin_min;
    amrex::Real const energy_bin_size = m_energy_bin_size;
    amrex::Real const E_in = m_plasma_Ein;

    amrex::ParallelFor(num_surf_elements,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        for (int irxn = 0; irxn < num_rxns; ++irxn) {
            amrex::Real const exp_val = rxn_exp_arr[irxn];
            int const gas_sp = rxn_gas_reactant_sp_idx[irxn];

            if (use_ebin && gas_sp >= 0) {
                amrex::Real flux_weighted_rate = 0.;
                for (int ie = 0; ie < num_ebin; ++ie) {
                    amrex::Real const E_bin = energy_bin_min + (ie + 0.5) * energy_bin_size;
                    amrex::Real const P_bin = rxn_P0[irxn]
                                    * (std::pow(E_bin,exp_val) - std::pow(rxn_E_th[irxn],exp_val))
                                    / (std::pow(rxn_E_ref[irxn],exp_val) - std::pow(rxn_E_th[irxn],exp_val));
                    if (P_bin > 0) {
                        amrex::Real const flux_ebin =
                            sp_influx_ebin[(gas_sp*num_surf_elements + i)*num_ebin + ie];
                        flux_weighted_rate += P_bin * flux_ebin;
                    }
                }
                rate[irxn*num_surf_elements + i] = flux_weighted_rate;
            } else {
                rate[irxn*num_surf_elements + i] = rxn_P0[irxn]
                                * (std::pow(E_in,exp_val) - std::pow(rxn_E_th[irxn],exp_val))
                                / (std::pow(rxn_E_ref[irxn],exp_val) - std::pow(rxn_E_th[irxn],exp_val));
            }
        }
    });
}

void
SurfacePhysicsBase::EvolveSurfacePhysics (amrex::Real cur_time, int pic_step)
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
    const int* rxn_num_react        = m_rxn_num_reactants.data();
    const int* react_is_gas         = m_reactant_is_gas.data();
    const int* react_sp_val         = m_reactant_sp_val.data();
    int* rxn_has_surface_products   = reaction_has_surface_products.data();
    int* surf_sp_is_reactant        = surface_sp_is_reactant.data();
    int* surf_sp_is_product         = surface_sp_is_product.data();
    int* rxn_has_gas_prod           = reaction_has_gas_products.data();
    int* gas_is_prod                = gas_sp_is_product.data();
    const int* rxn_gas_reactant_sp_idx = m_rxn_gas_reactant_sp_idx.data();
    bool const use_ebin = m_use_energy_binned_flux;
    int num_surf_elements = static_cast<int>(surf_ijk.size());
    computeInflux();
    computeFluxWeightedReactionRates();
    amrex::Print() << "computed flux weighted reaction rate \n";
    const amrex::Geometry& geom = WarpX::GetInstance().Geom(0);
    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();

    amrex::Print() << " start time " << m_start_time << " chem dt " << m_chem_dt << " end time " << m_end_time << "\n";
    m_cur_time = m_start_time;
    for (int istep = m_start_time/m_chem_dt; istep < m_end_time/m_chem_dt; istep ++ ) {
    const amrex::Real site_density = m_surface_site_density;
    const amrex::Real flux = m_plasma_influx;
    amrex::Real dt = m_chem_dt;

    const amrex::Real* rxn_flux_weighted_rate = m_rxn_flux_weighted_rate.data();

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
                amrex::Real reaction_rate = rxn_flux_weighted_rate[irxn*num_surf_elements + i];
                bool const rxn_uses_ebin_gas_reactant =
                    use_ebin && (rxn_gas_reactant_sp_idx[irxn] >= 0);
                if (reaction_rate > 0) {
                    for (int ir = 0; ir < rxn_num_react[irxn]; ir++) {
                        int const sp_val = react_sp_val[irxn * max_r + ir];
                        int index = sp_val * num_surf_elements + i;
                        bool const is_gas = (react_is_gas[irxn * max_r + ir] == 1);
                        // when using energy-binned flux, the gas reactant's flux is
                        // already folded into reaction_rate; skip it here
                        if (rxn_uses_ebin_gas_reactant && is_gas &&
                            sp_val == rxn_gas_reactant_sp_idx[irxn]) {
                            continue;
                        }
                        react_term *= is_gas ? sp_influx[index] : sp_surf_density[index];
                    }
                    react_term *= prefactor * reaction_rate / site_density;
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
        if (!m_surface_evolution_header_written) {
#if defined(WARPX_DIM_3D)
            amrex::PrintToFile("surface_evolution.txt")
                << "surface_mesh_id chem_istep chem_physical_time i j k x y z ";
#elif defined(WARPX_DIM_XZ)
            amrex::PrintToFile("surface_evolution.txt")
                << "surface_mesh_id chem_istep chem_physical_time i j x z ";
#endif
            for (int s_sp = 0; s_sp < static_cast<int>(surface_species_vec.size()); s_sp++) {
                amrex::PrintToFile("surface_evolution.txt")
                    << surface_species_vec[s_sp].first << "_name "
                    << surface_species_vec[s_sp].first << "_density_fraction ";
            }
            amrex::PrintToFile("surface_evolution.txt") << "\n";
            m_surface_evolution_header_written = true;
        }
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
                        amrex::Real reaction_rate = rxn_flux_weighted_rate[irxn*num_surf_elements + i];
                        bool const rxn_uses_ebin_gas_reactant =
                            use_ebin && (rxn_gas_reactant_sp_idx[irxn] >= 0);
                        if (reaction_rate > 0.) {
                            for (int ir = 0; ir < rxn_num_react[irxn] ; ++ir) {
                                int const sp_val = react_sp_val[irxn * max_r + ir];
                                int index = sp_val * num_surf_elements + i;
                                bool const is_gas = (react_is_gas[irxn * max_r + ir] == 1);
                                // when using energy-binned flux, the gas reactant's flux is
                                // already folded into reaction_rate; skip it here
                                if (rxn_uses_ebin_gas_reactant && is_gas &&
                                    sp_val == rxn_gas_reactant_sp_idx[irxn]) {
                                    continue;
                                }
                                react_term *= is_gas ? sp_influx[index] : sp_surf_density[index];
                            }
                            react_term *= prefactor * reaction_rate;
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
        if (!m_gas_influx_surface_written) {
            int const num_ebin = m_num_energy_bins;

            amrex::Vector<amrex::Real> h_rxn_flux_weighted_rate(m_rxn_flux_weighted_rate.size());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                             m_rxn_flux_weighted_rate.begin(),
                             m_rxn_flux_weighted_rate.end(),
                             h_rxn_flux_weighted_rate.begin());

            amrex::Vector<amrex::Real> h_gas_influx_ebin;
            if (num_ebin > 0) {
                h_gas_influx_ebin.resize(m_incoming_flux_ebin.size());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                                 m_incoming_flux_ebin.begin(),
                                 m_incoming_flux_ebin.end(),
                                 h_gas_influx_ebin.begin());
            }

            const std::string gas_influx_fname =
                "gas_influx_surface_step" + std::to_string(pic_step) + ".txt";

            amrex::PrintToFile(gas_influx_fname) << "pic_step " << pic_step << "\n";

            for (int irxn = 0; irxn < num_rxns; ++irxn) {
                amrex::PrintToFile(gas_influx_fname)
                    << "# rxn" << irxn << ": " << reactions[irxn].equation << "\n";
            }
            if (num_ebin > 0) {
                amrex::PrintToFile(gas_influx_fname)
                    << "# energy_bins: num_bins=" << num_ebin
                    << " bin_min=" << m_energy_bin_min
                    << " bin_max=" << m_energy_bin_max
                    << " bin_size=" << m_energy_bin_size << "\n";
            }

            // ---- Table 1: per-surface-element summary, always written ----
#if defined(WARPX_DIM_3D)
            amrex::PrintToFile(gas_influx_fname)
                << "surface_mesh_id i j k x y z ";
#elif defined(WARPX_DIM_XZ)
            amrex::PrintToFile(gas_influx_fname)
                << "surface_mesh_id i j x z ";
#endif
            for (int g_sp = 0; g_sp < static_cast<int>(gas_species_vec.size()); ++g_sp) {
                amrex::PrintToFile(gas_influx_fname)
                    << gas_species_vec[g_sp].first << "_name "
                    << gas_species_vec[g_sp].first << "_influx ";
            }
            for (int irxn = 0; irxn < num_rxns; ++irxn) {
                amrex::PrintToFile(gas_influx_fname) << "rxn" << irxn << "_rate ";
            }
            amrex::PrintToFile(gas_influx_fname) << "\n";

            for (int is = 0; is < num_surf_elements; ++is) {
                const amrex::IntVect& iv = surf_ijk[is];
#if defined(WARPX_DIM_3D)
                const amrex::Real x = plo[0] + (iv[0] + 0.5) * dx[0];
                const amrex::Real y = plo[1] + (iv[1] + 0.5) * dx[1];
                const amrex::Real z = plo[2] + (iv[2] + 0.5) * dx[2];
                amrex::PrintToFile(gas_influx_fname)
                    << is << " " << iv[0] << " " << iv[1] << " " << iv[2] << " "
                    << x << " " << y << " " << z << " ";
#elif defined(WARPX_DIM_XZ)
                const amrex::Real x = plo[0] + (iv[0] + 0.5) * dx[0];
                const amrex::Real z = plo[1] + (iv[1] + 0.5) * dx[1];
                amrex::PrintToFile(gas_influx_fname)
                    << is << " " << iv[0] << " " << iv[1] << " "
                    << x << " " << z << " ";
#endif
                for (int g_sp = 0; g_sp < static_cast<int>(gas_species_vec.size()); ++g_sp) {
                    amrex::PrintToFile(gas_influx_fname) << gas_species_vec[g_sp].first << " ";
		    int const plasma_sp = m_chem_gas_sp_to_plasma_sp_idx[g_sp];
                    amrex::Real const influx_val =
                        (plasma_sp >= 0) ? h_gas_influx[plasma_sp*surf_ijk.size()+is] : 0.;
                    amrex::PrintToFile(gas_influx_fname) << influx_val << " ";
                }
                for (int irxn = 0; irxn < num_rxns; ++irxn) {
                    amrex::PrintToFile(gas_influx_fname)
                        << h_rxn_flux_weighted_rate[irxn*num_surf_elements + is] << " ";
                }
                amrex::PrintToFile(gas_influx_fname) << "\n";
            }

            // ---- Table 2 (long format): one row per (surface_mesh_id, bin_index),
            //      only written when energy binning is enabled ----
            if (num_ebin > 0) {
                amrex::PrintToFile(gas_influx_fname) << "\n";
                amrex::PrintToFile(gas_influx_fname)
                    << "surface_mesh_id bin_index bin_energy_center ";
                for (int g_sp = 0; g_sp < static_cast<int>(gas_species_vec.size()); ++g_sp) {
                    amrex::PrintToFile(gas_influx_fname)
                        << gas_species_vec[g_sp].first << "_influx_bin ";
                }
                amrex::PrintToFile(gas_influx_fname) << "\n";

                for (int is = 0; is < num_surf_elements; ++is) {
                    for (int ie = 0; ie < num_ebin; ++ie) {
                        amrex::Real const bin_center =
                            m_energy_bin_min + (ie + 0.5) * m_energy_bin_size;
                        amrex::PrintToFile(gas_influx_fname)
                            << is << " " << ie << " " << bin_center << " ";
                        for (int g_sp = 0; g_sp < static_cast<int>(gas_species_vec.size()); ++g_sp) {
                            int const plasma_sp = m_chem_gas_sp_to_plasma_sp_idx[g_sp];
                            amrex::Real const influx_val = (plasma_sp >= 0)
                                ? h_gas_influx_ebin[(plasma_sp*num_surf_elements + is)*num_ebin + ie]
                                : 0.;
                            amrex::PrintToFile(gas_influx_fname) << influx_val << " ";
                        }
                        amrex::PrintToFile(gas_influx_fname) << "\n";
                    }
                }
            }

            m_gas_influx_surface_written = true;
        }
        if (!m_surface_flux_evolution_header_written) {
#if defined(WARPX_DIM_3D)
            amrex::PrintToFile("surface_flux_evolution.txt")
                << "surface_mesh_id chem_istep chem_physical_time i j k x y z ";
#elif defined(WARPX_DIM_XZ)
            amrex::PrintToFile("surface_flux_evolution.txt")
                << "surface_mesh_id chem_istep chem_physical_time i j x z ";
#endif
            for (int g_sp = 0; g_sp < static_cast<int>(gas_species_vec.size()); ++g_sp) {
                amrex::PrintToFile("surface_flux_evolution.txt")
                    << gas_species_vec[g_sp].first << "_name "
                    << gas_species_vec[g_sp].first << "_flux ";
            }
            amrex::PrintToFile("surface_flux_evolution.txt") << "\n";
            m_surface_flux_evolution_header_written = true;
        }
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
