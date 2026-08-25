/* Copyright 2024 Revathi Jambunathan
 *
 * This file is part of WarpX
 *
 * License: BSD-3-Clause-LBNL
 */

#ifdef WARPX_SURFACE_PHYSICS

#include "SurfacePhysicsBase.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

SurfacePhysicsBase::SurfacePhysicsBase ()
{
    amrex::Print() << " in surface physics base class \n";
    ReadParameters();
}

void SurfacePhysicsBase::ReadParameters ()
{
    amrex::ParmParse const pp_surface_chemistry("surface_chemistry");
    std::string chemistry_file;
    pp_surface_chemistry.query("input_file", chemistry_file);

    amrex::ParmParse::addfile(chemistry_file);
    amrex::ParmParse const pp_chem("chem");

    // Read gas species that participate in gas-surface physics
    amrex::Vector<std::string> chem_gas_species;
    pp_chem.queryarr("gasphase_species", chem_gas_species);
    amrex::ParmParse const pp_gasphase("gasphase_species");
    for (const auto& species : chem_gas_species) {
        std::string symbol;
        //pp_chem.query(("gasphase_species."+ species + ".symbol").c_str(), symbol);
        utils::parser::query(pp_gasphase, species, "symbol", symbol);
//        gas_species[species] = symbol;
        gas_species_vec.emplace_back(species,symbol);
    }
    m_num_gas_species = chem_gas_species.size();


    // Read surface species that participate in gas-surface physics
    amrex::Vector<std::string> chem_surface_species;
    pp_chem.queryarr("surface_species", chem_surface_species);
    amrex::ParmParse const pp_surface("surface_species");
    for (const auto& species : chem_surface_species) {
        std::string symbol;
        utils::parser::query(pp_surface, species, "symbol", symbol);
        amrex::Print() << " species : " << species << " " << symbol << "\n";
        surface_species_vec.emplace_back(species,symbol);
    }
    m_num_surface_species = chem_surface_species.size();
    amrex::Print() << " num surface species : " << m_num_surface_species << "\n";

    surface_species_fraction.resize(m_num_surface_species);
    pp_chem.queryarr("surface_species_fraction", surface_species_fraction);

    std::set<std::string> known_symbols;
    for (const auto& [species, symbol] : gas_species_vec) known_symbols.insert(symbol);
    for (const auto& [species, symbol] : surface_species_vec) known_symbols.insert(symbol);


    amrex::Vector<std::string> gas_surface_reactions;
    if (pp_chem.queryarr("reactions", gas_surface_reactions)) {
        for (const auto& line : gas_surface_reactions) {
            Reaction rxn;
            std::vector<std::string> equation_params = amrex::split(line,";");

            rxn.equation = amrex::trim(equation_params[0]);
            auto arrow_pos = rxn.equation.find("=>");
            if (arrow_pos == std::string::npos) {
                amrex::Abort( " Reaction eqution must contain '=>' separator.");
            }
            std::string lhs = amrex::trim(rxn.equation.substr(0,arrow_pos));
            std::string rhs = amrex::trim(rxn.equation.substr(arrow_pos+2));
            rxn.reactants = tokenize_reaction(lhs);
            rxn.products = tokenize_reaction(rhs);
            for (const auto& reactant : rxn.reactants) {
                std::string species_type = is_gas_species(reactant) ? "gas" : "surface";
                rxn.reactant_type.push_back(species_type);
                rxn.reactant_sp_val.push_back(-1);
            }
            for (const auto& product : rxn.products) {
                if (known_symbols.find(product) == known_symbols.end()) {
                    amrex::Abort("Unknown product in reaction"+ product+"\n");
                }
                std::string species_type = is_gas_species(product) ? "gas" : "surface";
                rxn.product_type.push_back(species_type);
                amrex::Print() << " product : " << product << " species_type : " << species_type << "\n";
            }

            rxn.P_energy0 = std::stod(amrex::trim(equation_params[1]));
            rxn.P0        = std::stod(amrex::trim(equation_params[2]));
            rxn.E_ref     = std::stod(amrex::trim(equation_params[3]));
            rxn.E_th      = std::stod(amrex::trim(equation_params[4]));
            rxn.exp       = std::stod(amrex::trim(equation_params[5]));

            reactions.push_back(rxn);
        }
    } else {
        amrex::Print() << " no reactions specified for surface physics \n";
    }



    // Use host-side temporaries — Gpu::DeviceVector cannot be written from host code
    int const num_rxns   = static_cast<int>(reactions.size());
    int const num_surf_sp = static_cast<int>(surface_species_vec.size());
    int const num_gas_sp  = static_cast<int>(gas_species_vec.size());

    amrex::Vector<int> h_reaction_has_surface_products(num_rxns, 0);
    amrex::Vector<int> h_reaction_has_gas_products(num_rxns, 0);
    amrex::Vector<int> h_surface_sp_is_reactant(num_surf_sp * num_rxns, 0);
    amrex::Vector<int> h_surface_sp_is_product(num_surf_sp * num_rxns, 0);
    amrex::Vector<int> h_gas_sp_is_product(num_gas_sp * num_rxns, 0);

    for (int irxn = 0; irxn < num_rxns; irxn++) {
        const Reaction& rxn = reactions[irxn];
        amrex::Print() << " Reaction " << irxn << "\n";
        for (int ip = 0; ip < (int)rxn.product_type.size(); ++ip) {
            if (rxn.product_type[ip] == "surface") {
                h_reaction_has_surface_products[irxn] = 1;
            }
        }
    }

    for (int irxn = 0; irxn < num_rxns; irxn++) {
        const Reaction& rxn = reactions[irxn];
        amrex::Print() << " Reaction " << irxn << "\n";
        for (int ip = 0; ip < (int)rxn.product_type.size(); ++ip) {
            amrex::Print() << " prod type " << rxn.product_type[ip] << "\n";
            if (rxn.product_type[ip] == "gas") {
                h_reaction_has_gas_products[irxn] = 1;
            }
        }
    }

    for (int irxn = 0; irxn < num_rxns; irxn++) {
        Reaction& rxn = reactions[irxn];
        amrex::Print() << " Reaction " << irxn << "\n";
        for (int ir = 0; ir < (int)rxn.reactant_type.size(); ++ir) {
            rxn.reactant_sp_val[ir] = -1;
            if (rxn.reactant_type[ir] == "surface") {
                amrex::Print() << " surf ir :  " << rxn.reactants[ir] << "\n";
                int index = -1;
                for (size_t i = 0; i < surface_species_vec.size(); i++){
                    if (surface_species_vec[i].second == rxn.reactants[ir]) {
                        index = static_cast<int>(i);
                        continue;
                    }
                }
                rxn.reactant_sp_val[ir] = index;
                amrex::Print() << " for surf rxtnt : index is : " << index << "\n";
            } else if (rxn.reactant_type[ir] == "gas") {
                amrex::Print() << " gas ir : " << rxn.reactants[ir] << "\n";
                int index = -1;
                for (size_t i = 0; i < gas_species_vec.size(); i++){
                    if (gas_species_vec[i].second == rxn.reactants[ir]) {
                        index = static_cast<int>(i);
                        continue;
                    }
                }
                rxn.reactant_sp_val[ir] = index;
                amrex::Print() << " for gas rxtnt : index is : " << rxn.reactant_sp_val[ir] << "\n";
            }
        }
    }

    for (int isp = 0; isp < num_surf_sp; ++isp) {
        const std::string& symbol = surface_species_vec[isp].second;
        const std::string& name = surface_species_vec[isp].first;
        amrex::Print() << " symbol : " << symbol << "\n";
        amrex::Print() << " name " << name << "\n";
        for (int irxn = 0; irxn < num_rxns; irxn++) {
            const Reaction& rxn = reactions[irxn];
            amrex::Print() << " eq : " << rxn.equation << "\n";
            bool found = (std::find(rxn.reactants.begin(), rxn.reactants.end(), symbol) != rxn.reactants.end());
            amrex::Print() << " found ? " << found << "\n";
            h_surface_sp_is_reactant[isp*num_rxns + irxn] = found ? 1 : 0;
            bool prod_found = (std::find(rxn.products.begin(), rxn.products.end(), symbol) != rxn.products.end());
            h_surface_sp_is_product[isp*num_rxns + irxn] = prod_found ? 1 : 0;
        }
    }
    for (int isp = 0; isp < num_surf_sp; ++isp) {
        const std::string& symbol = surface_species_vec[isp].second;
        amrex::Print() << " symbol : " << symbol << "\n";
        for (int irxn = 0; irxn < num_rxns; irxn++) {
            const Reaction& rxn = reactions[irxn];
            amrex::Print() << " eq : " << rxn.equation << "\n";
            amrex::Print() << " reaction has surface ? " << h_reaction_has_surface_products[irxn] << "\n";
            amrex::Print() << "is reactant : " << h_surface_sp_is_reactant[isp*num_rxns + irxn] << "\n";
            amrex::Print() << "is product : " << h_surface_sp_is_product[isp*num_rxns + irxn] << "\n";
        }
    }

    for (int isp = 0; isp < num_gas_sp; ++isp) {
        const std::string& symbol = gas_species_vec[isp].second;
        for (int irxn = 0; irxn < num_rxns; ++irxn) {
            const Reaction& rxn = reactions[irxn];
            bool prod_found = (std::find(rxn.products.begin(), rxn.products.end(), symbol) != rxn.products.end());
            h_gas_sp_is_product[isp * num_rxns + irxn] = prod_found ? 1 : 0;
        }
    }

    // for testing purposes reading in constant values for site density, influx, and plasma_Ein
    pp_chem.get("surface_site_density", m_surface_site_density);
    pp_chem.get("plasma_influx", m_plasma_influx);
    pp_chem.get("plasma_Ein", m_plasma_Ein);
    pp_chem.get("dt", m_chem_dt);
    pp_chem.get("start_time",m_start_time);
    pp_chem.get("end_time",m_end_time);
    pp_chem.get("influx_window_start_time",m_influx_window_start_time);
    m_cur_time = 0.;

    int max_r = 0;
    for (const auto& rxn : reactions) {
        max_r = std::max(max_r, static_cast<int>(rxn.reactants.size()));
    }
    m_max_reactants_per_rxn = max_r;

    amrex::Vector<amrex::Real> h_rxn_P0(num_rxns);
    amrex::Vector<amrex::Real> h_rxn_E_ref(num_rxns);
    amrex::Vector<amrex::Real> h_rxn_E_th(num_rxns);
    amrex::Vector<amrex::Real> h_rxn_exp(num_rxns);
    amrex::Vector<int> h_rxn_num_reactants(num_rxns);
    amrex::Vector<int> h_reactant_is_gas(num_rxns * max_r, 0);
    amrex::Vector<int> h_reactant_sp_val(num_rxns * max_r, -1);

    for (int irxn = 0; irxn < num_rxns; ++irxn) {
        const auto& rxn = reactions[irxn];
        h_rxn_P0[irxn]    = rxn.P0;
        h_rxn_E_ref[irxn] = rxn.E_ref;
        h_rxn_E_th[irxn]  = rxn.E_th;
        h_rxn_exp[irxn]   = rxn.exp;
        h_rxn_num_reactants[irxn] = static_cast<int>(rxn.reactants.size());
        for (int ir = 0; ir < h_rxn_num_reactants[irxn]; ++ir) {
            h_reactant_is_gas[irxn*max_r + ir] = (rxn.reactant_type[ir] == "gas") ? 1 : 0;
            h_reactant_sp_val[irxn*max_r + ir] = rxn.reactant_sp_val[ir];
        }
    }

    // Copy all host-side data to device vectors
    reaction_has_surface_products.resize(num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_reaction_has_surface_products.begin(),
                     h_reaction_has_surface_products.end(), reaction_has_surface_products.begin());

    reaction_has_gas_products.resize(num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_reaction_has_gas_products.begin(),
                     h_reaction_has_gas_products.end(), reaction_has_gas_products.begin());

    surface_sp_is_reactant.resize(num_surf_sp * num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_surface_sp_is_reactant.begin(),
                     h_surface_sp_is_reactant.end(), surface_sp_is_reactant.begin());

    surface_sp_is_product.resize(num_surf_sp * num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_surface_sp_is_product.begin(),
                     h_surface_sp_is_product.end(), surface_sp_is_product.begin());

    gas_sp_is_product.resize(num_gas_sp * num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_gas_sp_is_product.begin(),
                     h_gas_sp_is_product.end(), gas_sp_is_product.begin());

    m_rxn_P0.resize(num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_rxn_P0.begin(), h_rxn_P0.end(), m_rxn_P0.begin());

    m_rxn_E_ref.resize(num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_rxn_E_ref.begin(), h_rxn_E_ref.end(), m_rxn_E_ref.begin());

    m_rxn_E_th.resize(num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_rxn_E_th.begin(), h_rxn_E_th.end(), m_rxn_E_th.begin());

    m_rxn_exp.resize(num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_rxn_exp.begin(), h_rxn_exp.end(), m_rxn_exp.begin());

    m_rxn_num_reactants.resize(num_rxns);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_rxn_num_reactants.begin(),
                     h_rxn_num_reactants.end(), m_rxn_num_reactants.begin());

    m_reactant_is_gas.resize(num_rxns * max_r);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_reactant_is_gas.begin(),
                     h_reactant_is_gas.end(), m_reactant_is_gas.begin());

    m_reactant_sp_val.resize(num_rxns * max_r);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_reactant_sp_val.begin(),
                     h_reactant_sp_val.end(), m_reactant_sp_val.begin());
}

int
SurfacePhysicsBase::GetChemGasSpeciesIndex (int runtime_species_id) const
{
    if (runtime_species_id < 0 ||
        runtime_species_id >= static_cast<int>(m_runtime_to_chemistry_sp_idx.size()))
    {
        return -1;
    }
    return m_runtime_to_chemistry_sp_idx[runtime_species_id];
}


bool
SurfacePhysicsBase::is_gas_species (std::string species_symbol)
{
    for (const auto& [species,symbol] : gas_species_vec) {
        if (symbol == species_symbol) return true;
    }
    return false;
}


bool
SurfacePhysicsBase::is_surface_species (std::string species_symbol)
{
    for (const auto& [species,symbol] : surface_species_vec) {
        if (symbol == species_symbol) return true;
    }
    return false;
}

amrex::Vector<std::string>
SurfacePhysicsBase::tokenize_reaction (const std::string& input) {

    amrex::Vector<std::string> result;
    std::string modified = input;

    // replacing "+_" with "%_" temporarily
    std::string::size_type pos = 0;
    while ((pos = modified.find("+_", pos)) != std::string::npos) {
        modified.replace(pos, 2, "%_");
        pos += 2;
    }

    std::vector<std::string> terms = amrex::split(modified, "+");
    for (auto& term : terms) {
        std::string restored = amrex::trim(term);
        std::string::size_type p = 0;
        while ((p = restored.find("%_", p)) != std::string::npos) {
            restored.replace(p, 2, "+_");
            p += 2;
        }
        result.push_back(restored);
    }
    return result;
}

void
SurfacePhysicsBase::InitData ()
{
    initializeMapping();
    auto & warpx = WarpX::GetInstance();
    const auto & mpc = warpx.GetPartContainer();
    num_influx_species = mpc.nSpecies();
    num_outflux_species = num_influx_species; //for now
    // Build runtime species to chemistry gas species map
    int num_runtime_species = mpc.nSpecies();
    m_runtime_to_chemistry_sp_idx.resize(num_runtime_species,-1);
    std::vector<std::string> runtime_species_names = mpc.GetSpeciesNames();
    for (int runtime_id = 0; runtime_id < num_runtime_species; ++runtime_id) {
        const std::string& runtime_name = runtime_species_names[runtime_id];
        for (int chem_id = 0; chem_id < static_cast<int>(m_num_gas_species); ++chem_id) {
            if (gas_species_vec[chem_id].first == runtime_name) {
                m_runtime_to_chemistry_sp_idx[runtime_id] = chem_id;
                break;
            }
        }
    }    
    AllocAndInitInfluxBndVectors();
    AllocAndInitOutfluxBndVectors();
    AllocAndInitSurfaceDensityFraction();
}

//void
//SurfacePhysicsBase::initializeMapping ()
//{
//    // get a reference to WarpX instance
//    auto & warpx = WarpX::GetInstance();
//
//    const int lev = 0;
//
//    // check if EB is enabled
//    if (!EB::enabled() ) {
//        amrex::Print() << " current mapping works only with EB surfaces \n";
//        return;
//    }
//    //
//    amrex::EBFArrayBoxFactory const& eb_box_factory = warpx.fieldEBFactory(lev);
//    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_box_factory.getMultiEBCellFlagFab();
//    amrex::MultiCutFab const& eb_bnd_cent = eb_box_factory.getBndryCent();
//    amrex::MultiCutFab const& eb_bnd_normal = eb_box_factory.getBndryNormal();
//
//    ivect_map = std::make_unique< amrex::iMultiFab> (warpx.boxArray(lev), warpx.DistributionMap(lev), 1, 1);
//    ivect_map->setVal(0);
//
//    for (amrex::MFIter mfi(eb_flag); mfi.isValid(); ++mfi)
//    {
//        amrex::Box const box = mfi.tilebox();
//        amrex::FabType const fab_type = eb_flag[mfi].getType(box);
//        if (fab_type == amrex::FabType::regular) { continue;}
//        else if (fab_type == amrex::FabType::covered) { continue;}
//
//        // all cells in fab are open, i.e., outside EB
//        if (fab_type == amrex::FabType::regular) {continue;}
//        // all cells in fab are enclosed within EB
//        if (fab_type == amrex::FabType::covered) {continue;}
//
//        auto const& eb_flag_arr = eb_flag.array(mfi);
//        const amrex::Array4<const amrex::Real> & eb_bnd_normal_arr = eb_bnd_normal.array(mfi);
//        auto const ivect_arr = ivect_map->array(mfi);
//
//        amrex::LoopOnCpu( box,
//            [=] (int i, int j, int k) {
//
//            amrex::IntVect const iv(AMREX_D_DECL(i,j,k));            
//            if (eb_flag_arr(i,j,k).isRegular() ) {
//                return;
//            }
//            else if (eb_flag_arr(i,j,k).isCovered() ) {
//                return;
//            }
//            else {
//                surf_ijk.push_back(iv);
//                ivect_arr(i,j,k) = surf_ijk.size() - 1;
//
//                surf_normal_x.push_back(eb_bnd_normal_arr(i,j,k,0));
//#if (defined WARPX_DIM_XZ)
//                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,1));
//#elif (defined WARPX_DIM_3D)
//                surf_normal_y.push_back(eb_bnd_normal_arr(i,j,k,1));
//                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,2));
//#endif
//            }
//        });
//    }
//    
//}

void
SurfacePhysicsBase::initializeMapping ()
{
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    const int lev = 0;

    // check if EB is enabled
    if (!EB::enabled() ) {
        amrex::Print() << " current mapping works only with EB surfaces \n";
        return;
    }

    amrex::EBFArrayBoxFactory const& eb_box_factory = warpx.fieldEBFactory(lev);
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_box_factory.getMultiEBCellFlagFab();
    amrex::MultiCutFab const& eb_bnd_normal = eb_box_factory.getBndryNormal();

    ivect_map = std::make_unique<amrex::iMultiFab>(
        warpx.boxArray(lev), warpx.DistributionMap(lev), 1, 1);
    ivect_map->setVal(0);

    for (amrex::MFIter mfi(eb_flag); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::FabType const fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue; }
        if (fab_type == amrex::FabType::covered) { continue; }

        // --- Allocate host-side (pinned) copies of the device-resident fabs ---
        amrex::BaseFab<amrex::EBCellFlag> eb_flag_host(
            box, 1, amrex::The_Pinned_Arena());
        eb_flag_host.copy<amrex::RunOn::Device>(
            eb_flag[mfi], box, 0, box, 0, 1);

        const int ncomp_n = eb_bnd_normal.nComp();
        amrex::FArrayBox eb_bnd_normal_host(
            box, ncomp_n, amrex::The_Pinned_Arena());
        eb_bnd_normal_host.copy<amrex::RunOn::Device>(
            eb_bnd_normal[mfi], box, 0, box, 0, ncomp_n);

        // ivect_map is on the device; build it on the host then copy back
        amrex::IArrayBox ivect_host(box, 1, amrex::The_Pinned_Arena());
        ivect_host.setVal<amrex::RunOn::Host>(0);

        // Wait for the D2H copies above to finish before reading on the host
        amrex::Gpu::streamSynchronize();

        auto const& eb_flag_arr       = eb_flag_host.const_array();
        auto const& eb_bnd_normal_arr = eb_bnd_normal_host.const_array();
        auto const  ivect_arr         = ivect_host.array();

        amrex::LoopOnCpu(box,
            [&] (int i, int j, int k)
            {
                if (eb_flag_arr(i,j,k).isRegular() ||
                    eb_flag_arr(i,j,k).isCovered())
                {
                    return;
                }

                amrex::IntVect const iv(AMREX_D_DECL(i,j,k));
                surf_ijk.push_back(iv);
                ivect_arr(i,j,k) = static_cast<int>(surf_ijk.size()) - 1;

                surf_normal_x.push_back(eb_bnd_normal_arr(i,j,k,0));
#if (defined WARPX_DIM_XZ)
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,1));
#elif (defined WARPX_DIM_3D)
                surf_normal_y.push_back(eb_bnd_normal_arr(i,j,k,1));
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,2));
#endif
            });

        // Push the host-built index map back to the device fab
        (*ivect_map)[mfi].copy<amrex::RunOn::Device>(
            ivect_host, box, 0, box, 0, 1);
    }

    // Make sure all H2D copies have completed before any device kernel reads ivect_map
    amrex::Gpu::streamSynchronize();
}

void
SurfacePhysicsBase::AllocAndInitInfluxBndVectors ()
{
    num_in_particles.resize(num_influx_species);
    bnd_influx.resize(num_influx_species);
    m_incoming_flux.resize(num_influx_species * surf_ijk.size());
    for (int isp = 0; isp < num_influx_species; ++isp)
    {
        num_in_particles[isp].resize(surf_ijk.size());
        bnd_influx[isp].resize(surf_ijk.size());

        nullifyInfluxParticleCounter(isp);

// FOR TESTING
        initializeInflux(isp,m_plasma_influx);
    }
}

void
SurfacePhysicsBase::AllocAndInitOutfluxBndVectors ()
{
    num_out_particles.resize(num_outflux_species);
    bnd_outflux.resize(num_outflux_species);
    for (int isp = 0; isp < num_outflux_species; ++isp)
    {
        num_out_particles[isp].resize(surf_ijk.size());
        bnd_outflux[isp].resize(surf_ijk.size());
        nullifyOutfluxParticleCounter(isp);
    }

}

void
SurfacePhysicsBase::nullifyInfluxParticleCounter ()
{
    for (int isp = 0; isp < num_influx_species; ++isp) {
        nullifyInfluxParticleCounter(isp);
    }
}

void
SurfacePhysicsBase::nullifyInfluxParticleCounter (int isp)
{
    int const num_surf = static_cast<int>(surf_ijk.size());
    amrex::Real* p_num_part = num_in_particles[isp].dataPtr();
    amrex::Real* p_bnd_influx = bnd_influx[isp].dataPtr();
    amrex::ParallelFor(num_surf,
        [=] AMREX_GPU_DEVICE (int i) noexcept{
	    p_num_part[i] = 0.;
	    p_bnd_influx[i] = 0.;
        });
}

void
SurfacePhysicsBase::initializeInflux(int isp, amrex::Real flux_val)
{
    int const num_surf = static_cast<int>(surf_ijk.size());
    int const offset   = isp * num_surf;
    amrex::Real* p_bnd_influx = bnd_influx[isp].dataPtr();
    amrex::ParallelFor(num_surf,
        [=] AMREX_GPU_DEVICE (int i) noexcept{
	    p_bnd_influx[i] = flux_val;
        });
    // m_incoming_flux is device memory; use a host temporary and copy
    amrex::Vector<amrex::Real> h_tmp(num_surf, 0.);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_tmp.begin(), h_tmp.end(),
                     m_incoming_flux.begin() + offset);
    amrex::Print() << " initialized incoming flux " << "\n";
}

void
SurfacePhysicsBase::nullifyOutfluxParticleCounter ()
{
    for (int isp = 0; isp < num_outflux_species; ++isp) {
        nullifyOutfluxParticleCounter(isp);
    }
}

void
SurfacePhysicsBase::nullifyOutfluxParticleCounter (int isp)
{    
    int const num_surf = static_cast<int>(surf_ijk.size());
    amrex::Real* p_num_out_part = num_out_particles[isp].dataPtr();
    amrex::Real* p_bnd_outflux = bnd_outflux[isp].dataPtr();
    amrex::ParallelFor(num_surf,
        [=] AMREX_GPU_DEVICE (int i) noexcept{
            p_num_out_part[i] = 0.;
            p_bnd_outflux[i] = 5.e5;
        });
}

//void
//SurfacePhysicsBase::AllocAndInitSurfaceDensityFraction ()
//{
//    m_surface_density_fraction.resize(m_num_surface_species * surf_ijk.size());
//    for (int isp = 0; isp < m_num_surface_species; ++isp) {
//        for (int i = 0; i < surf_ijk.size(); ++i) {
//            m_surface_density_fraction[isp*surf_ijk.size() + i] = surface_species_fraction[isp];
//        }
//    }
//
//    m_returning_gas_flux.resize(m_num_gas_species * surf_ijk.size());
//    for (int isp = 0; isp < m_num_gas_species; ++isp) {
//       for (int i = 0; i < surf_ijk.size() ; ++i) {
//           m_returning_gas_flux[isp * surf_ijk.size() + i] = 0.;
//       }
//    }
//}

void
SurfacePhysicsBase::AllocAndInitSurfaceDensityFraction ()
{
    const int n_surf = static_cast<int>(surf_ijk.size());
    const int n_surf_sp = m_num_surface_species;
    const int n_gas_sp  = m_num_gas_species;
    // --- Surface density fractions ---
    m_surface_density_fraction.resize(n_surf_sp * n_surf);
    // surface_species_fraction is (presumably) a host std::vector / array.
    // Copy it to the device so we can read it from a kernel.
    amrex::Gpu::DeviceVector<amrex::Real> d_species_fraction(n_surf_sp);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          surface_species_fraction.begin(),
                          surface_species_fraction.begin() + n_surf_sp,
                          d_species_fraction.begin());
    amrex::Gpu::streamSynchronize();

    amrex::Real*       p_dens_frac      = m_surface_density_fraction.dataPtr();
    amrex::Real const* p_species_frac   = d_species_fraction.dataPtr();
    amrex::ParallelFor(n_surf_sp * n_surf,
        [=] AMREX_GPU_DEVICE (int idx) noexcept
        {
            const int isp = idx / n_surf;
            p_dens_frac[idx] = p_species_frac[isp];
        });
    // --- Returning gas flux (initialize to zero) ---
    m_returning_gas_flux.resize(n_gas_sp * n_surf);
    amrex::Real* p_gas_flux = m_returning_gas_flux.dataPtr();
    amrex::ParallelFor(n_gas_sp * n_surf,
        [=] AMREX_GPU_DEVICE (int idx) noexcept
        {
            p_gas_flux[idx] = 0.0;
        });
    amrex::Gpu::streamSynchronize();
}


void
SurfacePhysicsBase::computeInflux ()
{
    for (int isp = 0; isp < num_influx_species; ++isp) {
        computeInflux(isp);
    }
}

void
SurfacePhysicsBase::computeInflux (int isp)
{
    // check if EB is enabled
    if (!EB::enabled() ) {
        amrex::Print() << " current mapping works only with EB surfaces \n";
        return;
    }
    auto& warpx = WarpX::GetInstance();
    const int lev = 0;
    amrex::Real const dt = warpx.getdt(lev);
    amrex::Real const cur_time = warpx.gett_new(lev);
    amrex::Real const influx_window = cur_time - m_influx_window_start_time;
    amrex::Print() << " influx window " << influx_window << "\n";
    //
    amrex::EBFArrayBoxFactory const& eb_box_factory = warpx.fieldEBFactory(lev);
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_box_factory.getMultiEBCellFlagFab();
    amrex::MultiCutFab const& eb_bnd_cent = eb_box_factory.getBndryCent();
    amrex::MultiCutFab const& eb_bnd_normal = eb_box_factory.getBndryNormal();
    amrex::MultiCutFab const& eb_bnd_area = eb_box_factory.getBndryArea();

    double* const AMREX_RESTRICT dptr_num_in_particles = num_in_particles[isp].dataPtr();
    double* const AMREX_RESTRICT dptr_bnd_influx = bnd_influx[isp].dataPtr();
    amrex::Real* sp_influx = m_incoming_flux.data();
    int num_surf_elements = surf_ijk.size();


    amrex::Print() << m_influx_window_started << " influx window " << influx_window << "\n";
    if (!m_influx_window_started || influx_window < 0.) {
        amrex::Real* p_sp_influx = sp_influx + isp * num_surf_elements;
        amrex::ParallelFor(num_surf_elements,
            [=] AMREX_GPU_DEVICE (int ibnd) noexcept {
                p_sp_influx[ibnd] = 0.;
            });
        amrex::Gpu::streamSynchronize();
    }

    for (amrex::MFIter mfi(eb_flag); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::FabType const fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue;}
        if (fab_type == amrex::FabType::covered) { continue;}

        auto const& eb_flag_arr = eb_flag.array(mfi);
        const amrex::Array4<const amrex::Real>& eb_bnd_area_arr = eb_bnd_area.array(mfi);
        auto const& ivect_arr = ivect_map->array(mfi);

        amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            if (eb_flag_arr(i,j,k).isRegular() ) { 
                return;
            } else if (eb_flag_arr(i,j,k).isCovered() ) {
                return;
            } else {
                int ivec = ivect_arr(i,j,k);
            //    dptr_bnd_influx[ivec] = dptr_num_in_particles[ivec]/eb_bnd_area_arr(i,j,k)/dt + 1e19;
	        if (influx_window > 0) {
                    sp_influx[isp*num_surf_elements + ivec] = dptr_num_in_particles[ivec]/eb_bnd_area_arr(i,j,k)/influx_window;
		}    else {
                    sp_influx[isp*num_surf_elements + ivec] = dptr_num_in_particles[ivec]/eb_bnd_area_arr(i,j,k)/dt; }
            }
        });
    }    
}
#endif
