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


    // Build runtime species to chemistry gas species map
    auto & warpx = WarpX::GetInstance();
    const auto & mpc = warpx.GetPartContainer();
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



    reaction_has_surface_products.resize(reactions.size(),0);
    surface_sp_is_reactant.resize(surface_species_vec.size()*reactions.size(),0);
    surface_sp_is_product.resize(surface_species_vec.size()*reactions.size(),0);
    for (int irxn = 0; irxn < reactions.size(); irxn++) {
        const Reaction& rxn = reactions[irxn];
        amrex::Print() << " Reaction " << irxn << "\n";
        for (int ip = 0; ip < rxn.product_type.size(); ++ip) {
            if (rxn.product_type[ip] == "surface") {
                reaction_has_surface_products[irxn] = 1;
                continue;
            }
        }
    }

    reaction_has_gas_products.resize(reactions.size(), 0);
    gas_sp_is_product.resize(gas_species_vec.size()*reactions.size(),0);
    for (int irxn = 0; irxn < reactions.size(); irxn++) {
        const Reaction& rxn = reactions[irxn];
        amrex::Print() << " Reaction " << irxn << "\n";
        for (int ip = 0; ip < rxn.product_type.size(); ++ip) {
            amrex::Print() << " prod type " << rxn.product_type[ip] << "\n";
            if (rxn.product_type[ip] == "gas") {
                reaction_has_gas_products[irxn] = 1;
                continue;
            }
        }
    }


    for (int irxn = 0; irxn < reactions.size(); irxn++) {
        Reaction& rxn = reactions[irxn];
        amrex::Print() << " Reaction " << irxn << "\n";
        for (int ir = 0; ir < rxn.reactant_type.size(); ++ir) {
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

    for (int isp = 0; isp < surface_species_vec.size() ; ++isp) {
        const std::string& symbol = surface_species_vec[isp].second;
        const std::string& name = surface_species_vec[isp].first;
        amrex::Print() << " symbol : " << symbol << "\n";
        amrex::Print() << " name " << name << "\n";
//        amrex::Print() << " symbol from srf " << surface_species[isp] << "\n";
        for (int irxn = 0; irxn < reactions.size(); irxn++) {
            const Reaction& rxn = reactions[irxn];
            amrex::Print() << " eq : " << rxn.equation << "\n";
            bool found = (std::find(rxn.reactants.begin(), rxn.reactants.end(), symbol) != rxn.reactants.end());
            amrex::Print() << " found ? " << found << "\n";
            surface_sp_is_reactant[isp*reactions.size() + irxn] = found ? 1 : 0;
            bool prod_found = (std::find(rxn.products.begin(), rxn.products.end(), symbol) != rxn.products.end());
            surface_sp_is_product[isp*reactions.size() + irxn] = prod_found ? 1 : 0;
        }
    }
    for (int isp = 0; isp < surface_species_vec.size() ; ++isp) {
        const std::string& symbol = surface_species_vec[isp].second;
        amrex::Print() << " symbol : " << symbol << "\n";
        for (int irxn = 0; irxn < reactions.size(); irxn++) {
            const Reaction& rxn = reactions[irxn];
            amrex::Print() << " eq : " << rxn.equation << "\n";
            amrex::Print() << " reaction has surface ? " << reaction_has_surface_products[irxn] << "\n";
            amrex::Print() << "is reactant : " << surface_sp_is_reactant[isp*reactions.size() + irxn] << "\n";
            amrex::Print() << "is product : " << surface_sp_is_product[isp*reactions.size() + irxn] << "\n";
        }
    }

    for (int isp = 0; isp < gas_species_vec.size(); ++isp) {
        const std::string& symbol = gas_species_vec[isp].second;
        const std::string& name = gas_species_vec[isp].first;
        for (int irxn = 0; irxn < reactions.size(); ++irxn) {
            const Reaction& rxn = reactions[irxn];
            bool prod_found = (std::find(rxn.products.begin(), rxn.products.end(), symbol) != rxn.products.end());
            gas_sp_is_product[isp * reactions.size() + irxn] = prod_found ? 1 : 0;
        }
    }

    for (int isp = 0; isp < gas_species_vec.size(); ++isp) {
        const std::string & symbol = gas_species_vec[isp].second;
        for (int irxn = 0; irxn < reactions.size(); ++irxn) {
            const Reaction& rxn = reactions[irxn];
//            amrex::Print() << " gas is product : " >> gas_sp_is_product[isp*reactions.size() + irxn] << "\n";
        }
    }

    // for testing purposes reading in constant values for site density, influx, and plasma_Ein
    pp_chem.get("surface_site_density", m_surface_site_density);
    pp_chem.get("plasma_influx", m_plasma_influx);
    pp_chem.get("plasma_Ein", m_plasma_Ein);
    pp_chem.get("dt", m_chem_dt);
    pp_chem.get("start_time",m_start_time);
    pp_chem.get("end_time",m_end_time);
    m_cur_time = 0.;
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
    AllocAndInitInfluxBndVectors();
    AllocAndInitOutfluxBndVectors();
    AllocAndInitSurfaceDensityFraction();
}

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
    //
    amrex::EBFArrayBoxFactory const& eb_box_factory = warpx.fieldEBFactory(lev);
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_box_factory.getMultiEBCellFlagFab();
    amrex::MultiCutFab const& eb_bnd_cent = eb_box_factory.getBndryCent();
    amrex::MultiCutFab const& eb_bnd_normal = eb_box_factory.getBndryNormal();

    ivect_map = std::make_unique< amrex::iMultiFab> (warpx.boxArray(lev), warpx.DistributionMap(lev), 1, 1);
    ivect_map->setVal(0);

    for (amrex::MFIter mfi(eb_flag); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::FabType const fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue;}
        else if (fab_type == amrex::FabType::covered) { continue;}

        // all cells in fab are open, i.e., outside EB
        if (fab_type == amrex::FabType::regular) {continue;}
        // all cells in fab are enclosed within EB
        if (fab_type == amrex::FabType::covered) {continue;}

        auto const& eb_flag_arr = eb_flag.array(mfi);
        const amrex::Array4<const amrex::Real> & eb_bnd_normal_arr = eb_bnd_normal.array(mfi);
        auto const ivect_arr = ivect_map->array(mfi);

        amrex::LoopOnCpu( box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            amrex::IntVect const iv(AMREX_D_DECL(i,j,k));            
            if (eb_flag_arr(i,j,k).isRegular() ) {
                return;
            }
            else if (eb_flag_arr(i,j,k).isCovered() ) {
                return;
            }
            else {
                surf_ijk.push_back(iv);
                ivect_arr(i,j,k) = surf_ijk.size() - 1;

                surf_normal_x.push_back(eb_bnd_normal_arr(i,j,k,0));
#if (defined WARPX_DIM_XZ)
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,1));
#elif (defined WARPX_DIM_3D)
                surf_normal_y.push_back(eb_bnd_normal_arr(i,j,k,1));
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,2));
#endif
            }
        });
    }
    
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
    for (int ibnd = 0; ibnd < surf_ijk.size(); ++ibnd)
    {
        num_in_particles[isp][ibnd] = 0;
        bnd_influx[isp][ibnd] = 0.;
    }
}

void
SurfacePhysicsBase::initializeInflux(int isp, amrex::Real flux_val)
{
    for (int ibnd = 0; ibnd < surf_ijk.size(); ++ibnd)
    {
        bnd_influx[isp][ibnd] = flux_val;
        m_incoming_flux[isp * surf_ijk.size() + ibnd] = 0.;
    }
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
    for (int ibnd = 0; ibnd < surf_ijk.size(); ++ibnd)
    {
        num_out_particles[isp][ibnd] = 0;
        bnd_outflux[isp][ibnd] = 5.e5;
    }
}

void
SurfacePhysicsBase::AllocAndInitSurfaceDensityFraction ()
{
    m_surface_density_fraction.resize(m_num_surface_species * surf_ijk.size());
    for (int isp = 0; isp < m_num_surface_species; ++isp) {
        for (int i = 0; i < surf_ijk.size(); ++i) {
            m_surface_density_fraction[isp*surf_ijk.size() + i] = surface_species_fraction[isp];
        }
    }

    m_returning_gas_flux.resize(m_num_gas_species * surf_ijk.size());
    for (int isp = 0; isp < m_num_gas_species; ++isp) {
       for (int i = 0; i < surf_ijk.size() ; ++i) {
           m_returning_gas_flux[isp * surf_ijk.size() + i] = 0.;
       }
    }
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

    if (!m_influx_window_started || influx_window <= 0.) {
        for (int ibnd = 0; ibnd < num_surf_elements; ++ibnd) {
            sp_influx[isp*num_surf_elements + ibnd] = 0.;
        }
        return;
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
                sp_influx[isp*num_surf_elements + ivec] = dptr_num_in_particles[ivec]/eb_bnd_area_arr(i,j,k)/influx_window; // + 1e19;
            }
        });
    }    
}
#endif
