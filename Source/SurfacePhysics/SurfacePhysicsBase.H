#ifndef WARPX_SURFACE_PHYSICS_BASE_H_
#define WARPX_SURFACE_PHYSICS_BASE_H_

#ifdef WARPX_SURFACE_PHYSICS

#include "Particles/MultiParticleContainer_fwd.H"
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

class SurfacePhysicsBase
{
public:
    SurfacePhysicsBase ();

    ~SurfacePhysicsBase() = default;

    void ReadParameters ();
    void InitData ();
    void initializeMapping ();
    void AllocAndInitInfluxBndVectors();
    void AllocAndInitOutfluxBndVectors();
    void AllocAndInitSurfaceDensityFraction();

    void countParticlesFromEmbeddedBoundaries(
        MultiParticleContainer& mypc, ablastr::fields::MultiLevelScalarField const& distance_to_eb);

    void EvolveSurfacePhysics ();

    amrex::Real m_influx_window_start_time = 0.;
    bool m_influx_window_started = false;
    void nullifyInfluxParticleCounter ();
    void nullifyInfluxParticleCounter (int isp);
    void nullifyOutfluxParticleCounter ();
    void nullifyOutfluxParticleCounter (int isp);

    std::unique_ptr< amrex::iMultiFab > ivect_map;

    amrex::Vector<amrex::IntVect> surf_ijk;
    amrex::Vector<amrex::Real> surf_normal_x;
    amrex::Vector<amrex::Real> surf_normal_y;
    amrex::Vector<amrex::Real> surf_normal_z;
    int num_influx_species;
    int num_outflux_species;
    amrex::Vector< amrex::Gpu::DeviceVector< amrex::Real> > bnd_influx;
    amrex::Vector< amrex::Gpu::DeviceVector< amrex::Real> > bnd_outflux;
    amrex::Vector< amrex::Gpu::DeviceVector< amrex::Real > > num_in_particles;
    amrex::Vector< amrex::Gpu::DeviceVector< amrex::Real > > num_out_particles;

    int m_num_surface_species;
    int m_num_gas_species;
    amrex::Vector< amrex::Real > surface_species_fraction;
    amrex::Gpu::DeviceVector< amrex::Real > m_surface_density_fraction;
    amrex::Gpu::DeviceVector< amrex::Real > m_returning_gas_flux;
    amrex::Gpu::DeviceVector< amrex::Real > m_incoming_flux;

    // run-time gas species to chemistry species mapping
    int GetChemGasSpeciesIndex (int runtime_species_id) const;

    amrex::Gpu::DeviceVector<int> reaction_has_surface_products;
    amrex::Gpu::DeviceVector<int> reaction_has_gas_products;
    amrex::Gpu::DeviceVector<int> surface_sp_is_reactant;
    amrex::Gpu::DeviceVector<int> surface_sp_is_product;
    amrex::Gpu::DeviceVector<int> gas_sp_is_reactant;
    amrex::Gpu::DeviceVector<int> gas_sp_is_product;

    // constant values only for testing
    amrex::Real m_surface_site_density;
    amrex::Real m_plasma_influx;
    amrex::Real m_plasma_Ein; // in eV - constant for all species for all surface mesh elements
    amrex::Real m_chem_dt;
    amrex::Real m_start_time;
    amrex::Real m_end_time;
    amrex::Real m_cur_time;

    // Checkpoint and restart
    void WriteCheckpoint (const std::string& checkpoint_dir);
    void ReadCheckpoint (const std::string& checkpoint_dir);

// for testing
    void initializeInflux(int isp, amrex::Real flux_val);
    void computeInflux();
    void computeInflux(int isp);
private:
    struct Reaction {
        std::string equation;
        amrex::Vector<std::string> reactants;
        amrex::Vector<std::string> products;
        amrex::Vector<std::string> reactant_type;
        amrex::Vector<std::string> product_type;
        amrex::Vector<int> reactant_sp_val;
        amrex::Real P_energy0;
        amrex::Real P0;
        amrex::Real E_ref;
        amrex::Real E_th;
        amrex::Real exp;
    };
    amrex::Vector<std::string> tokenize_reaction (const std::string& input);
    bool is_gas_species (std::string species_symbol);
    bool is_surface_species (std::string species_symbol);

//    std::map<std::string, std::string> gas_species; //stores gas_species with the symbol
//    std::map<std::string, std::string> surface_species; //stores surface_species with the symbol
    std::vector<std::pair<std::string, std::string>> gas_species_vec;
    std::vector<std::pair<std::string, std::string>> surface_species_vec;
    std::vector<Reaction> reactions;

    // reaction data
    amrex::Gpu::DeviceVector<amrex::Real> m_rxn_P0;
    amrex::Gpu::DeviceVector<amrex::Real> m_rxn_E_ref;
    amrex::Gpu::DeviceVector<amrex::Real> m_rxn_E_th;
    amrex::Gpu::DeviceVector<amrex::Real> m_rxn_exp;
    amrex::Gpu::DeviceVector<int> m_rxn_num_reactants;
    int m_max_reactants_per_rxn = 0;
    amrex::Gpu::DeviceVector<int> m_reactant_is_gas; // [irxn*max_r + ir] =1 if gas, 0 if surface
    amrex::Gpu::DeviceVector<int> m_reactant_sp_val; // [irxn*max_r + ir] = species index

    // runtime species id to chemistry species id (-1 if there is no run time species id corresponding to chem)
    amrex::Vector<int> m_runtime_to_chemistry_sp_idx;
};


#endif // ifdef(WARPX_SURFACE_PHYSICS)

#endif
