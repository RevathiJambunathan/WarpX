#include "MacroscopicProperties.H"
#include <AMReX_ParmParse.H>
#include "WarpX.H"

using namespace amrex;

MacroscopicProperties::MacroscopicProperties ()
{
    ReadParameters();
}

void
MacroscopicProperties::ReadParameters ()
{
    ParmParse pp("macroscopic");
    // Since macroscopic maxwell solve is turned on, user must define sigma, mu, and epsilon //
    pp.get("sigma_init_style", m_sigma_s);
    // constant initialization
    if (m_sigma_s == "constant") pp.get("sigma", m_sigma);
    // initialization of sigma (conductivity) with parser
    if (m_sigma_s == "parse_sigma_function") {
        Store_parserString(pp, "sigma_function(x,y,z)", m_str_sigma_function);
        m_sigma_parser.reset(new ParserWrapper<3>(
                                 makeParser(m_str_sigma_function,{"x","y","z"}) ) );
    }

    pp.get("epsilon_init_style", m_epsilon_s);
    if (m_epsilon_s == "constant") pp.get("epsilon", m_epsilon);
    // initialization of epsilon (permittivity) with parser
    if (m_epsilon_s == "parse_epsilon_function") {
        Store_parserString(pp, "epsilon_function(x,y,z)", m_str_epsilon_function);
        m_epsilon_parser.reset(new ParserWrapper<3>(
                                 makeParser(m_str_epsilon_function,{"x","y","z"}) ) );
    }

    pp.get("mu_init_style", m_mu_s);
    if (m_mu_s == "constant") pp.get("mu", m_mu);
    // initialization of mu (permeability) with parser
    if (m_mu_s == "parse_mu_function") {
        Store_parserString(pp, "mu_function(x,y,z)", m_str_mu_function);
        m_mu_parser.reset(new ParserWrapper<3>(
                                 makeParser(m_str_mu_function,{"x","y","z"}) ) );
    }

}

void
MacroscopicProperties::InitData ()
{
    amrex::Print() << "we are in init data of macro \n";
    auto & warpx = WarpX::GetInstance();

    // Get BoxArray and DistributionMap of warpx instant.
    int lev = 0;
    BoxArray ba = warpx.boxArray(lev);
    DistributionMapping dmap = warpx.DistributionMap(lev);
    int ng = 3;
    // Define material property multifabs using ba and dmap from WarpX instance
    // sigma is cell-centered MultiFab
    m_sigma_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng); 
    // epsilon is cell-centered MultiFab
    m_eps_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    // mu is cell-centered MultiFab
    m_mu_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    // Initialize sigma
    if (m_sigma_s == "constant") {

        m_sigma_mf->setVal(m_sigma);

    } else if (m_sigma_s == "parse_sigma_function") {

        InitializeMacroMultiFabUsingParser(m_sigma_mf.get(), m_sigma_parser.get(), lev);
    }
    // Initialize epsilon
    if (m_epsilon_s == "constant") {

        m_eps_mf->setVal(m_epsilon);

    } else if (m_epsilon_s == "parse_epsilon_function") {

        InitializeMacroMultiFabUsingParser(m_eps_mf.get(), m_epsilon_parser.get(), lev);

    }
    // Initialize mu
    if (m_mu_s == "constant") {

        m_mu_mf->setVal(m_mu);

    } else if (m_mu_s == "parse_mu_function") {

        InitializeMacroMultiFabUsingParser(m_mu_mf.get(), m_mu_parser.get(), lev);

    }


    sigma_IndexType.resize( 3 );
    epsilon_IndexType.resize( 3 );
    mu_IndexType.resize( 3 );
    Ex_IndexType.resize( 3 );
    Ey_IndexType.resize( 3 );
    Ez_IndexType.resize( 3 );
    macro_cr_ratio.resize( 3 );

    IntVect sigma_stag = m_sigma_mf->ixType().toIntVect();
    IntVect epsilon_stag = m_eps_mf->ixType().toIntVect();
    IntVect mu_stag = m_mu_mf->ixType().toIntVect();
    IntVect Ex_stag = warpx.getEfield_fp(0,0).ixType().toIntVect();
    IntVect Ey_stag = warpx.getEfield_fp(0,1).ixType().toIntVect();
    IntVect Ez_stag = warpx.getEfield_fp(0,2).ixType().toIntVect();
    
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        sigma_IndexType[idim]   = sigma_stag[idim];
        epsilon_IndexType[idim] = epsilon_stag[idim];
        mu_IndexType[idim]      = mu_stag[idim];
        Ex_IndexType[idim]      = Ex_stag[idim];
        Ey_IndexType[idim]      = Ey_stag[idim];
        Ez_IndexType[idim]      = Ez_stag[idim];
        macro_cr_ratio[idim]    = 1;
    }
    

}

void
MacroscopicProperties::InitializeMacroMultiFabUsingParser (
                       MultiFab *macro_mf, ParserWrapper<3> *macro_parser,
                       int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const RealBox& real_box = warpx.Geom(lev).ProbDomain();
    IntVect iv = macro_mf->ixType().toIntVect();
    IntVect grown_iv = iv + amrex::IntVect(macro_mf->nGrow());
    for ( MFIter mfi(*macro_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Initialize ghost cells in addition to valid cells
      
        const Box& tb = mfi.growntilebox(grown_iv);
        auto const& macro_fab =  macro_mf->array(mfi);
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type
                Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;

                Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;

                Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;

                // initialize the macroparameter
                macro_fab(i,j,k) = (*macro_parser)(x,y,z);
        });

    }


}

