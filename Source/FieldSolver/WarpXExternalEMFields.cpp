#include "WarpX.H"
#include "Utils/WarpXConst.H"
#ifdef PULSAR
    #include "Particles/PulsarParameters.H"
#endif
#ifdef WARPX_USE_PY
#   include "Python/WarpX_py.H"
#endif
#include <limits>
#include <AMReX_Array.H>

using namespace amrex;

#ifdef PULSAR
void
WarpX::ApplyPulsarEBFieldsOnGrid ()
{
    amrex::Print() << " in apply ext EB for pulsar \n";
    for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::Real cur_time = gett_new(lev);
        amrex::Print() << " lev : " << lev << "\n";
        amrex::MultiFab *Ex, *Ey, *Ez;
        amrex::MultiFab *Bx, *By, *Bz;
        Ex = Efield_fp[lev][0].get();
        Ey = Efield_fp[lev][1].get();
        Ez = Efield_fp[lev][2].get();
        Bx = Bfield_fp[lev][0].get();
        By = Bfield_fp[lev][1].get();
        Bz = Bfield_fp[lev][2].get();
        // GPU vector to store Ex-Bz staggering
        GpuArray<int,3> Ex_stag, Ey_stag, Ez_stag, Bx_stag, By_stag, Bz_stag;
        amrex::IntVect ex_type = Ex->ixType().toIntVect();
        amrex::IntVect ey_type = Ey->ixType().toIntVect();
        amrex::IntVect ez_type = Ez->ixType().toIntVect();
        amrex::IntVect bx_type = Bx->ixType().toIntVect();
        amrex::IntVect by_type = By->ixType().toIntVect();
        amrex::IntVect bz_type = Bz->ixType().toIntVect();
        for (int idim = 0; idim < AMREX_SPACEDIM-1; ++idim) {
            Ex_stag[idim] = ex_type[idim];
            Ey_stag[idim] = ey_type[idim];
            Ez_stag[idim] = ez_type[idim];
            Bx_stag[idim] = bx_type[idim];
            By_stag[idim] = by_type[idim];
            Bz_stag[idim] = bz_type[idim];
        }
        const auto problo = Geom(lev).ProbLoArray();
        const auto probhi = Geom(lev).ProbHiArray();
        const auto dx = Geom(lev).CellSizeArray();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const Box& tex  = mfi.tilebox( Ex->ixType().toIntVect() );
            const Box& tey  = mfi.tilebox( Ey->ixType().toIntVect() );
            const Box& tez  = mfi.tilebox( Ez->ixType().toIntVect() );

            auto const& Exfab = Ex->array(mfi);
            auto const& Eyfab = Ey->array(mfi);
            auto const& Ezfab = Ez->array(mfi);
            amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Er, Etheta, Ephi;
                PulsarParm::ComputeCellCoordinates( i, j, k, Ex_stag, problo,
                                                    dx, x, y, z);
                PulsarParm::ConvertCartesianToSphericalCoord( x, y, z, problo, probhi,
                                                     r, theta, phi );
                PulsarParm::ExternalEFieldSpherical( r, theta, phi, cur_time, 
                                                     Er, Etheta, Ephi );
                PulsarParm::ConvertSphericalToCartesianXComponent( Er, Etheta,
                                                     Ephi, r, theta, phi, Exfab(i,j,k) );
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Er, Etheta, Ephi;
                PulsarParm::ComputeCellCoordinates(i, j, k, Ey_stag, problo,
                                                   dx, x, y, z);
                PulsarParm::ConvertCartesianToSphericalCoord( x, y, z, problo, probhi,
                                                         r, theta, phi);
                PulsarParm::ExternalEFieldSpherical( r, theta, phi, cur_time,
                                                     Er, Etheta, Ephi);
                PulsarParm::ConvertSphericalToCartesianYComponent( Er, Etheta,
                                                     Ephi, r, theta, phi, Eyfab(i,j,k) );
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Er, Etheta, Ephi;
                PulsarParm::ComputeCellCoordinates(i, j, k, Ez_stag, problo,
                                                   dx, x, y, z);
                PulsarParm::ConvertCartesianToSphericalCoord( x, y, z, problo, probhi,
                                                         r, theta, phi);
                PulsarParm::ExternalEFieldSpherical( r, theta, phi, cur_time,
                                                     Er, Etheta, Ephi);
                PulsarParm::ConvertSphericalToCartesianZComponent( Er, Etheta,
                                                     Ephi, r, theta, phi, Ezfab(i,j,k) );
            });
        }
        for ( MFIter mfi(*Bx, TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const Box& tex  = mfi.tilebox( Bx->ixType().toIntVect() );
            const Box& tey  = mfi.tilebox( By->ixType().toIntVect() );
            const Box& tez  = mfi.tilebox( Bz->ixType().toIntVect() );

            auto const& Bxfab = Bx->array(mfi);
            auto const& Byfab = By->array(mfi);
            auto const& Bzfab = Bz->array(mfi);

            amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Br, Btheta, Bphi;
                PulsarParm::ComputeCellCoordinates(i, j, k, Bx_stag, problo,
                                                   dx, x, y, z);
                PulsarParm::ConvertCartesianToSphericalCoord( x, y, z, problo, probhi,
                                                              r, theta, phi);
                PulsarParm::ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                     Br, Btheta, Bphi);
                PulsarParm::ConvertSphericalToCartesianXComponent( Br, Btheta,
                                                     Bphi, r, theta, phi, Bxfab(i,j,k) );
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Br, Btheta, Bphi;
                PulsarParm::ComputeCellCoordinates(i, j, k, By_stag, problo,
                                                   dx, x, y, z);
                PulsarParm::ConvertCartesianToSphericalCoord( x, y, z, problo, probhi,
                                                              r, theta, phi);
                PulsarParm::ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                     Br, Btheta, Bphi);
                PulsarParm::ConvertSphericalToCartesianYComponent( Br, Btheta,
                                                     Bphi, r, theta, phi, Byfab(i,j,k) );
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real x, y, z;
                amrex::Real r, theta, phi;
                amrex::Real Br, Btheta, Bphi;
                PulsarParm::ComputeCellCoordinates( i, j, k, Bz_stag, problo,
                                                    dx, x, y, z);
                PulsarParm::ConvertCartesianToSphericalCoord( x, y, z, problo, probhi,
                                                              r, theta, phi);
                PulsarParm::ExternalBFieldSpherical( r, theta, phi, cur_time,
                                                     Br, Btheta, Bphi);
                PulsarParm::ConvertSphericalToCartesianZComponent( Br, Btheta,
                                                     Bphi, r, theta, phi, Bzfab(i,j,k) );
            });
        }
    }
}

#endif
