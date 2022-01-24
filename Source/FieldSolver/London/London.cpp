#include "London.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include "Utils/WarpXUtil.H"
#include "WarpX.H"
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Scan.H>

London::London ()
{
    amrex::Print() << " London class is constructed\n";
}


void
London::EvolveLondonJ (amrex::Real dt)
{
    amrex::Print() << " evolve london J using E\n";
    
}

