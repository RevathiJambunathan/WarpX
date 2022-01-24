#ifndef LONDON_H
#define LONDON_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>

using namespace amrex;

class London {

public:
    //Constructor
    London ();    

    void EvolveLondonJ (amrex::Real dt);
};


#endif
