#ifndef WARPX_SPHERICALCOMPONENTFUNCTOR_H_
#define WARPX_SPHERICALCOMPONENTFUNCTOR_H_

#include "ComputeDiagFunctor.H"

class
SphericalComponentFunctor final : public ComputeDiagFunctor
{
public:
    SphericalComponentFunctor ( const amrex::MultiFab * const mfx_src,
                                const amrex::MultiFab * const mfy_src, 
                                const amrex::MultiFab * const mfz_src,
                                const int lev,
                                const amrex::IntVect crse_ratio,
                                const int sphericalcomp,
                                const bool Efield,
                                int ncomp=1);

    virtual void operator ()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const override;
private:
    amrex::MultiFab const * const m_mfx_src = nullptr;
    amrex::MultiFab const * const m_mfy_src = nullptr;
    amrex::MultiFab const * const m_mfz_src = nullptr;
    int m_lev;
    int m_sphericalcomp;
    bool m_Efield;    
};

#endif
