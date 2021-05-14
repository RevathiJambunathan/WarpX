#ifndef WARPX_POYNTINGVECTORFUNCTOR_H_
#define WARPX_POYNTINGVECTORFUNCTOR_H_

#include "ComputeDiagFunctor.H"

class
PoyntingVectorFunctor final : public ComputeDiagFunctor
{
public:
    PoyntingVectorFunctor ( const amrex::MultiFab * const Ex_src,
                            const amrex::MultiFab * const Ey_src,
                            const amrex::MultiFab * const EZ_src,
                            const amrex::MultiFab * const Bx_src,
                            const amrex::MultiFab * const By_src,
                            const amrex::MultiFab * const Bz_src,
                            const int lev,
                            const amrex::IntVect crse_ratio,
                            int vectorcomp,
                            int ncomp = 1);
    virtual void operator() (amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const override;    
private:
    amrex::MultiFab const * const m_Ex_src = nullptr;
    amrex::MultiFab const * const m_Ey_src = nullptr;
    amrex::MultiFab const * const m_Ez_src = nullptr;
    amrex::MultiFab const * const m_Bx_src = nullptr;
    amrex::MultiFab const * const m_By_src = nullptr;
    amrex::MultiFab const * const m_Bz_src = nullptr;
    int m_lev;
    int m_vectorcomp;
};

#endif
