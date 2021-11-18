#ifndef WARPX_EDOTBFUNCTOR_H_
#define WARPX_EDOTBFUNCTOR_H_

#include "ComputeDiagFunctor.H"

class
EdotBFunctor final : public ComputeDiagFunctor
{
public:
    EdotBFunctor ( const amrex::MultiFab * const Ex_src,
                   const amrex::MultiFab * const Ey_src,
                   const amrex::MultiFab * const EZ_src,
                   const amrex::MultiFab * const Bx_src,
                   const amrex::MultiFab * const By_src,
                   const amrex::MultiFab * const Bz_src,
                   const int lev,
                   const amrex::IntVect crse_ratio,
                   int ncomp = 1);
    virtual void operator() (amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const override;
    void ComputeEdotB(amrex::MultiFab& mf_dst, int dcomp) const;
private:
    amrex::MultiFab const * const m_Ex_src = nullptr;
    amrex::MultiFab const * const m_Ey_src = nullptr;
    amrex::MultiFab const * const m_Ez_src = nullptr;
    amrex::MultiFab const * const m_Bx_src = nullptr;
    amrex::MultiFab const * const m_By_src = nullptr;
    amrex::MultiFab const * const m_Bz_src = nullptr;
    int m_lev;
};

#endif
