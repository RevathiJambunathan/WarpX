#ifndef WARPX_EDOTBFUNCTOR_H_
#define WARPX_EDOTBFUNCTOR_H_

#include "ComputeDiagFunctor.H"

class
jdotEFunctor final : public ComputeDiagFunctor
{
public:
    jdotEFunctor ( const amrex::MultiFab * const Ex_src,
                   const amrex::MultiFab * const Ey_src,
                   const amrex::MultiFab * const Ez_src,
                   const amrex::MultiFab * const jx_src,
                   const amrex::MultiFab * const jy_src,
                   const amrex::MultiFab * const jz_src,
                   const int lev,
                   const amrex::IntVect crse_ratio,
                   int ncomp = 1);
    virtual void operator() (amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const override;
    void ComputejdotE(amrex::MultiFab& mf_dst, int dcomp) const;
private:
    amrex::MultiFab const * const m_Ex_src = nullptr;
    amrex::MultiFab const * const m_Ey_src = nullptr;
    amrex::MultiFab const * const m_Ez_src = nullptr;
    amrex::MultiFab const * const m_jx_src = nullptr;
    amrex::MultiFab const * const m_jy_src = nullptr;
    amrex::MultiFab const * const m_jz_src = nullptr;
    int m_lev;
};

#endif
