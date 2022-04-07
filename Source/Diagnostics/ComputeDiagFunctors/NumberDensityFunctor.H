#ifndef WARPX_NUMBERDENSITYFUNCTOR_H_
#define WARPX_NUMBERDENSITYFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to cell-center MF and store result in mf_out.
 */
class
NumberDensityFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     *
     * \param[in] mf_src source multifab.
     * \param[in] lev level of multifab. Used for averaging in rz.
     * \param[in] crse_ratio for interpolating field values from the simulation MultiFab, src_mf,
                  to the output diagnostic MultiFab, mf_dst.
     * \param[in] species_index index of species for which number density is computed
     * \param[in] scomp starting component of mf_src 
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    NumberDensityFunctor(const amrex::MultiFab * const mf_src, const int lev,
                      const amrex::IntVect crse_ratio,
                      int species_index, int ncomp=1);
    /** \brief Cell-center m_mf_src and write the result in mf_dst.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    virtual void operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const override;
private:
    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab const * const m_mf_src = nullptr;
    int m_lev; /**< level on which mf_src is defined (used in cylindrical) */
    int m_scomp;
};

#endif // WARPX_NUMBERDENSITYFUNCTOR_H_
