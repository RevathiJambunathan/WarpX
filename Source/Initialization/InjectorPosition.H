/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef INJECTOR_POSITION_H_
#define INJECTOR_POSITION_H_

#include "InjectorPosition_fwd.H"

#include <AMReX_Dim3.H>
#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>

#ifdef PULSAR
    #include "Particles/PulsarParameters.H"
#endif

// struct whose getPositionUnitBox returns x, y and z for a particle with
// random distribution inside a unit cell.
struct InjectorPositionRandom
{
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int /*i_part*/, int /*ref_fac*/,
                        amrex::RandomEngine const& engine
#ifdef PULSAR
                        , int ModifyParticleWtAtInjection = 0,
                        amrex::Real Ninj_fraction = 1.
#endif
                        ) const noexcept
    {
        return amrex::XDim3{amrex::Random(engine), amrex::Random(engine), amrex::Random(engine)};
    }
};

// struct whose getPositionUnitBox returns x, y and z for a particle with
// random distribution on a plane inside a unit cell.
struct InjectorPositionRandomPlane
{
    InjectorPositionRandomPlane (int const& a_dir) noexcept : dir(a_dir) {}

    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int /*i_part*/, int /*ref_fac*/,
                        amrex::RandomEngine const& engine
#ifdef PULSAR
                        , int ModifyParticleWtAtInjection = 0,
                        amrex::Real Ninj_fraction = 1.
#endif
                        ) const noexcept
    {
        using namespace amrex::literals;
        if (dir == 0) return amrex::XDim3{0._rt, amrex::Random(engine), amrex::Random(engine)};
#if (defined WARPX_DIM_RZ)
        if (dir == 1) return amrex::XDim3{amrex::Random(engine), 0._rt, 0._rt};
#else
        if (dir == 1) return amrex::XDim3{amrex::Random(engine), 0._rt, amrex::Random(engine)};
#endif
        else          return amrex::XDim3{amrex::Random(engine), amrex::Random(engine), 0._rt};
    }
private:
    int dir;
};

// struct whose getPositionUnitBox returns x, y and z for a particle with
// regular distribution inside a unit cell.
struct InjectorPositionRegular
{
    InjectorPositionRegular (amrex::Dim3 const& a_ppc) noexcept : ppc(a_ppc) {}

    // i_part: particle number within the cell, required to evenly space
    // particles within the cell.
    // ref_fac: the number of particles evenly-spaced within a cell
    // is a_ppc*(ref_fac**AMREX_SPACEDIM).
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int const i_part, int const ref_fac,
                        amrex::RandomEngine const&
#ifdef PULSAR
                        , int ModifyParticleWtAtInjection = 0,
                        amrex::Real Ninj_fraction = 1.
#endif
                        ) const noexcept
    {
        using namespace amrex;
#ifdef PULSAR
#if (defined WARPX_DIM_3D) || (defined WARPX_DIM_RZ)
        int nx, ny, nz;
        if (ModifyParticleWtAtInjection == 1) {
            nx = ref_fac*ppc.x;
            ny = ref_fac*ppc.y;
            nz = ref_fac*ppc.z;
        } else if (ModifyParticleWtAtInjection == 0) {
            amrex::Real ninj_per_dim = std::cbrt(Ninj_fraction);
            nx = ref_fac*ppc.x*ninj_per_dim;
            ny = ref_fac*ppc.y*ninj_per_dim;
            nz = ref_fac*ppc.z*ninj_per_dim;
        }
#else
        amrex::Abort("pulsar sim does not support 2d yet");
#endif // endif for 3D
#else // else ifndef PULSAR
        int const nx = ref_fac*ppc.x;
        int const ny = ref_fac*ppc.y;
#if (defined WARPX_DIM_3D) || (defined WARPX_DIM_RZ)
        int const nz = ref_fac*ppc.z;
#else
        int const nz = 1;
#endif // 3D ifdef
#endif // PULSAR ifdef
        int const ix_part = i_part / (ny*nz);  // written this way backward compatibility
        int const iz_part = (i_part-ix_part*(ny*nz)) / ny;
        int const iy_part = (i_part-ix_part*(ny*nz)) - ny*iz_part;
        return XDim3{
            (0.5_rt + ix_part) / nx,
            (0.5_rt + iy_part) / ny,
            (0.5_rt + iz_part) / nz
        };
    }

#ifdef PULSAR
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getppcInEachDim () const noexcept
    {
        using namespace amrex;
        return XDim3{ppc.x, ppc.y, ppc.z};
    }
#endif

private:
    amrex::Dim3 ppc;
};

// Base struct for position injector.
// InjectorPosition contains a union (called Object) that holds any one
// instance of:
// - InjectorPositionRandom : to generate random distribution;
// - InjectorPositionRegular: to generate regular distribution.
// The choice is made at runtime, depending in the constructor called.
// This mimics virtual functions.
struct InjectorPosition
{
    // This constructor stores a InjectorPositionRandom in union object.
    InjectorPosition (InjectorPositionRandom* t,
                      amrex::Real a_xmin, amrex::Real a_xmax,
                      amrex::Real a_ymin, amrex::Real a_ymax,
                      amrex::Real a_zmin, amrex::Real a_zmax)
        : type(Type::random),
          object(t),
          xmin(a_xmin), xmax(a_xmax),
          ymin(a_ymin), ymax(a_ymax),
          zmin(a_zmin), zmax(a_zmax)
    { }

    // This constructor stores a InjectorPositionRandomPlane in union object.
    InjectorPosition (InjectorPositionRandomPlane* t,
                      amrex::Real a_xmin, amrex::Real a_xmax,
                      amrex::Real a_ymin, amrex::Real a_ymax,
                      amrex::Real a_zmin, amrex::Real a_zmax,
                      int const& a_dir)
        : type(Type::randomplane),
          object(t, a_dir),
          xmin(a_xmin), xmax(a_xmax),
          ymin(a_ymin), ymax(a_ymax),
          zmin(a_zmin), zmax(a_zmax)
    { }

    // This constructor stores a InjectorPositionRegular in union object.
    InjectorPosition (InjectorPositionRegular* t,
                      amrex::Real a_xmin, amrex::Real a_xmax,
                      amrex::Real a_ymin, amrex::Real a_ymax,
                      amrex::Real a_zmin, amrex::Real a_zmax,
                      amrex::Dim3 const& a_ppc)
        : type(Type::regular),
          object(t, a_ppc),
          xmin(a_xmin), xmax(a_xmax),
          ymin(a_ymin), ymax(a_ymax),
          zmin(a_zmin), zmax(a_zmax)
    { }

    // Explicitly prevent the compiler from generating copy constructors
    // and copy assignment operators.
    InjectorPosition (InjectorPosition const&) = delete;
    InjectorPosition (InjectorPosition&&) = delete;
    void operator= (InjectorPosition const&) = delete;
    void operator= (InjectorPosition &&) = delete;

    // call getPositionUnitBox from the object stored in the union
    // (the union is called Object, and the instance is called object).
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int const i_part, int const ref_fac,
                        amrex::RandomEngine const& engine
#ifdef PULSAR
                        , int ModifyParticleWtAtInjection = 0,
                        amrex::Real Ninj_fraction = 1.
#endif
                        ) const noexcept
    {
        switch (type)
        {
        case Type::regular:
        {
            return object.regular.getPositionUnitBox(i_part, ref_fac, engine
#ifdef PULSAR
                                                     , ModifyParticleWtAtInjection,
                                                     Ninj_fraction
#endif
                                                    );
        }
        case Type::randomplane:
        {
            return object.randomplane.getPositionUnitBox(i_part, ref_fac, engine
#ifdef PULSAR
                                                     , ModifyParticleWtAtInjection,
                                                     Ninj_fraction
#endif
                                                    );
        }
        default:
        {
            return object.random.getPositionUnitBox(i_part, ref_fac, engine
#ifdef PULSAR
                                                     , ModifyParticleWtAtInjection,
                                                     Ninj_fraction
#endif
                                                    );
        }
        };
    }

#ifdef PULSAR
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getppcInEachDim () const noexcept
    {
        switch (type)
        {
        case Type::regular:
        {
            return object.regular.getppcInEachDim();
        }
        default:
        {
            return object.regular.getppcInEachDim();
        }
        };
    }
#endif

    // bool: whether position specified is within bounds.
    AMREX_GPU_HOST_DEVICE
    bool
    insideBounds (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
        return (x < xmax and x >= xmin and
                y < ymax and y >= ymin and
                z < zmax and z >= zmin);
    }

#ifdef PULSAR
    // bool: whether cell-center position specified is within pulsar injection region
    // So that these cells can inject particles
    // a buffer factor of 2*dR (2*dx) is used to ensure cells partially cut by sphere
    // inject particles
    AMREX_GPU_HOST_DEVICE
    bool
    insidePulsarBoundsCC (amrex::Real r, amrex::Real r_min, amrex::Real r_max, amrex::Real dR_factor) const noexcept
    {
        return (r<=(r_max + dR_factor) and r>=(r_min-dR_factor));
    }

    // bool: whether particle position specified is within pulsar injection region
    // So that these cells can inject particles
    AMREX_GPU_HOST_DEVICE
    bool
    insidePulsarBounds (amrex::Real r, amrex::Real r_min, amrex::Real r_max) const noexcept
    {
        return (r<=(r_max) and r>=(r_min));
    }
#endif

    // bool: whether the region defined by lo and hi overaps with the plasma region
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool
    overlapsWith (const amrex::XDim3& lo, const amrex::XDim3& hi) const noexcept
    {
        return ! (   (xmin > hi.x) || (xmax < lo.x)
                  || (ymin > hi.y) || (ymax < lo.y)
                  || (zmin > hi.z) || (zmax < lo.z) );
    }

private:
    enum struct Type { random, randomplane, regular };
    Type type;

    // An instance of union Object constructs and stores any one of
    // the objects declared (random or regular).
    union Object {
        Object (InjectorPositionRandom*) noexcept : random() {}
        Object (InjectorPositionRandomPlane*, int const& a_dir) noexcept
            : randomplane(a_dir) {}
        Object (InjectorPositionRegular*, amrex::Dim3 const& a_ppc) noexcept
            : regular(a_ppc) {}
        InjectorPositionRandom random;
        InjectorPositionRandomPlane randomplane;
        InjectorPositionRegular regular;
    };
    Object object;

    amrex::Real xmin, xmax;
    amrex::Real ymin, ymax;
    amrex::Real zmin, zmax;
};

#endif
