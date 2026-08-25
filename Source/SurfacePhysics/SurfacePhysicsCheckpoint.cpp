/* Copyright 2024 Revathi Jambunathan
 *
 * This file is part of WarpX
 *
 * License: BSD-3-Clause-LBNL
 */

#ifdef WARPX_SURFACE_PHYSICS

#include "SurfacePhysicsBase.H"
#include "Utils/TextMsg.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <fstream>
#include <sstream>
#include <string>


namespace {
    std::string surface_chk_dir (const std::string& chk)
    {
        return chk + "/SurfacePhysics";
    }

    std::string rank_file (const std::string& chk,
                           const std::string& name,
                           int rank)
    {
        return surface_chk_dir(chk) + "/" + name + "_r" + std::to_string(rank);
    }
}

void
SurfacePhysicsBase::WriteCheckpoint (const std::string& checkpoint_dir)
{
    const int myrank = amrex::ParallelDescriptor::MyProc();
    const int n_surf = static_cast<int>(surf_ijk.size());

    // Create subdirectory — IOProcessor creates it, Barrier ensures it exists
    // before other ranks try to write into it
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::UtilCreateDirectory(surface_chk_dir(checkpoint_dir), 0755);
    }
    amrex::ParallelDescriptor::Barrier();

    // -----------------------------------------------------------------------
    // 1.  Header — scalars only, IOProcessor writes
    // -----------------------------------------------------------------------
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
        std::ofstream hdr;
        hdr.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        const std::string hdr_path = surface_chk_dir(checkpoint_dir) + "/Header";
        hdr.open(hdr_path.c_str(),
                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        if (!hdr.good()) { amrex::FileOpenFailed(hdr_path); }
        hdr.precision(17);

        hdr << m_cur_time                 << "\n";
        hdr << m_influx_window_start_time << "\n";
        hdr << static_cast<int>(m_influx_window_started) << "\n";
        hdr << static_cast<int>(m_surface_evolution_header_written)      << "\n";
        hdr << static_cast<int>(m_surface_flux_evolution_header_written) << "\n";
        hdr << num_influx_species         << "\n";
        hdr << num_outflux_species        << "\n";
        hdr << m_num_surface_species      << "\n";
        hdr << m_num_gas_species          << "\n";
        hdr << n_surf                     << "\n";
    }

    auto write_array = [&](const amrex::Real* d_ptr, int count,
                           const std::string& name)
    {
        amrex::Vector<amrex::Real> h(count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_ptr, d_ptr + count, h.data());
        amrex::Gpu::streamSynchronize();

        const std::string fname = rank_file(checkpoint_dir, name, myrank);
        std::ofstream ofs(fname, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) { amrex::FileOpenFailed(fname); }
        ofs.write(reinterpret_cast<const char*>(h.data()),
                  static_cast<std::streamsize>(count * sizeof(amrex::Real)));
    };

    write_array(m_surface_density_fraction.dataPtr(),
                m_num_surface_species * n_surf,
                "surface_density_fraction");

    write_array(m_incoming_flux.dataPtr(),
                num_influx_species * n_surf,
                "incoming_flux");

    write_array(m_returning_gas_flux.dataPtr(),
                m_num_gas_species * n_surf,
                "returning_gas_flux");

    for (int isp = 0; isp < num_influx_species; ++isp) {
        write_array(num_in_particles[isp].dataPtr(),
                    n_surf,
                    "num_in_particles_sp" + std::to_string(isp));
    }

    for (int isp = 0; isp < num_outflux_species; ++isp) {
        write_array(num_out_particles[isp].dataPtr(),
                    n_surf,
                    "num_out_particles_sp" + std::to_string(isp));
    }

    amrex::ParallelDescriptor::Barrier();
    amrex::Print() << " SurfacePhysics checkpoint written to "
                   << surface_chk_dir(checkpoint_dir) << "\n";
}

void
SurfacePhysicsBase::ReadCheckpoint (const std::string& checkpoint_dir)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !surf_ijk.empty(),
        "SurfacePhysics restart: ReadCheckpoint called before InitData().");

    const int myrank = amrex::ParallelDescriptor::MyProc();
    const int nranks = amrex::ParallelDescriptor::NProcs();
    const int n_surf = static_cast<int>(surf_ijk.size()); // rebuilt by initializeMapping()

    amrex::Vector<char> fileCharPtr;
    amrex::ParallelDescriptor::ReadAndBcastFile(
        surface_chk_dir(checkpoint_dir) + "/Header", fileCharPtr);
    std::istringstream is(std::string(fileCharPtr.dataPtr()),
                          std::istringstream::in);
    is.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    amrex::Real chk_cur_time, chk_influx_window_start;
    int chk_influx_started;
    int chk_surf_evol_hdr_written, chk_surf_flux_hdr_written;
    int chk_num_influx, chk_num_outflux, chk_num_surf_sp, chk_num_gas_sp;
    int chk_n_surf;

    is >> chk_cur_time;
    is >> chk_influx_window_start;
    is >> chk_influx_started;
    is >> chk_surf_evol_hdr_written;
    is >> chk_surf_flux_hdr_written;
    is >> chk_num_influx;
    is >> chk_num_outflux;
    is >> chk_num_surf_sp;
    is >> chk_num_gas_sp;
    is >> chk_n_surf;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        chk_num_surf_sp == m_num_surface_species &&
        chk_num_gas_sp  == m_num_gas_species     &&
        chk_num_influx  == num_influx_species    &&
        chk_num_outflux == num_outflux_species   &&
        chk_n_surf      == n_surf,
        "SurfacePhysics restart: species counts in checkpoint do not match input.");

    m_cur_time                 = chk_cur_time;
    m_influx_window_start_time = chk_influx_window_start;
    m_influx_window_started    = static_cast<bool>(chk_influx_started);
    m_surface_evolution_header_written      = static_cast<bool>(chk_surf_evol_hdr_written);
    m_surface_flux_evolution_header_written = static_cast<bool>(chk_surf_flux_hdr_written);

    // -----------------------------------------------------------------------
    // 2.  Helper: read binary file, check size, copy to device
    //     File size mismatch means mesh or MPI decomposition changed
    // -----------------------------------------------------------------------
    auto read_array = [&](amrex::Real* d_ptr, int count, const std::string& name)
    {
        const std::string fname = rank_file(checkpoint_dir, name, myrank);
        std::ifstream ifs(fname, std::ios::binary | std::ios::ate);
        if (!ifs.good()) { amrex::FileOpenFailed(fname); }

        const std::streamsize expected =
            static_cast<std::streamsize>(count * sizeof(amrex::Real));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ifs.tellg() == expected,
            "SurfacePhysics restart: file " + fname +
            " has unexpected size — mesh or MPI decomposition changed.");

        ifs.seekg(0);
        amrex::Vector<amrex::Real> h(count);
        ifs.read(reinterpret_cast<char*>(h.data()), expected);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ifs.good(),
            "SurfacePhysics restart: failed reading " + fname);

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h.data(), h.data() + count, d_ptr);
        amrex::Gpu::streamSynchronize();
    };

    // -----------------------------------------------------------------------
    // 3.  Arrays — each rank reads its own local slice independently
    // -----------------------------------------------------------------------
    read_array(m_surface_density_fraction.dataPtr(),
               m_num_surface_species * n_surf,
               "surface_density_fraction");

    read_array(m_incoming_flux.dataPtr(),
               num_influx_species * n_surf,
               "incoming_flux");

    read_array(m_returning_gas_flux.dataPtr(),
               m_num_gas_species * n_surf,
               "returning_gas_flux");

    for (int isp = 0; isp < num_influx_species; ++isp) {
        read_array(num_in_particles[isp].dataPtr(),
                   n_surf,
                   "num_in_particles_sp" + std::to_string(isp));
    }

    for (int isp = 0; isp < num_outflux_species; ++isp ) {
        read_array(num_out_particles[isp].dataPtr(),
                   n_surf,
                   "num_out_particles_sp" + std::to_string(isp));
    }

    amrex::Print() << " SurfacePhysics checkpoint read from "
                   << surface_chk_dir(checkpoint_dir) << "\n";
}

#endif
