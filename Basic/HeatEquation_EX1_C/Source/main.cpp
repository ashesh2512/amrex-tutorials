#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "myfunc.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions
    int n_cell, max_grid_size, nsteps, plot_int;
    Real beta;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Scaling parameters associated with mesh transformation
        beta = 1.001;
        pp.query("beta",beta);
    }

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

       // This defines the physical box, [-1,1] in each direction.
        RealBox real_box({AMREX_D_DECL( Real(0.0), Real(0.0), Real(0.0))},
                         {AMREX_D_DECL( Real(128), Real(128), Real(128))});

        // periodic in all direction
        Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

        // This defines a Geometry object
        geom.define(domain,real_box,CoordSys::cartesian,is_periodic);
    }

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp_scalar  = 1;
    int Ncomp_vector  = 3;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab phi_old(ba, dm, Ncomp_scalar, Nghost);
    MultiFab phi_new(ba, dm, Ncomp_scalar, Nghost);

    // coordinate transformation parameter
    MultiFab d_eta(ba, dm, Ncomp_vector, Nghost);

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    init_vars(d_eta, phi_new, geom, beta);
    // ========================================

    Real cfl = 0.9;
    Real coeff = AMREX_D_TERM(   1./(dx[0]*dx[0]),
                               + 1./(dx[1]*dx[1]),
                               + 1./(dx[2]*dx[2]) );
    Real dt = cfl/(2.0*coeff);

    // time = starting time in the simulation
    Real time = 0.0;

    // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        // assemble variables for plotting
        Vector<std::string> vnames{"d_eta-x", "d_eta-y", "d_eta-z", "phi"};
        Vector<const MultiFab*> vars;
        vars.push_back(&d_eta);
        vars.push_back(&phi_new);

        int n = 0;
        const std::string& pltfile = amrex::Concatenate("plt",n,5);
        Vector<Geometry> geomarr(1,geom);
        Vector<int> level_steps(1,n);
        Vector<IntVect> ref_ratio;

        WriteMultiLevelPlotfile(pltfile, 1, vars, vnames, geomarr, time,
                                level_steps, ref_ratio);
//        WriteSingleLevelPlotfile(pltfile, d_eta, {"d_eta"}, geom, time, n);
    }

    // build the flux multifabs
    Array<MultiFab, AMREX_SPACEDIM> flux;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        BoxArray edge_ba = ba;
        edge_ba.surroundingNodes(dir);
        flux[dir].define(edge_ba, dm, 1, 0);
    }

    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        // new_phi = old_phi + dt * (something)
        advance(phi_old, phi_new, flux, dt, geom);
        time = time + dt;

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << n << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            // assemble variables for plotting
            Vector<std::string> vnames{"d_eta-x", "d_eta-y", "d_eta-z", "phi"};
            Vector<const MultiFab*> vars;
            vars.push_back(&d_eta);
            vars.push_back(&phi_new);

            const std::string& pltfile = amrex::Concatenate("plt",n,5);
            Vector<Geometry> geomarr(1,geom);
            Vector<int> level_steps(1,n);
            Vector<IntVect> ref_ratio;

            WriteMultiLevelPlotfile(pltfile, 1, vars, vnames, geomarr, time,
                                    level_steps, ref_ratio);
//            WriteSingleLevelPlotfile(pltfile, d_eta, {"d_eta"}, geom, time, n);
        }
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    auto stop_time = ParallelDescriptor::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}
