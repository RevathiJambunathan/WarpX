# Description:
#
# This inputs file sets up a NS with these properties:
# - E = -(omega*R)[cross]B inside the NS
# - external E and B outside the star
#
# See the "Pulsar Setup" section at the end for the options
#
# This initializes the electrons and positrons with a corotating momentum function.
# Based on the pulsar_type = "active" or "dead", particles are injected continuously or 
#                             until rho_GJ is reached

#################################
####### GENERAL PARAMETERS ######
#################################
max_step = 5000
#amr.n_cell = 256 256 256
amr.n_cell = 128 128 128
amr.max_grid_size = 64
amr.blocking_factor = 64
amr.max_level = 0
amr.plot_file = "plotfiles/plt"
amr.plot_int = 25
geometry.coord_sys   = 0                  # 0: Cartesian
geometry.is_periodic = 0 0 0     # Is periodic?
geometry.prob_lo     = 0.0    0.0    0.0
geometry.prob_hi     = 180000 180000 180000

#################################
############ NUMERICS ###########
#################################
algo.maxwell_fdtd_solver = yee
warpx.verbose = 1
warpx.plot_raw_fields = 0
warpx.do_dive_cleaning = 0
warpx.use_filter = 1
warpx.cfl = .99
warpx.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz part_per_cell rho divE
my_constants.pi    = 3.141592653589793
my_constants.dens  = 5.544e6
my_constants.xc  = 90000
my_constants.yc  = 90000
my_constants.zc  = 90000
my_constants.r_star  = 12000
my_constants.omega = 6245.676
my_constants.c = 299792458.
my_constants.B_star = 8.0323e-6                  # magnetic field of NS (T)
my_constants.dR = 2e3
interpolation.nox = 3
interpolation.noy = 3
interpolation.noz = 3


#################################
############ PLASMA #############
#################################
particles.nspecies = 2
particles.species_names = plasma_e plasma_p

plasma_e.charge = -q_e
plasma_e.mass = m_e
plasma_e.injection_style = "NUniformPerCell"
plasma_e.profile = parse_density_function
plasma_e.density_function(x,y,z) = "( ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star) * ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>(r_star-dR)) )*dens"
plasma_e.num_particles_per_cell_each_dim = 3 3 3
plasma_e.momentum_distribution_type = parse_momentum_function
plasma_e.momentum_function_ux(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(-omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (y-yc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_e.momentum_function_uy(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (x-xc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_e.momentum_function_uz(x,y,z) = "0.0"
plasma_e.do_continuous_injection = 0
plasma_e.density_min = 1E0

plasma_p.charge = q_e
plasma_p.mass = m_e
plasma_p.injection_style = "NUniformPerCell"
plasma_p.profile = parse_density_function
plasma_p.density_function(x,y,z) = "( ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)* ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>(r_star-dR)))*dens"
plasma_p.num_particles_per_cell_each_dim = 3 3 3
plasma_p.momentum_distribution_type = parse_momentum_function
plasma_p.momentum_function_ux(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(-omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (y-yc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_p.momentum_function_uy(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (x-xc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_p.momentum_function_uz(x,y,z) = "0.0"
plasma_p.do_continuous_injection = 0
plasma_p.density_min = 1E0


#################################
######### PULSAR SETUP ##########
#################################
pulsar.pulsarType = "dead"             # [dead/active]: sets particle injection type
pulsar.omega_star = 6245.676               # angular velocity of NS (rad/s)
pulsar.R_star = 12.e3                  # radius of NS (m)
pulsar.B_star = 8.0323e-6                  # magnetic field of NS (T)
pulsar.dR = 2e3
pulsar.verbose = 0                     # [0/1]: turn on verbosity for debugging print statements
