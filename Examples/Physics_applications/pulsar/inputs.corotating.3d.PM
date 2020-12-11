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
max_step = 50000
amr.n_cell = 128 128 128
amr.max_grid_size = 128
amr.blocking_factor = 128
amr.max_level = 0
geometry.coord_sys   = 0                  # 0: Cartesian
geometry.is_periodic = 0 0 0     # Is periodic?
geometry.prob_lo     = 0.0    0.0    0.0
geometry.prob_hi     = 180000 180000 180000

#################################
############ NUMERICS ###########
#################################
#algo.maxwell_fdtd_solver = yee
warpx.verbose = 1
warpx.do_dive_cleaning = 0
warpx.use_filter = 1
warpx.cfl = .2
warpx.do_pml = 1
my_constants.pi    = 3.141592653589793
my_constants.dens  = 5.544e6
my_constants.xc  = 90000
my_constants.yc  = 90000
my_constants.zc  = 90000
my_constants.r_star  = 12032
my_constants.omega = 6245.676
my_constants.c = 299792458.
my_constants.B_star = 8.0323e-6                  # magnetic field of NS (T)
my_constants.dR = 1400
my_constants.to = 2.e-4
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
plasma_e.density_function(x,y,z) = "( ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<=(r_star)) * ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>=(r_star-dR)) )*dens"
plasma_e.num_particles_per_cell_each_dim = 4 4 4
plasma_e.momentum_distribution_type = parse_momentum_function

## Inject stationary pairs
plasma_e.momentum_function_ux(x,y,z) = "0.0"
plasma_e.momentum_function_uy(x,y,z) = "0.0"
## uz is always 0 for the injection methods above
plasma_e.momentum_function_uz(x,y,z) = "0.0"

plasma_e.do_continuous_injection = 0
plasma_e.density_min = 1E0

plasma_p.charge = q_e
plasma_p.mass = m_e
plasma_p.injection_style = "NUniformPerCell"
plasma_p.profile = parse_density_function
plasma_p.density_function(x,y,z) = "( ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<=(r_star)) * ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>=(r_star-dR)) )*dens"
plasma_p.num_particles_per_cell_each_dim = 4 4 4
plasma_p.momentum_distribution_type = parse_momentum_function

## Inject stationary pairs
plasma_p.momentum_function_ux(x,y,z) = "0.0"
plasma_p.momentum_function_uy(x,y,z) = "0.0"
## uz is always 0 for the injection methods above
plasma_p.momentum_function_uz(x,y,z) = "0.0"

plasma_p.do_continuous_injection = 0
plasma_p.density_min = 1E0

diagnostics.diags_names = plt chk
plt.diag_type = Full
plt.period = 20
plt.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz part_per_cell rho divE

chk.format = checkpoint
chk.diag_type = Full
chk.period = 1000

#################################
######### PULSAR SETUP ##########
#################################
pulsar.pulsarType = "dead"             # [dead/active]: sets particle injection type
pulsar.omega_star = 6245.676               # angular velocity of NS (rad/s)
pulsar.ramp_omega_time = 0.0             # time over which to ramp up NS angular velocity (s)
                                           # if ramp_omega_time < 0, omega = omega_star for any t
                                           # consistency requires ramp_omega_time = my_constants.to
pulsar.center_star = 90000 90000 90000
pulsar.R_star = 12032                  # radius of NS (m)
pulsar.B_star = 8.0323e-6                  # magnetic field of NS (T)
pulsar.dR = 1400                       # thickness on the surface of the pulsar
pulsar.Chi = 45
                                       # consistency requires dR = my_constants.dR
pulsar.verbose = 0                     # [0/1]: turn on verbosity for debugging print statements
pulsar.EB_external = 1                 # [0/1]: to apply external E and B
pulsar.E_external_monopole = 1                  # [0/1]
pulsar.max_ndens = 5.54e6              # max ndens == ndens used in initializing density
pulsar.Ninj_fraction = 0.2            # fraction of sigma injected
pulsar.rhoGJ_scale = 1e0              # scaling down of rho_GJ
pulsar.damp_EB_internal = 1             # Damp internal electric field
pulsar.damping_scale = 1000.0            # Damping scale factor for internal electric field
