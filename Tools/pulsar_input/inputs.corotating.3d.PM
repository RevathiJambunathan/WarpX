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
max_step = 20000
amr.n_cell = 768 768 768
amr.max_grid_size = 64
amr.blocking_factor = 64
amr.max_level = 0
geometry.coord_sys   = 0                  # 0: Cartesian
geometry.dims = 3
geometry.prob_lo     = 0.0 0.0 0.0
geometry.prob_hi     = 180000   180000   180000
boundary.field_lo = pml pml pml
boundary.field_hi = pml pml pml
#amr.restart = ./diags/chk004500

#################################
############ NUMERICS ###########
#################################
algo.maxwell_solver = yee
warpx.verbose = 1
warpx.do_dive_cleaning = 0
warpx.use_filter = 1
warpx.cfl = .95
warpx.do_pml_in_domain = 1
warpx.pml_has_particles = 1
warpx.do_pml_j_damping = 1
my_constants.pi    = 3.141592653589793
my_constants.dens  = 5.544e6
my_constants.xc  = 90000
my_constants.yc  = 90000
my_constants.zc  = 90000
my_constants.r_star  = 12000
my_constants.omega = 6245.676
my_constants.c = 299792458.
my_constants.B_star = 8.0323e-6                  # magnetic field of NS (T)
my_constants.dR = 234.375
my_constants.to = 2.e-4
my_constants.skin_depth = 2000
algo.particle_shape = 3

########################################
########### load balancing #############
########################################
algo.load_balance_intervals = 100
algo.load_balance_efficiency_ratio_threshold = 1.05
algo.load_balance_knapsack_factor = 1.2
algo.costs_heuristic_particles_wt = 0.65
algo.costs_heuristic_cells_wt = 0.35
algo.load_balance_costs_update = heuristic
algo.load_balance_with_sfc = 1

#################################
############ PLASMA #############
#################################
particles.species_names = plasma_e plasma_p

plasma_e.charge = -q_e
plasma_e.mass = m_e
plasma_e.injection_style = "NUniformPerCell"
plasma_e.profile = parse_density_function
plasma_e.density_function(x,y,z) = "( ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<=(r_star + skin_depth)) * ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>=(r_star+skin_depth-dR)) )*dens"
plasma_e.num_particles_per_cell_each_dim = 2 2 2
plasma_e.momentum_distribution_type = parse_momentum_function

## Inject stationary pairs
plasma_e.momentum_function_ux(x,y,z) = "0.0"
plasma_e.momentum_function_uy(x,y,z) = "0.0"
## uz is always 0 for the injection methods above
plasma_e.momentum_function_uz(x,y,z) = "0.0"
plasma_e.do_continuous_injection = 0
plasma_e.density_min = 0.1

plasma_p.charge = q_e
plasma_p.mass = m_e
plasma_p.injection_style = "NUniformPerCell"
plasma_p.profile = parse_density_function
plasma_p.density_function(x,y,z) = "( ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<=(r_star+skin_depth)) * ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>=(r_star+skin_depth-dR)) )*dens"
plasma_p.num_particles_per_cell_each_dim = 2 2 2
plasma_p.momentum_distribution_type = parse_momentum_function
## Inject stationary pairs
plasma_p.momentum_function_ux(x,y,z) = "0.0"
plasma_p.momentum_function_uy(x,y,z) = "0.0"
## uz ip always 0 for the injection methods above
plasma_p.momentum_function_uz(x,y,z) = "0.0"
plasma_p.do_continuous_injection = 0
plasma_p.density_min = 0.1

#### Adding particle attributes ###
plasma_e.addRealAttributes = injectiontime
plasma_e.attribute.injectiontime(x,y,z,ux,uy,uz,t) = t
plasma_p.addRealAttributes = injectiontime
plasma_p.attribute.injectiontime(x,y,z,ux,uy,uz,t) = t

diagnostics.diags_names = plt chk
plt.diag_type = Full
plt.intervals = 500
plt.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz divE conductor rho part_per_cell rho_plasma_e rho_plasma_p magnetization ndens_plasma_p ndens_plasma_e injectionflag

chk.format = checkpoint
chk.diag_type = Full
chk.intervals = 500

############ Reduced Diags #########
warpx.reduced_diags_names = total_rho_domain
#warpx.reduced_diags_names = divE_conductor divE_domain Ex_xmin Ey_ymin Ez_zmin Ex_xmax Ey_ymax Ez_zmax total_rho_conductor total_rho_domain LB_cost LB_eff PartNum

LB_cost.type = LoadBalanceCosts
LB_cost.intervals = 50

LB_eff.type = LoadBalanceEfficiency
LB_eff.intervals = 50

PartNum.type = ParticleNumber
PartNum.intervals = 10

total_rho_conductor.type = ParticleNumber
total_rho_conductor.filter_function = " ( (x >= xc-r_star) * (x <= xc+r_star) * (y >= yc-r_star) * (y <= yc+r_star) * (z >= zc-r_star) * (z <= zc+r_star) ) * 1 + 0"
total_rho_conductor.intervals = 1

total_rho_domain.type = ParticleNumber
total_rho_domain.filter_function = " (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc) )^0.5 >= (r_star + skin_depth)"
total_rho_domain.intervals = 500

divE_conductor.type = divEReduction
divE_conductor.reduction_type = integral
divE_conductor.integration_type = volume
divE_conductor.reduced_function(x,y,z) = " ( (x >= xc-r_star) * (x <= xc+r_star) * (y >= yc-r_star) * (y <= yc+r_star) * (z >= zc-r_star) * (z <= zc+r_star) ) * 1 + 0"
divE_conductor.intervals = 1

divE_domain.type = divEReduction
divE_domain.reduction_type = integral
divE_domain.integration_type = volume
divE_domain.reduced_function(x,y,z) = " 1"
divE_domain.intervals = 1

my_constants.bbox = 14702
my_constants.bbox_mindx = 14655.2
my_constants.bbox_maxdx = 14877.5

Ex_xmin.type = RawEFieldReduction
Ex_xmin.reduction_type = integral
Ex_xmin.integration_type = surface
Ex_xmin.reduced_function(x,y,z) = " ( (x >= xc-bbox_maxdx) * (x <= xc-bbox_mindx) * (y >= yc-bbox) * (y <= yc+bbox) * (z >= zc-bbox) * (z <= zc+bbox) ) * 1 + 0"
Ex_xmin.intervals = 1

Ex_xmax.type = RawEFieldReduction
Ex_xmax.reduction_type = integral
Ex_xmax.integration_type = surface
Ex_xmax.reduced_function(x,y,z) = " ( (x >= xc+bbox_mindx) * (x <= xc+bbox_maxdx) * (y >= yc-bbox) * (y <= yc+bbox) * (z >= zc-bbox) * (z <= zc+bbox) ) * 1 + 0"
Ex_xmax.intervals = 1

Ey_ymin.type = RawEFieldReduction
Ey_ymin.reduction_type = integral
Ey_ymin.integration_type = surface
Ey_ymin.reduced_function(x,y,z) = " ( (x >= xc-bbox) * (x <= xc+bbox) * (y >= yc-bbox_maxdx) * (y <= yc-bbox_mindx) * (z >= zc-bbox) * (z <= zc+bbox) ) * 1 + 0"
Ey_ymin.intervals = 1

Ey_ymax.type = RawEFieldReduction
Ey_ymax.reduction_type = integral
Ey_ymax.integration_type = surface
Ey_ymax.reduced_function(x,y,z) = " ( (x >= xc-bbox) * (x <= xc+bbox) * (y >= yc+bbox_mindx) * (y <= yc+bbox_maxdx) * (z >= zc-bbox) * (z <= zc+bbox) ) * 1 + 0"
Ey_ymax.intervals = 1

Ez_zmin.type = RawEFieldReduction
Ez_zmin.reduction_type = integral
Ez_zmin.integration_type = surface
Ez_zmin.reduced_function(x,y,z) = " ( (x >= xc-bbox) * (x <= xc+bbox) * (y >= yc-bbox) * (y <= yc+bbox) * (z >= zc-bbox_maxdx) * (z <= zc-bbox_mindx) ) * 1 + 0"
Ez_zmin.intervals = 1

Ez_zmax.type = RawEFieldReduction
Ez_zmax.reduction_type = integral
Ez_zmax.integration_type = surface
Ez_zmax.reduced_function(x,y,z) = " ( (x >= xc-bbox) * (x <= xc+bbox) * (y >= yc-bbox) * (y <= yc+bbox) * (z >= zc+bbox_mindx) * (z <= zc+bbox_maxdx) ) * 1 + 0"
Ez_zmax.intervals = 1

#################################
######### PULSAR SETUP ##########
#################################
pulsar.pulsarType = "dead"             # [dead/active]: sets particle injection type
pulsar.omega_star = 6245.676               # angular velocity of NS (rad/s)
pulsar.ramp_omega_time = 5e-4             # time over which to ramp up NS angular velocity (s)
                                           # if ramp_omega_time < 0, omega = omega_star for any t
                                           # consistency requires ramp_omega_time = my_constants.to
pulsar.center_star = 90000 90000 90000
pulsar.R_star = 12000                  # radius of NS (m)
pulsar.B_star = 8.0323e-6                  # magnetic field of NS (T)
pulsar.dR = 234.375                       # thickness on the surface of the pulsar
                                       # consistency requires dR = my_constants.dR
pulsar.verbose = 0                     # [0/1]: turn on verbosity for debugging print statements
pulsar.EB_external = 0                 # [0/1]: to apply external E and B
pulsar.EB_corotating_maxradius = 14000 # The radius where E-field changes from corotating
                                       # inside the star to quadrapole outside.
                                       # Default is R_star. (r<=cor_radius) i.e. the includes
                                       # the r specified in the input
pulsar.max_ndens = 5.54e6              # max ndens == ndens used in initializing density
pulsar.Ninj_fraction = 1.0            # fraction of sigma injected
pulsar.ModifyParticleWeight = 0       # (0/1) the fractional injection is achieved
                                      # by modifying particle weight if 1
                                      # by modifying num_ppc if 0
pulsar.particle_inj_rmin = 13672.6      # default is Rstar-dR (consistent with density function above)
pulsar.particle_inj_rmax = 14000      # default is Rstar (consistent with density function)
pulsar.max_particle_absorption_radius = 12000   # Maximum radius within which particles are
                                                # deleted every timestep.
                                                # Default is Rstar
pulsar.rhoGJ_scale = 1e0              # scaling down of rho_GJ
pulsar.damp_EB_internal = 0             # Damp internal electric field
pulsar.damp_EB_radius = 12000         # The radius within which EB should be damped.
                                      # default is r_star (damping will include this radius)
pulsar.damping_scale = 1000.0            # Damping scale factor for internal electric field
pulsar.turnoffdeposition = 0         # (0/1) 0 for depositing (default)
                                      #       1 for no deposition
pulsar.max_nodepos_radius = 0.        # radius within which particle deposition for j/rho
                                      # is turned off. (r<=max_radius)
pulsar.turnoff_plasmaEB_gather = 0    # (0/1) 0 is default. always gather
                                      # 1 for no gather of self-consistent fields
pulsar.max_nogather_radius = 0.       # radius within which self-consistent field gather
pulsar.init_dipoleBfield = 1          # default is 1
pulsar.init_corotatingEfield = 1      # default is 1
pulsar.init_corotatingAndExternalEField = 0 # default is 0
pulsar.corotatingE_maxradius = 14000  # default is Rstar
pulsar.enforceDipoleB_maxradius = 12000 # default is Rstar
pulsar.enforceCorotatingE = 1       # default 1
pulsar.enforceDipoleB = 1        # default 1
pulsar.singleParticleTest = 0    # default 0, if 1, then a particle pair will be introduced.
                                 # Single particle position/momentum will need to be specified
pulsar.continuous_injection = 0
pulsar.injection_time = 0.
pulsar.DampBDipoleInRing = 0
pulsar.Bdamping_scale = 10000
pulsar.EnforceTheoreticalEBInGrid = 0
pulsar.E_external_monopole = 0                  # [0/1] must be 1 when using use_theoretical = 1
pulsar.use_theoreticalEB = 0     # theoretical EB is added to particles directly
pulsar.theory_max_rstar = 0  # one cell below rstar
pulsar.AddExternalMonopoleOnly = 1
pulsar.conductor_function(x,y,z) = " ( ((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc))^0.5 <= r_star + skin_depth ) * 1 + 0."    # sphere
pulsar.ApplyEfieldBCusingConductor = 0   # this sets Efield to 0 in the conductor
pulsar.FilterWithConductor = 0           # uses method of images for filtering. does not work at corners
pulsar.minimum_Sigma0 = 1.e-300
pulsar.maximum_Sigma0 = 1.e+300
pulsar.injection_tuning_interval = 10
pulsar.plasma_injection_rate = 0.2
pulsar.ROI_avg_size = 50
pulsar.sigma_tune_method = relative_difference   # 10percent or relative_difference
pulsar.modify_sigma_threshold = 1
pulsar.print_injected_celldata = 0
pulsar.print_celldata_starttime = 100000000
# to change lower bound number density for magnetization use the input below
#pulsar.lowerBound_ndens_magnetization = 1.e-16
# to change upperBound for relative difference method to tune sigma use input below
#pulsar.upperBound_reldiff_sigma0 = 0.1
