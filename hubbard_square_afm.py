#DMFT calculation for a hubbard model on a 2D square lattice
#creates a supercell and allows for anti-ferromagnetism
#uses Triqs version 2.2

from pytriqs.sumk import *
from pytriqs.gf import *
from pytriqs.lattice.tight_binding import TBLattice
from pytriqs.lattice.super_lattice import TBSuperLattice
import pytriqs.utility.mpi as mpi
from pytriqs.utility.dichotomy import dichotomy
from triqs_cthyb import *
from pytriqs.archive import HDFArchive
from pytriqs.operators import *

beta = 40.
t = -1.                   #nearest neighbor hopping
tp = 0.                   #next nearest neighbor hopping
U = 3                     # hubbard U parameter
nloops = 15               # number of DMFT loops
nk = 30                   # number of k points in each dimension
density_required = 1.     # target density for setting the chemical potential

outfile = 'U%.1f_afm'%U

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 200
p["n_warmup_cycles"] = int(5e4)  
p["n_cycles"] = int(1e7/mpi.size)
# tail fit
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4
p["fit_min_w"] = 5
p["fit_max_w"] = 15

S = Solver(beta=beta, gf_struct = [('up', [0]), ('down', [0])])
h_int = U * n('up',0) * n('down',0) #local interating hamiltonian

hop= {  (1,0)  :  [[ t]],       
        (-1,0) :  [[ t]],     
        (0,1)  :  [[ t]],
        (0,-1) :  [[ t]],
        (1,1)  :  [[ tp]],
        (-1,-1):  [[ tp]],
        (1,-1) :  [[ tp]],
        (-1,1) :  [[ tp]]}

L = TBLattice(units = [(1, 0, 0) , (0, 1, 0)], hopping = hop, orbital_names= range(1), orbital_positions= [(0., 0., 0.)]*1)
SL = TBSuperLattice(tb_lattice =L, super_lattice_units = [ (1,1,0), (1,-1,0)])
SK = SumkDiscreteFromLattice(lattice=SL, n_points=nk)

mesh = S.G_iw.mesh
Gloc = BlockGf( name_block_generator = [ (s, GfImFreq(indices = SK.GFBlocIndices, mesh = mesh)) for s in ['up', 'down'] ], make_copies = False)
Sigma_lat = Gloc.copy()

#function to extract density for a given mu, to be used by dichotomy function to determine mu
def Dens(mu):
    dens =  SK(mu = mu, Sigma = Sigma_lat).total_density()
    if abs(dens.imag) > 1e-20:
            mpi.report("Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
    return dens.real

#check if there are previous runs in the outfile and if so restart from there
previous_runs = 0
previous_present = False
mu = 0.
if mpi.is_master_node():
    ar = HDFArchive(outfile+'.h5','a')
    if 'iterations' in ar:
        previous_present = True
        previous_runs = ar['iterations']
        S.Sigma_iw = ar['Sigma_iw']
        mu = ar['mu-%d'%previous_runs]
        del ar
previous_runs    = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)
S.Sigma_iw = mpi.bcast(S.Sigma_iw)
mu = mpi.bcast(mu)

for iteration_number in range(1,nloops+1):
    it = iteration_number + previous_runs
    if mpi.is_master_node():
        print('-----------------------------------------------')
        print("Iteration = %s"%it)
        print('-----------------------------------------------')

    if it > 1:
        #set the lattice self energy from the impurity self energy
        Sigma_lat['up'].data[:,0,0] = S.Sigma_iw['up'].data[:,0,0]
        Sigma_lat['down'].data[:,0,0] = S.Sigma_iw['down'].data[:,0,0]
        
        #switch the up and down self energy for the second site
        Sigma_lat['up'].data[:,1,1] = S.Sigma_iw['down'].data[:,0,0]
        Sigma_lat['down'].data[:,1,1] = S.Sigma_iw['up'].data[:,0,0]
        
    mu, density = dichotomy(Dens, mu, density_required, 1e-4, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
    Gloc << SK(mu = mu, Sigma = Sigma_lat)

    Gloc1 = S.G_iw.copy() #Gloc for one of the sites
    Gloc1['up'].data[:,0,0] = Gloc['up'].data[:,0,0]
    Gloc1['down'].data[:,0,0] = Gloc['down'].data[:,0,0]
    
    nlat = Gloc1.total_density().real # lattice density

    # set starting guess for Sigma = U/2 at first iteration
    if it == 1:
        S.Sigma_iw << .5*U

    S.G0_iw << inverse(S.Sigma_iw + inverse(Gloc1))
    S.solve(h_int=h_int, **p)

    #force self energy obtained from solver to be hermitian
    for name, s_iw in S.Sigma_iw:
        S.Sigma_iw[name] = make_hermitian(s_iw)

    nimp = S.G_iw.total_density().real  #impurity density

    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
        ar['iterations'] = it
        ar['G_0'] = S.G0_iw
        ar['G_tau'] = S.G_tau
        ar['G_iw'] = S.G_iw
        ar['Sigma_iw'] = S.Sigma_iw
        ar['Sigma_-%s'%it] = S.Sigma_iw
        ar['nimp-%s'%it] = nimp
        ar['nlat-%s'%it] = nlat
        ar['mu-%s'%it] = mu
        del ar



