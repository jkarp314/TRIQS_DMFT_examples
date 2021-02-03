from pytriqs.sumk import *
from pytriqs.gf import *
from pytriqs.lattice.tight_binding import TBLattice
from pytriqs.lattice.super_lattice import TBSuperLattice
import pytriqs.utility.mpi as mpi
from pytriqs.utility.dichotomy import dichotomy
from triqs_cthyb import *
from pytriqs.archive import HDFArchive
from pytriqs.operators import *
import numpy as np


beta = 40.
U = 3.
t = -1.
tp = 0.
nk = 40
nloops = 15
prec_mu = 1e-5

outfile = 'U%.1f'%U

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 100
p["n_warmup_cycles"] = int(5e5)
p["n_cycles"] = int(2e9/mpi.size)
p["move_double"]= True
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4
p["fit_min_w"] = 5
p["fit_max_w"] = 15

a = .5
b = 1/np.sqrt(2)
Q = np.array([[a, b, 0, a], [a, 0, b, -a], [a, 0, -b, -a], [a, -b, 0, a]]) #matrix that diagonalizes the local G0

#find 
h_int = 0
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(4):
                    coeff = U*Q[i,j]*Q[i,k]*Q[i,l]*Q[i,m]
                    if abs(coeff) > 1e-8:
                        h_int += coeff*c_dag('up_%d'%j,0)*c('up_%d'%k,0)*c_dag('down_%d'%l,0)*c('down_%d'%m,0)


hop= {  (1,0)  :  [[ t]],       
        (-1,0) :  [[ t]],     
        (0,1)  :  [[ t]],
        (0,-1) :  [[ t]],
        (1,1)  :  [[ tp]],
        (-1,-1):  [[ tp]],
        (1,-1) :  [[ tp]],
        (-1,1) :  [[ tp]]}


L = TBLattice(units = [(1, 0, 0) , (0, 1, 0), (0,0,1)], hopping = hop, orbital_names= range(1), orbital_positions= [(0., 0., 0.)])
SL = TBSuperLattice(tb_lattice =L, super_lattice_units = [ (2,0,0), (0,2,0)])
SK = SumkDiscreteFromLattice(lattice=SL, n_points=nk)
mesh = GfImFreq(indices = [0], beta = beta).mesh
Gloc = BlockGf( name_block_generator = [ (s, GfImFreq(indices = SK.GFBlocIndices, mesh = mesh)) for s in ['up', 'down'] ], make_copies = False)
Sigma_lat = Gloc.copy()
Gloc1 = Solver(beta=beta, gf_struct = [('up', range(4)), ('down', range(4))]).G_iw.copy() #Gloc in just the correlated subspace
Sigma_orig = Gloc1.copy() #Sigma for the correlated subspace in the original basis
Sigma_orig.zero()
Sigma_lat.zero()

previous_runs = 0
previous_present = False
if mpi.is_master_node():
    ar = HDFArchive(outfile+'.h5', 'a')
    if 'iterations' in ar:
        previous_present = True
        previous_runs = ar['iterations']
        Sigma_orig = ar['Sigma_orig']
        chemical_potential = ar['mu-%d'%previous_runs]
        del ar
else:
    chemical_potential = None
previous_runs = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)
Sigma_orig = mpi.bcast(Sigma_orig)
chemical_potential = mpi.bcast(chemical_potential)

gf_struct = []
for sp in ['up', 'down']:
    for i in range(4):
        gf_struct.append((sp+'_%d'%i,[0]))
S = Solver(beta = beta, gf_struct = gf_struct)


for iteration_number in range(1,nloops+1):
    niter = iteration_number + previous_runs
    if mpi.is_master_node():
        print '-----------------------------------------------'
        print "Iteration = ", niter
        print '-----------------------------------------------'

    for sp in ['up', 'down']:
        for i in range(4):
            Sigma_lat[sp].data[:,i,i] = Sigma_orig[sp].data[:, i, i] - dc
    
            for j in range(4):
                if i != j:
                    Sigma_lat[sp].data[:,i,j] = Sigma_orig[sp].data[:, i, j]
    
    def Dens(mu):
        return SK(mu = mu, Sigma = Sigma_lat).total_density().real

    chemical_potential, density = dichotomy(Dens, chemical_potential, 4., prec_mu, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
    Gloc << SK(mu = chemical_potential, Sigma = Sigma_lat)

    for sp in ['up', 'down']:
        Gloc1[sp].data[:,:,:] = Gloc[sp].data[:, :4, :4]

    nlat = Gloc1.total_density().real

    G0_orig = Gloc1.copy()
    G0_orig << inverse(Sigma_orig + inverse(Gloc1))

    G0_diag = G0_orig.copy() #G0 in basis that diagonalizes it
    for i in range(len(G0_orig.mesh)):
        G0_diag['up'].data[i,:,:] = Q.T.dot(G0_orig['up'].data[i,:,:]).dot(Q)
        G0_diag['down'].data[i,:,:] = Q.T.dot(G0_orig['up'].data[i,:,:]).dot(Q)

    for i in range(4):
        S.G0_iw['up_%d'%i].data[:,0,0] = G0_diag['up'].data[:,i,i]
        S.G0_iw['down_%d'%i].data[:,0,0] = G0_diag['down'].data[:,i,i]

    S.solve(h_int=h_int, **p)

    for name, s_iw in S.Sigma_iw:
        S.Sigma_iw[name] = s_iw.make_hermitian()

    Sigma_diag = G0_diag.copy()
    for i in range(4):
        Sigma_diag['up'].data[:,i,i] = S.Sigma_iw['up_%d'%i].data[:,0,0]
        Sigma_diag['down'].data[:,i,i] = S.Sigma_iw['down_%d'%i].data[:,0,0]

    for i in range(len(G0_orig.mesh)):
        Sigma_orig['up'].data[i,:,:] = .5*(Q.dot(Sigma_diag['up'].data[i,:,:]).dot(Q.T) + Q.dot(Sigma_diag['down'].data[i,:,:]).dot(Q.T))
        Sigma_orig['down'] << Sigma_orig['up']

    G_orig = G0_orig.copy()
    G_orig << inverse(inverse(G0_orig) - Sigma_orig)

    nimp = G_orig.total_density().real

    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
        ar['iterations'] = niter
        ar['G0_diag'] = S.G0_iw
        ar['G_tau_diag'] = S.G_tau
        ar['G_orig'] = G_orig
        ar['G_diag'] = S.G_iw
        ar['Sigma_diag'] = S.Sigma_iw
        ar['Sigma_orig'] = Sigma_orig
        ar['Sigma_orig-%d'%niter] = Sigma_orig
        ar['nimp-%d'%niter] = nimp
        ar['nlat-%d'%niter] = nlat
        ar['mu-%d'%niter] = chemical_potential
        del ar

    if mpi.is_master_node():
        ar = HDFArchive(outfile+'_iterations.h5', 'a')
        ar['iterations'] = niter
        ar['solver-%d'%niter] = S
        ar['Gloc-%d'%niter] = Gloc
        ar['Gloc1-%d'%niter] = Gloc1
        ar['mu-%d'%niter] = chemical_potential
        ar['G_orig-%d'%niter] = G_orig
        ar['G_diag-%d'%niter] = S.G_iw
        ar['Sigma_diag-%d'%niter] = S.Sigma_iw
        ar['Sigma_orig-%d'%niter] = Sigma_orig
        del ar


