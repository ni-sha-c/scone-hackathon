# grad(L (phi) + grad phi.q) = p - q
from torch import optim,nn,cuda,sparse,linalg,func 
from torch import *
from scipy.interpolate import RectBivariateSpline as i2d
from matplotlib.pyplot import *
# initialize grid
device = 'cuda' if cuda.is_available() else 'cpu'
eye_cuda = eye(2).to(device)
def laplacian(n_x,n_y,dx,dy):
    nx, ny = n_x-2, n_y-2
    idx2, idy2 = 1/dx/dx, 1/dy/dy
    n = nx*ny
    diags = ones(3,n)
    diags[0] = -2.0*diags[0]
    A = sparse.spdiags(idx2*diags, tensor([0, 1, -1]), (n,n))
    A += sparse.spdiags(idy2*diags, tensor([0,nx,-nx]), (n,n))
    A = A.to_dense()
    bcy = nx*arange(1, ny, dtype=int)
    A[bcy-1, bcy] = 0
    A[bcy, bcy-1] = 0
    return A.to(device)

def q_gradphi(q, n_x, n_y, dx, dy):
    ny, nx = n_y-2, n_x-2
    idx, idy = 0.5/dx, 0.5/dy
    n = ny*nx
    d_dx_p, d_dx_m = idx*q[0,:-1],-idx*q[0,1:]
    d_dy_p, d_dy_m = idy*q[1,:-nx],-idy*q[1,nx:]
    A = diagflat(d_dx_p, 1) + diagflat(d_dx_m, -1)
    A += diagflat(d_dy_p, nx) + diagflat(d_dy_m, -nx)
    # BC
    bcy = nx*arange(1, ny, dtype=int)
    A[bcy-1, bcy] = 0
    A[bcy, bcy-1] = 0
    return A.to(device)

def def_grid(x_max, x_min, y_max, y_min, n_gr_x, n_gr_y):
    dx = (x_max - x_min)/(n_gr_x-1) 
    idx = 1/dx
    dy = (y_max -y_min)/(n_gr_y-1)
    idy = 1/dy
    grid_x = linspace(x_min, x_max, n_gr_x)
    grid_y = linspace(y_min, y_max, n_gr_y)  
    x_grid, y_grid = meshgrid(grid_x, grid_y,indexing="xy") 
    interior_pts = stack((x_grid[1:-1,1:-1], y_grid[1:-1,1:-1]), dim=2).reshape((n_gr_x-2)*(n_gr_y-2), 2)
    return x_grid, y_grid, dx, dy, interior_pts




def target_score(z):
    return eye_cuda @ ones(2).to(device) 


def density_ratio(z):
    x, y = z
    return -2*sin(y)*sin(x) + sin(x+y)

def pde_setup(x_max, x_min, y_max, y_min, n_gr_x, n_gr_y):
    x_grid, y_grid, dx, dy, grid_pts = def_grid(x_max, x_min, y_max, y_min, n_gr_x, n_gr_y)
    # x_grid.shape = y_grid.shape = [n_gr_y, n_gr_x]
    # grid_pts = interior points
    grid_pts = grid_pts.to(device)
    q = vmap(target_score)(grid_pts).reshape(-1,2).mT
    rhs = vmap(density_ratio)(grid_pts)
    return q, rhs, dx, dy, x_grid, y_grid

def discretized_pdo(q, nx, ny, dx, dy):
    A1 = q_gradphi(q, nx, ny, dx, dy)
    A2 = laplacian(nx, ny, dx, dy)
    return A1 + A2
    




def solve_pde(x_max, x_min, y_max, y_min, n_gr_x, n_gr_y):
    q, rhs, dx, dy, x_gr, y_gr = pde_setup(x_max, x_min, y_max, y_min, n_gr_x, n_gr_y)
    n = (n_gr_x-2)*(n_gr_y-2)
    lhs = discretized_pdo(q, n_gr_x, n_gr_y, dx, dy) #+ 1.e-8*eye(n, n).to(device)
    int_sol = linalg.solve(lhs,rhs)
    # reshape and pad
    phi = zeros(n_gr_y, n_gr_x)
    int_sol = int_sol.reshape(n_gr_y-2,n_gr_x-2)
    phi[1:-1,1:-1] = int_sol.detach().cpu()
    return phi, x_gr, y_gr

def dphi(phi, dx, dy):
    ny, nx = phi.shape
    vx, vy = zeros(ny-2,nx-2), zeros(ny-2,nx-2)
    vx = (phi[1:ny-1,2:]-phi[1:ny-1,:nx-2])/(2*dx)
    vy = (phi[2:,1:nx-1]-phi[:ny-2,1:nx-1])/(2*dy)
    return stack([vx, vy])  

#def d2phi(dphi, dx, dy):

def logdet_dT(v, dx, dy):
    ny, nx = v[0].shape
    dvx, dvy = zeros(2,ny-2,nx-2).to(device), zeros(2,ny-2,nx-2).to(device)
    dvx[0] = (vx[1:-1,2:]-vx[1:-1,:-1])/(2*dx)
    dvx[1] = (vx[2:,1:-1]-vx[:-1,1:-1])/(2*dy)
    dvy[1] = (vy[2:,1:-1]-vy[:-1,1:-1])/(2*dy)
    dvy[0] = dvx[1]
    det_dT_gr = tensor(dvx[0]*dvy[1]- dvx[1]*dvy[0]).reshape((ny-2)*(nx-2))




def transport(x, T):
    #v_gr = stack([tensor(diag(v_fn_x(x[0], x[1]))).to(device), tensor(diag(v_fn_y(x[0], x[1]))).to(device)])
    #T_samp = func.vmap(T_fn)(x.mT) 
    n_samp = x.shape[0]
    T_samp = zeros(n_samp, 2).to(device)
    for i in arange(0, n_samp):
        T_samp[i] = T(x[i])
    return T_samp
        

def create_T_fn(v, x_gr):
    x_gr_1d = x_gr[0,1:-1]
    y_gr_1d = y_gr[1:-1,0]
    v_fn_x = i2d(x_gr_1d, y_gr_1d, v[0].numpy().T)
    v_fn_y = i2d(x_gr_1d, y_gr_1d, v[1].numpy().T)
    T_fn = lambda y: y + tensor([v_fn_x(y[0],y[1])[0,0], v_fn_y(y[0],y[1])[0,0]])
    return T_fn


#def create_dT_fn(v, x_gr, v_gr):
    

#def update_score(p_gr, T, x_gr, y_gr):
    


#def update_density(v, rho, x_gr, y_gr):
    

def plot_solutions(phi, x, y):
    nx, ny = x.shape
    color_map = 'jet' 
    color_levels = linspace(-1, 1, 30)
    fig, ax = subplots()
    
    phi_ana = sin(x)*sin(y)
    x, y, phi = x.numpy(), y.numpy(), phi.numpy() 
    phi_ana = phi_ana.numpy()
    pphi = ax.contourf(x, y, phi, vmin=-2, vmax=2, cmap='jet', levels=color_levels)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    cax = colorbar(pphi, cmap='jet') 
    cax.ax.tick_params(labelsize=24)
    cax.set_label("Comp. Soln", fontsize=20)
    ax.grid(True)
    ax.set_xlabel("1st coordinate",fontsize=20)
    ax.set_ylabel("2nd coordinate",fontsize=20)
    tight_layout()
    fig1, ax1 = subplots()

    pphi = ax1.contourf(x, y, phi_ana, cmap='jet', vmin=-2, vmax=2, levels=color_levels)
    ax1.xaxis.set_tick_params(labelsize=24)
    ax1.yaxis.set_tick_params(labelsize=24)
    ax1.grid(True)
    ax1.set_xlabel("1st coordinate",fontsize=20)
    ax1.set_ylabel("2nd coordinate",fontsize=20)
    ax1.axis("scaled")
    tight_layout()
    show()

def plot_samples(x):
    samples = x.mT.detach().cpu().numpy()
    fig, ax = subplots()
    color_levels = linspace(-1, 1, 30)
    rho = ax.hist2d(samples[0], samples[1], vmin=-2, vmax=2, density =True, cmap='jet')
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    cax = colorbar(cm.ScalarMappable(cmap='jet'),ax=gca()) 
    cax.ax.tick_params(labelsize=24)
    cax.set_label("Trans. Samples", fontsize=20)
    ax.grid(True)
    ax.set_xlabel("1st coordinate",fontsize=20)
    ax.set_ylabel("2nd coordinate",fontsize=20)
    tight_layout()
    
if __name__=="__main__":
    cuda.empty_cache()
    x_max, x_min, y_max, y_min = 2*pi,0.0,2*pi,0.0
    n_gr_x, n_gr_y = 6, 8
    phi, x_gr, y_gr = solve_pde(x_max, x_min, y_max, y_min, n_gr_x, n_gr_y)
    dx, dy = x_gr[0,1] - x_gr[0,0], y_gr[1,0] - y_gr[0,0]
    v = dphi(phi, dx, dy)
    n_samp = 1000
    samples = zeros(2,n_samp).to(device)
    samples[0] = x_min + (x_max-x_min)*rand(n_samp)
    samples[1] = y_min + (y_max-y_min)*rand(n_samp)
    samples = samples.mT
    T = create_T_fn(v, x_gr, y_gr)
    #samples =  transport(samples, T)
    # write score_update 
    #plot_solutions(phi, x_gr, y_gr)
    #plot_samples(samples)
