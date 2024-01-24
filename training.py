from data_generation import *
from upscaling import *
from utils import *
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def sample_p_data():
    """
    Randomly sample a batch size number of training samples
    
    Outputs
    ------------
    Random samples: torch.Tensor
    """
    return data[torch.LongTensor(batch_size).random_(0,500)].detach()

def sample_p_0(batch_size):
    """
    Initialisation of the langevin dynamics to sample from posterior of u_l
    
    Args
    ----------
    batch_size: int
    """
    # prior = torch.randn(*[batch_size,1,20,20]).to(device)
    mean = w_low.reshape(1,1,20,20)
    b = np.repeat(mean,batch_size,axis=0)
    prior = torch.tensor(b).to(device).to(torch.float32)
    prior = prior + torch.rand_like(prior)
    
    return prior

def sqrt_matrix(M):
    """
    Work out the square root of a matrix
    
    Args
    ------------
    M: torch.Tensor
        A positive definite matrix
    """
    b = sqrtm(M.cpu().data.numpy())
    
    return b

def calculate_log_likelihood(x_hat,x,batch_size):
    """
    Calculate the logarithm of the likelihood distribution
    
    Args
    -------------
    x_hat: torch.Tensor
        Mean of the likelihood distribution
    x: torch.Tensor
        True data
    batch_size: int
    """
    ll = -1/(2*math.pow(ll_sigma, 2)) * torch.matmul((x-x_hat).reshape(batch_size,1,N_high**2),(x-x_hat).reshape(batch_size,N_high**2,1))
    
    return ll.sum()
       
    
def ula_posterior(z, x, G):
    """
    Langevin dynamics to sample from posterior of u_l
    
    Args
    -------------
    z: torch.tensor
        Latent variables
    x: torch.tensor
        True data
    G: nn.Module
        Upscaling network
    """
    z = z.clone().detach().requires_grad_(True)
    chains_evolution = []
    for i in range(K):
        x_hat = G(z)
        g_log_likelihood = calculate_log_likelihood(x_hat,x,batch_size)
        grad_g = torch.autograd.grad(g_log_likelihood, z)[0]
        grad_log_likelihood = s**2 / (N_low**2) * torch.matmul(covariance,grad_g.reshape(batch_size,N_low**2,1)).reshape(batch_size,1,N_low,N_low)
        
        w_low_tensor = torch.tensor(w_low).to(device).to(torch.float32)
        difference = z.reshape(batch_size,1,N_low**2) - w_low_tensor.reshape(1,N_low**2)
        grad_log_prior = - s**2 * difference.reshape(batch_size,1,N_low,N_low)
        
        W = torch.randn(*[batch_size,1,N_low,N_low]).to(device)
        random = torch.sqrt(torch.tensor(2.)) * s * torch.matmul(sqrt_covariance,W.reshape(batch_size,N_low**2,1)).reshape(batch_size,1,N_low,N_low)
        
        z = z + grad_log_prior + grad_log_likelihood + random
        chains_evolution.append(z.cpu().data.numpy())   
           
    return z.detach(), chains_evolution


if __name__ == "__main__":

    os.environ['OMP_NUM_THREADS']='2'
    os.environ['LD_LIBRARY_PATH']=''
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    # Initialisation
    N_low = 20
    N_high = 100
    batch_size = 32

    # Epoch number and step size of Langevin dynamics
    K = 200
    s = 0.05

    GP_l = 0.1
    GP_sigma = 0.1
    ll_sigma = 0.1
    epoch_num = 1000
    minimum_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    w_low, r_low, A_low, x_low, y_low = generate_data(N_low)
    mean_u, covariance_u = u_low_prior(GP_l,GP_sigma,N_low)
    covariance = torch.tensor(covariance_u).to(device).to(torch.float32)
    positive_covariance = covariance + 1e-5 * torch.eye(400).to(device)
    sqrt_covariance = torch.tensor(sqrt_matrix(positive_covariance)).to(device).to(torch.float32)

    # Load training data
    with h5py.File('high_res_500.h5', 'r') as hf:
        data = hf['high_res_GP'][:]
    data = torch.tensor(data.reshape(500,1,N_high,N_high))
    data = data.to(torch.float32).to(device)
    dataloader = torch.utils.data.DataLoader(dataset=data,batch_size=batch_size,shuffle=True)

    G = UpScale()
    G.apply(weights_init_xavier).to(device)
    mse = nn.MSELoss(reduction='sum')
    optG = torch.optim.Adam(G.parameters(), lr = 0.0001, weight_decay=0, betas=(0.5, 0.999))
    # optG = torch.optim.SGD(G.parameters(), lr = 0.0001)
    r_scheduleG = torch.optim.lr_scheduler.StepLR(optG, step_size=60, gamma=0.5)
    
    for epoch in range(1, epoch_num+1):
        
        x = sample_p_data()
        posterior_initial = sample_p_0(batch_size)
        posterior_final, posterior_chain = ula_posterior(posterior_initial, x, G)

        optG.zero_grad()
        x_hat = G(posterior_final.detach())
        loss_g = -calculate_log_likelihood(x_hat,x,batch_size)/batch_size
        # loss_g = mse(x_hat,x)/batch_size
        loss_g.backward()
        optG.step()
        
        r_scheduleG.step()
        
        writer.add_scalar("Loss/train", loss_g, epoch)
        if loss_g < minimum_loss:
            torch.save(G.state_dict(), 'best_G.pth')
            minimum_loss = loss_g
            
        print("Epoch: {}".format(epoch), "Loss: {}".format(loss_g.item()))

   
    '''for epoch in range(1, epoch_num+1):
        for batch, x in enumerate(dataloader,0):
            batch_size = x.shape[0]
            posterior_initial = sample_p_0(batch_size)
            posterior_final, posterior_chain = ula_posterior(posterior_initial, x, G)

            optG.zero_grad()
            x_hat = G(posterior_final.detach())
            # loss_g = -calculate_log_likelihood(x_hat,x,batch_size)/batch_size
            loss_g = mse(x_hat,x)/batch_size
            loss_g.backward()
            optG.step()
            
            writer.add_scalar("Loss/train", loss_g)
            if loss_g < minimum_loss:
                torch.save(G.state_dict(), 'best_G.pth')
                minimum_loss = loss_g
            
            print("Epoch: {}".format(epoch), "Batch: {}".format(batch), "Loss: {}".format(loss_g.item()))'''
            
    