from data_generation import *
from upscaling import *
from utils import *
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if __name__ == "__main__":

    os.environ['OMP_NUM_THREADS']='2'
    os.environ['LD_LIBRARY_PATH']=''
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    # Initialisation
    N_low = 20
    N_high = 100
    batch_size = 32
    training_size = 500

    GP_l = 0.1
    GP_sigma = 0.1
    
    epoch_num = 1000
    lr = 0.0005
    gamma = 0.01
    minimum_loss = float('inf')
    loss_track = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    w_low, r_low, A_low, x_low, y_low = generate_data(N_low)
    mean_u, covariance_u = u_low_prior(GP_l,GP_sigma,N_low)
    covariance = torch.tensor(covariance_u).to(device).to(torch.float32)

    # Load training data
    trainset = DataFromH5File("data/high_res_1000.h5","data/low_res_1000.h5")
    train_loader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    # Initialise training model
    G = UpScale()
    G.apply(weights_init_xavier).to(device)
    mse = nn.MSELoss(reduction='sum')
    optG = torch.optim.Adam(G.parameters(), lr = lr, weight_decay=0, betas=(0.5, 0.999))
    # optG = torch.optim.SGD(G.parameters(), lr = 0.0001)
    r_scheduleG = torch.optim.lr_scheduler.StepLR(optG, step_size=50, gamma=gamma)
    # r_scheduleG = torch.optim.lr_scheduler.ExponentialLR(optG, 0.98)
    
    # Logger info
    dir_name = f'models/pairs_training/{training_size}samples/lr{lr}_gamma{gamma}'
    makedir(dir_name)
    logger = setup_logging('job0', dir_name, console=True)
    logger.info(f'Training for {epoch_num} epoches and learning rate is {lr}')
    
    for epoch in range(1, epoch_num+1):
        
        for i, data in enumerate(train_loader, 0):
            
            lr, hr = data
            
            optG.zero_grad()
            sr = G(lr)
            loss = mse(sr,hr)/batch_size
            loss.backward()
            optG.step()
            
            if loss < minimum_loss:
                save_model(dir_name, epoch, 'best_model', r_scheduleG, G, optG)
                minimum_loss = loss
                
            if epoch%100 == 0:
                save_model(dir_name, epoch, 'model_epoch_{}'.format(epoch), r_scheduleG, G, optG)
                
            loss_track.append(loss.cpu().data.numpy())
            np.save(f'{dir_name}/chains/loss_curve.npy', np.array(loss_track))
            
            logger.info(f'Epoch {epoch}/{epoch_num}: Loss = {loss}')
            writer.add_scalar("Loss/train", loss, epoch)

            r_scheduleG.step()