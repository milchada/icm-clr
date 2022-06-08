import torch
import torch.nn
import torch.optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from . import combined_model
from . import losses
from scripts.model.optimizer import Optimizer
import scripts.data.data as data

import config as c

import os

#Load Parameters
import yaml
params = yaml.safe_load(open('params.yaml'))

neptune_params = params['neptune']
train_default_params = params['train_cinn']



def train_cinn(params={},
               model_index=0,
               save_model=True,
               experiment_tracking=True):
    """
    Function to set up and train the cINN

    :param params: dictionary with training and model parameter; fields which are not defined are taken from params.yaml
    :param model_index: define the index of the model for ensemble averaging
    :param save_model: if true metrics and model are saved; otherwise only the validation losses are returned (for parameter optimization)
    :param experiment_tracking: switch for experiment tracking by neptune.ai (has to be set up in params.yaml)
    
    """ 
    

    #Use default parameters if not set
    params = dict(train_default_params, **params)
    
    #Init neptune tracking
    if experiment_tracking:
        
        try:

            import neptune
            project = neptune.init(project_qualified_name=neptune_params["project_name"],
                                   api_token=neptune_params["api"])
            
            #Use name of branch as experiment name
            from git import Repo
            experiment_name = Repo('./').active_branch.name
            
            #Create neptune experiment and save all parameters in the parameter file
            from pandas.io.json._normalize import nested_to_record
            experiment = project.create_experiment(name=experiment_name, 
                                                   params=nested_to_record(params),
                                                   tags=neptune_params["tags"] + ["cinn"])
            
        except:
            
            print("WARNING: Neptune init failed. Experiment tracking deactivated")
            experiment_tracking = False
            
    
    #Load and prepare Model
    cinn = combined_model.CombinedModel(params)
    
    if params["FIX_RESNET_PARAMS"]:
        cinn.fix_resnet_weights()
    
    if params["USE_PRETRAINED_RESNET"]:
        cinn.load_pretrained_resnet(c.resnet_path)
        
    cinn.to(c.device)
    
    #Init the optimizer
    optimizer = Optimizer(cinn, params)
    
    #Prepare the data 
    train_loader = data.get_train_loader(batch_size=params["BATCH_SIZE"], labels=True, transform=True, n_views=1, shuffle=True, drop_last=True) 
    val_loader = data.get_val_loader(batch_size=params["BATCH_SIZE"], labels=True, transform=False, n_views=1, shuffle=True, drop_last=True)
    
    #Training/Evaluation Loop
    #-----------------------------------------------------------------------------
    
    with torch.cuda.device(c.device):
        
        #Init the early stopping
        nll_val_memory = []

        for epoch in tqdm(range(params["NUM_EPOCHS"]), total=params["NUM_EPOCHS"], disable=False):

            #Lists to log all the (validation) losses for the current epoch    
            nll_mean = []
            fmmd_mean = []
            bmmd_mean = []
            mse_mean = []
            mae_mean = []

            nll_val = []
            fmmd_val = []
            bmmd_val = []
            mse_val = []
            mae_val = []
            z_val = []
            
            #Perform Training
            cinn.train()

            for i, (image, label) in enumerate(train_loader):
                image = torch.cat(image, dim=0)
                image = image.to(c.device)
                label = label.to(c.device)

                #Augmentation
                label += params["NOISE"] * torch.randn_like(label)
                
                with autocast(enabled=True):
                    
                    #Forward Pass
                    z, log_j = cinn(label, image)  

                    #Init loss list
                    loss = []

                    #Negative log likelihood
                    if losses.lambd_max_likelihood != 0:
                        nll = losses.loss_max_likelihood(z, log_j)
                        nll_mean.append(nll.item())
                        loss.append(nll * losses.lambd_max_likelihood)

                    #Forward MMD
                    #Ensure that z is multivariate Gaussian
                    if losses.lambd_mmd_forw != 0:
                        fmmd = losses.loss_forward_mmd(z)
                        fmmd_mean.append(fmmd.item())
                        loss.append(fmmd * losses.lambd_mmd_forw)

                    #Backward MMD
                    #Ensure that x prior distribution is reproduced
                    if losses.lambd_mmd_back != 0:
                        x_samples = cinn.reverse_sample(torch.randn_like(z), image)
                        bmmd = losses.loss_backward_mmd(label, x_samples)
                        bmmd_mean.append(bmmd.item())
                        loss.append(bmmd * losses.lambd_mmd_back)

                    #MSE l -> x
                    #Just as for the MLP model
                    if losses.lambd_mse != 0:
                        x_zero = cinn.reverse_sample(torch.zeros_like(z), image)
                        mse = losses.loss_mse(label, x_zero)
                        mse_mean.append(mse.item()) 
                        loss.append(mse * losses.lambd_mse)

                    #MAE l -> x
                    #Just as for the MLP model
                    if losses.lambd_mae != 0:
                        x_zero = cinn.reverse_sample(torch.zeros_like(z), image)
                        mae = losses.loss_mae(label, x_zero)
                        mae_mean.append(mae.item()) 
                        loss.append(mae * losses.lambd_mae)

                    #Use all losses (!=0) to perform the error propagation
                    l_total = sum(loss)

                    #Clip Gradient to prevent too large gradients 
                    torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 1.)

                    #Perform optimizer step using the grad scaler
                    optimizer.backward(l_total)

            #Perform Validation
            cinn.eval()
            
            for i, (image, label) in enumerate(val_loader):

                with torch.no_grad(), autocast(enabled=True):
                    image = torch.cat(image, dim=0)
                    image = image.to(c.device)
                    label = label.to(c.device)

                    z, log_j = cinn(label, image)  

                    #NSegative log likelihood
                    nll_val.append(losses.loss_max_likelihood(z, log_j).item())

                    #Forward MMD
                    fmmd_val.append(losses.loss_forward_mmd(z).item())

                    #Backward MMD
                    x_samples = cinn.reverse_sample(torch.randn_like(z), image)
                    bmmd_val.append(losses.loss_backward_mmd(label, x_samples).item())

                    #MSE and MAE
                    x_zero = cinn.reverse_sample(torch.zeros_like(z), image)
                    mse_val.append(losses.loss_mse(label, x_zero).item())
                    mae_val.append(losses.loss_mae(label, x_zero).item())

                    #Z
                    z_val.append(z.cpu().numpy())


            #Log training losses if they are used abd if tracking is activated
            if experiment_tracking:

                if len(nll_mean) != 0:
                    experiment.log_metric('training_loss', np.mean(nll_mean))
                if len(fmmd_mean) != 0:
                    experiment.log_metric('training_fmmd_loss', np.mean(fmmd_mean))
                if len(bmmd_mean) != 0:
                    experiment.log_metric('training_bmmd_loss', np.mean(bmmd_mean))
                if len(mse_mean) != 0:
                    experiment.log_metric('training_mse_loss', np.mean(mse_mean))
                if len(mae_mean) != 0:
                    experiment.log_metric('training_mae_loss', np.mean(mae_mean))

                #Log validation losses
                experiment.log_metric('validation_loss', np.mean(nll_val))
                experiment.log_metric('validation_fmmd_loss', np.mean(fmmd_val))
                experiment.log_metric('validation_bmmd_loss', np.mean(bmmd_val))
                experiment.log_metric('validation_mse_loss', np.mean(mse_val))
                experiment.log_metric('validation_mae_loss', np.mean(mae_val))

                z_val = np.concatenate(z_val)
                experiment.log_metric('validation_z_std', np.max(np.std(z_val, axis=0)))
                experiment.log_metric('validation_z_mean', np.max(np.abs(np.mean(z_val, axis=0))))
                
                experiment.log_metric('learning_rate', optimizer.lr)

            #Check if Early Stopping Condition is reached
            nll_val_memory.append(np.mean(nll_val))
            
            epoch_min_val =  np.argmin(nll_val_memory)
            if (epoch + 1) > epoch_min_val + params["PATIENCE"]:
                print("Early Stopping")
                break
                    
            #Update the learning rate
            optimizer.lr_step(np.mean(nll_val))
                    
            #Check if RESNET params should be unfixed
            if params["FIX_RESNET_PARAMS"] and optimizer.lr < params["RESNET_LR_THRESHOLD"]:
                params["FIX_RESNET_PARAMS"] = False
                cinn.unfix_resnet_weights()
                optimizer.update(keep_lr=True)
                print("Unfix RESNET")
    
    #Save model
    if save_model:

        if not os.path.exists(c.model_path):
            os.makedirs(c.model_path)

        torch.save(cinn.state_dict(), c.model_path + 'cinn_' + str(model_index) + '.pt')

        #Save final validation metrics
        metrics={"validation_loss": float(np.mean(nll_val)),
                 "validation_fmmd_loss": float(np.mean(fmmd_val)),
                 "validation_bmmd_loss": float(np.mean(bmmd_val)),
                 "validation_mse_loss": float(np.mean(mse_val)),
                 "validation_mae_loss": float(np.mean(mae_val)),
                 "validation_z_std": float(np.max(np.std(z_val, axis=0))),
                 "validation_z_mean": float(np.max(np.abs(np.mean(z_val, axis=0))))}    

        #Store metrics if model_index==0
        if model_index == 0:
            import json
            
            if not os.path.exists(c.metrics_path):
                os.makedirs(c.metrics_path)
            
            with open(c.metrics_path + "train.json", "w") as f:
                      json.dump(metrics, f)
    
    #End Experiment
    if experiment_tracking:
        experiment.stop()
        
    #Return the validation loss for the hyper-parameter-optimization    
    return [np.mean(nll_val), np.mean(fmmd_val), np.mean(bmmd_val), np.max(np.abs(np.std(z_val, axis=0)-1)), np.max(np.abs(np.mean(z_val, axis=0)))]


if __name__ == "__main__":
    
    NUM_MODELS = train_default_params["NUM_MODELS"]
    
    for i in range(NUM_MODELS):
        train_cinn(model_index=i)
        