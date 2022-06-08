import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from scripts.data.SimClrDataset import SimClrDataset
from scripts.model.resnet_simclr import ResNetSimCLR
from scripts.model.losses import loss_simclr
import config as c
from scripts.data import data
from tqdm import tqdm
import numpy as np
import os

#Load Parameters
import yaml
params = yaml.safe_load(open('params.yaml'))

neptune_params = params['neptune']
train_default_params = params['train_simclr']

#At this point we hardcode the number of simclr views to be 2
N_VIEWS = 2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_simclr(params={},
                 save_model=True,
                 experiment_tracking=True):
    """
    Function to set up and train the resnet with simclr

    :param params: dictionary with training and model parameter; fields which are not defined are taken from params.yaml
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
                                                   tags=neptune_params["tags"] + ["simclr"])
            
        except:
            
            print("WARNING: Neptune init failed. Experiment tracking deactivated")
            experiment_tracking = False
    
    #Prepare the data 
    train_loader = data.get_train_loader(batch_size=params["BATCH_SIZE"], labels=False, transform=True, n_views=N_VIEWS, shuffle=True, drop_last=True) 
    val_loader = data.get_val_loader(batch_size=params["BATCH_SIZE"], labels=False, transform=True, n_views=N_VIEWS, shuffle=True, drop_last=True)

    #Load the model
    model = ResNetSimCLR(params)
    model.to(c.device)
    
    optimizer = torch.optim.Adam(model.parameters(), params["LEARNING_RATE"], weight_decay=params["WEIGHT_DECAY"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #Start the training
    with torch.cuda.device(c.device):
        
        #Use 16bit precision
        scaler = GradScaler(enabled=True)

        loss_memory = []

        for epoch_counter in range(params["NUM_EPOCHS"]):
            
            #Training
            for images in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(c.device)

                with autocast(enabled=True):
                    features = model(images)
                    loss, logits, labels = loss_simclr(features, N_VIEWS, params["BATCH_SIZE"])

                optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                
                if experiment_tracking:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    experiment.log_metric('training_loss', loss)
                    experiment.log_metric('training_acc/top1', top1[0])
                    experiment.log_metric('training_acc/top5', top5[0])
                    experiment.log_metric('learning_rate', scheduler.get_last_lr()[0])
                    
                
            #Validation
            val_loss = []
            val_logits, val_labels = None, None
            
            for images in tqdm(val_loader):
                images = torch.cat(images, dim=0)
                images = images.to(c.device)

                with autocast(enabled=True), torch.no_grad():
                    features = model(images)
                    loss, logits, labels = loss_simclr(features, N_VIEWS, params["BATCH_SIZE"])
                    val_loss.append(loss.cpu().detach().numpy())
                    
                    if val_logits is None or val_labels is None:
                        val_logits = logits
                        val_labels = labels
                    else:
                        val_logits = torch.cat((val_logits, logits), 0)
                        val_labels = torch.cat((val_labels, labels), 0)
                
            loss = np.mean(val_loss)
            top1, top5 = accuracy(val_logits, val_labels, topk=(1, 5))
                
            if experiment_tracking:
                experiment.log_metric('validation_loss', loss)
                experiment.log_metric('validation_acc/top1', top1[0])
                experiment.log_metric('validation_acc/top5', top5[0])
                
            #Early stopping
            loss_memory.append(loss)
        
            epoch_min_val =  np.argmin(loss_memory)
            if (epoch_counter + 1) > epoch_min_val + params["PATIENCE"]:
                break
            
            
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            

            # save model
            if save_model:

                if not os.path.exists(c.model_path):
                    os.makedirs(c.model_path)

                torch.save(model.state_dict(), c.resnet_path)
                
        
        return [loss, top1[0], top5[0]]
            


if __name__ == "__main__":
    train_simclr()