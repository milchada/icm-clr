import torch

from scripts.model.resnet_simclr import ResNetSimCLR
from scripts.model.losses import loss_simclr, loss_nnclr, loss_adaption, lambd_simclr_train, lambd_simclr_domain, lambd_simclr_adaption
from scripts.model.optimizer import Optimizer
from scripts.model.training import Trainer, run_training
from scripts.model.experiment_tracking import NeptuneExperimentTracking, VoidExperimentTracking
from scripts.model.batch_queue import init_batch_queue
from scripts.util.logging import logger

import config as c

from scripts.data import data
from scripts.data.augmentations import SimCLRAugmentation, FlipAugmentation

#Load Parameters
import yaml
params = yaml.safe_load(open('params.yaml'))
train_default_params = params['train_clr']

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

def images_2_device(x):
    x = torch.cat(x, dim=0)
    return x #x.to(c.device)

def loss_2_host(x):
    return x.cpu().detach().numpy()

def loss_dict_2_host(d):
    return dict(map(lambda x: (x[0], loss_2_host(x[1])), d.items()))
    
def train_clr(params={},
              save_model=True,
              experiment_tracking=True):
    """
    Function to set up and train the resnet with clr

    :param params: dictionary with training and model parameter; fields which are not defined are taken from params.yaml
    :param save_model: if true metrics and model are saved; otherwise only the validation losses are returned (for parameter optimization)
    :param experiment_tracking: switch for experiment tracking by neptune.ai (has to be set up in params.yaml)
    
    """ 

    #Use default parameters if not set
    params = dict(train_default_params, **params)
    
    #Init experiment tracking
    if experiment_tracking:
        experiment_tracker = NeptuneExperimentTracking(params, tags=['clr'])
    else: 
        experiment_tracker = VoidExperimentTracking()
   
    #PrepareDataPara
    class DataParallelWrapper(torch.nn.DataParallel):
        @property
        def trainable_parameters(self):
            return self.module.trainable_parameters

    #Load the model
    logger.info('Load model...')
    model = ResNetSimCLR(params) 
        
    model.to(c.device)

    if params["PARALLEL_TRAINING"]:
        model = DataParallelWrapper(module=model)
    logger.info('Model loaded.')

    #Init the optimizer
    optimizer = Optimizer(model, params)
    
    #Set the save path of the model
    if save_model:
        save_path = c.resnet_path
    else:
        save_path = None

    #Init the trainer
    trainer = Trainer(model,
                      optimizer,
                      experiment_tracker,
                      params["PATIENCE"],
                      params["NUM_EPOCHS"],
                      save_path,
                      max_num_batches = params["MAX_NUM_BATCHES_PER_EPOCH"],
                      max_runtime_seconds = params["MAX_RUNTIME_SECONDS"],
                      use_checkpoint=False)
    
    #Prepare training data 
    augmentation = SimCLRAugmentation(params["AUGMENTATION_PARAMS"])
    flip_augmentation = FlipAugmentation(params["AUGMENTATION_PARAMS"])
    train_loader = data.get_train_loader(batch_size=params["BATCH_SIZE"], labels=False, augmentation=augmentation, n_views=N_VIEWS, shuffle=True, drop_last=True)
    training_data = [train_loader]
    
    if params['DOMAIN_ADAPTION']:
        domain_loader = data.get_domain_loader(batch_size=params["BATCH_SIZE"], labels=False, augmentation=augmentation, n_views=N_VIEWS, shuffle=True, drop_last=True)
        train_mmd_loader = data.get_train_loader(batch_size=params["BATCH_SIZE"], labels=False, augmentation=flip_augmentation, n_views=1, shuffle=True, drop_last=True)
        domain_mmd_loader = data.get_domain_loader(batch_size=params["BATCH_SIZE"], labels=False, augmentation=flip_augmentation, n_views=1, shuffle=True, drop_last=True)
        
        training_data += [domain_loader, train_mmd_loader, domain_mmd_loader]
    
    #Prepare validation data
    val_loader = data.get_val_loader(batch_size=params["BATCH_SIZE"], labels=False, augmentation=augmentation, n_views=N_VIEWS, shuffle=True, drop_last=True)
    validation_data = [val_loader]
    
    #Prepare lossfunctions
    if params['CLR_TYPE'] == 'SIMCLR':
        training_loss = lambda img, rep, model: loss_simclr(rep, N_VIEWS, params["BATCH_SIZE"])
        validation_loss = lambda img, rep, model: loss_simclr(rep, N_VIEWS, params["BATCH_SIZE"])
        domain_loss = lambda img, rep, model: loss_simclr(rep, N_VIEWS, params["BATCH_SIZE"])
    
    elif params['CLR_TYPE'] == 'NNCLR':
        logger.info('Prepare NNCLR queue...')
        train_batch_queue = init_batch_queue(model, train_loader, params["NNCLR_QUEUE_SIZE"])
        val_batch_queue = init_batch_queue(model, val_loader, params["NNCLR_QUEUE_SIZE"])
        
        training_loss = lambda img, rep, model: loss_nnclr(img, rep, model, N_VIEWS, train_batch_queue) 
        validation_loss = lambda img, rep, model: loss_nnclr(img, rep, model, N_VIEWS, val_batch_queue)
        
        if params['DOMAIN_ADAPTION']:
            domain_batch_queue = init_batch_queue(model, domain_loader, params["NNCLR_QUEUE_SIZE"])
            domain_loss = lambda img, rep, model: loss_nnclr(img, rep, model, N_VIEWS, domain_batch_queue)
            
        logger.info('NNCLR queue prepared.')
            
    else:
        raise ValueError("Invalid CLR_TYPE given!")
        
    
    #Prepare Training Lossfunction
    def training_lossfunction(model, batch):
        
        train_images = images_2_device(batch[0])
        
        if params['DOMAIN_ADAPTION']:
            domain_images = images_2_device(batch[1])
            train_adaption_images = images_2_device(batch[2])
            domain_adaption_images = images_2_device(batch[3])
        
        #Loss for the training set
        features = model(train_images)
        train_loss, logits, labels = training_loss(train_images, features, model)
        train_top1, train_top5 = accuracy(logits, labels, topk=(1, 5))

        #Loss for the domain set
        if params['DOMAIN_ADAPTION']:
            features = model(domain_images)
            domain_loss, logits, labels = domain_loss(domain_images, features, model)
            domain_top1, domain_top5 = accuracy(logits, labels, topk=(1, 5))

        #Loss for the training - domain representation distance
        if params['DOMAIN_ADAPTION']:
            train_rep = model(train_adaption_images, projection_head=False)
            domain_rep = model(domain_adaption_images, projection_head=False)
            adaption_loss = loss_adaption(train_rep, domain_rep)

        #Calculate total loss
        loss = lambd_simclr_train * train_loss

        loss_dict = {'training_loss': train_loss,
                     'training_acc/top1': train_top1[0],
                     'training_acc/top5': train_top5[0]}
        
        if params['DOMAIN_ADAPTION']:
            loss += lambd_simclr_domain * domain_loss + lambd_adaption * adaption_loss
            
            loss_dict = {**loss_dict,
                         'domain_loss': domain_loss,
                         'domain_acc/top1': domain_top1[0],
                         'domain_acc/top5': domain_top5[0],
                         'adaption_loss': adaption_loss}
            
        loss_dict = {**loss_dict,
                     'total_loss': loss}

        loss_dict = loss_dict_2_host(loss_dict)
        
        return loss, loss_dict

    #Prepare Validation Lossfunction
    def validation_lossfunction(model, batch):
        val_images = images_2_device(batch[0])

        #Loss for validation set
        features = model(val_images)
        val_loss, logits, labels = validation_loss(val_images, features, model)
        val_top1, val_top5 = accuracy(logits, labels, topk=(1, 5))

        loss_dict = {'validation_loss': val_loss,
                     'validation_acc/top1': val_top1[0],
                     'validation_acc/top5': val_top5[0]}

        loss_dict = loss_dict_2_host(loss_dict)
        val_loss = loss_2_host(val_loss)

        return val_loss, loss_dict

    #Perform training
    return run_training(trainer, training_lossfunction, training_data, validation_lossfunction, validation_data)


if __name__ == "__main__":
    train_clr()
