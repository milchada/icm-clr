import torch
from torch.cuda.amp import GradScaler

class Optimizer():
    """Container class for the optimizer, scheduler and scaler"""
    
    def __init__(self, model, params={}):
        
        self.model = model
        self.params = params
        
        self.update()
    
    def update(self, keep_lr=False):
        """Reinitialize the optimizer; use this after changing the optimizer parameters or the trainable parameters of the model"""
        
        self.scaler = GradScaler(enabled=True)
        
        if not keep_lr:
            self.optimizer = torch.optim.Adam(self.model.trainable_parameters,
                                              lr=self.params["LEARNING_RATE"],
                                              weight_decay=self.params["L2_DECAY"],
                                              betas=(self.params["BETA_1"], self.params["BETA_2"]))
        else:
            self.optimizer = torch.optim.Adam(self.model.trainable_parameters,
                                              lr=self.lr,
                                              weight_decay=self.params["L2_DECAY"],
                                              betas=(self.params["BETA_1"], self.params["BETA_2"]))
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=self.params["LR_DECAY"], 
                                                                    patience=self.params["LR_PATIENCE"])
        
    def backward(self, loss):
        """Perform backward propagation using the grad scaler"""
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
    def lr_step(self, loss):
        """Perform a scheduler learning rate step"""
        self.scheduler.step(loss)
        
    @property
    def lr(self):
        """Return the current learning rate"""
        return self.optimizer.param_groups[0]['lr']