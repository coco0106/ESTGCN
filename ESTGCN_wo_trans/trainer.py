import torch.optim as optim
import math
from net import *
import util


class Trainer():
    def __init__(self,
                 model,
                 lrate,
                 wdecay,
                 clip,
                 step_size,
                 seq_out_len,
                 scaler,
                 device,
                 cl=False,
                 new_training_method=False):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lrate,
                                    weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size

        self.iter = 0
        self.task_level = 12
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.new_training_method = new_training_method
        self.exponentials = util.exponentialDecay(100)

    def train(self, input, real_val, ycl, idx=None, batches_seen=None,epoch=None):
        self.iter += 1
        if self.iter % self.step == 0 and self.task_level < self.seq_out_len:
            self.task_level += 1
            if self.new_training_method:
                self.iter = 0
        
        self.model.train()
        self.optimizer.zero_grad()
        if self.cl:
            
            output, average_time,grad_mask,baselines,log_pi,wait_penalty = self.model(input,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.task_level,epsilon=self.exponentials[epoch-1])
        else:
            output, average_time,grad_mask,baselines,log_pi,wait_penalty = self.model(input,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.task_level,epsilon=self.exponentials[epoch-1])
       
        real = real_val[:,:,-1]
        predict = self.scaler.inverse_transform(output)
        
        if self.cl:

            loss = self.loss(predict[:, :],
                             real[:, :], 0.0)
            mape = util.masked_mape(predict[:, :],
                                    real[:, :],
                                    0.0).item()
            rmse = util.masked_rmse(predict[:, :],
                                    real[:, :],
                                    0.0).item()
        else:
            loss = self.loss(predict, real, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
        
        r = (2*(self.scaler.transform(predict).float().round() == self.scaler.transform(real).round()).float()-1)
        R = r.float() * grad_mask.float()
        b = grad_mask.float() * baselines.float()
        adjusted_reward = R - b
        loss_b = self.loss(b, R, 0.0) 
        loss_r = (-log_pi*adjusted_reward).sum(0).mean()
        lam = torch.tensor([0], dtype=torch.float, requires_grad=False).cuda(0)
        loss_all = loss_r + loss_b+ loss + lam*wait_penalty
        
        loss_all.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        return loss.item(), mape, rmse,average_time

    def eval(self, input, real_val, ycl):
        self.model.eval()
        with torch.no_grad():
            output, average_time,grad_mask,baselines,log_pi,wait_penalty = self.model(input, ycl=ycl,epsilon=0.0)
        
        real = real_val[:,:,-1]
        predict = self.scaler.inverse_transform(output)
        
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse, average_time