"""
"""
#from networks import Gen, Dis, DisShift
from networks import Gen, Dis
from utils import weights_init, get_model_list
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import copy

def update_average(model_tgt, model_src, beta=0.99):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

class HiSD(nn.Module):
    def __init__(self, hyperparameters):
        super(HiSD, self).__init__()
        self.gen = Gen(hyperparameters)
        self.dis = Dis(hyperparameters)

        self.hyperparameters = hyperparameters

    def forward(self, args, mode):
        if mode == 'gen':
            return self.gen_losses(*args)
        elif mode == 'dis':
            return self.dis_losses(*args)
        else:
            pass

    def gen_losses(self, x, i, j, j_trg):
        batch = x.size(0)
        # non-translation path
        e, s = self.gen.encode(x)
        x_rec = self.gen.decode(e, s)
        alpha = self.gen.extract(e, i) 

        # cycle-translation path
        ## translate
        alpha_sampled = torch.rand(batch, 1).cuda()
        alpha_trg = self.gen.map(alpha_sampled, i, j_trg)
        alpha_trg_diff = alpha_trg - alpha
        e_trg = self.gen.translate(e, i, alpha_trg_diff)
        x_trg = self.gen.decode(e_trg, s)
        ## cycle-back
        e_gen_rec, s_gen_rec = self.gen.encode(x_trg)
        alpha_trg_rec = self.gen.extract(e_gen_rec, i)
        alpha_trg_rec_diff = alpha - alpha_trg_rec # transform x_cycle_1 to x
        e_cyc = self.gen.translate(e_gen_rec, i, alpha_trg_rec_diff)
        x_cyc = self.gen.decode(e_cyc, s_gen_rec)
        
        # Added style in discriminator (ALI, Adversarially Learned Inference)
        # has not been added into the submission now, 
        # which helps disentangle the extracted style.
        # I will add this part in the next version.
        # To stable the training and avoid training crash, detaching is necessary.
        # Adding ALI will possibly make the metrics different from the paper,
        # but I do think this version would be better.
        
        loss_gen_adv = self.dis.calc_gen_loss_real(x, alpha, i, j) + \
                       self.dis.calc_gen_loss_fake_trg(x_trg, alpha_trg.detach(), i, j_trg) + \
                       self.dis.calc_gen_loss_fake_cyc(x_cyc, alpha.detach(), i, j)
       
        #loss_shift_cls, loss_shift_alpha = self.dis_2.calc_dis2_loss(x, x_trg, alpha_trg.detach(), i)

        loss_gen_sty = F.l1_loss(alpha_trg_rec, alpha_trg)

        loss_gen_rec = F.l1_loss(x_rec, x) + \
                       F.l1_loss(x_cyc, x)
        
        loss_disentanglement = self.gen.calc_disentanglement_loss(e, e_trg, i) + \
                self.gen.calc_disentanglement_loss(e_gen_rec, e_cyc, i)

        loss_sparsity = self.gen.calc_sparsity_loss()

        #loss_shift_dis = loss_shift_cls + loss_shift_alpha

        loss_gen_total = self.hyperparameters['adv_w'] * loss_gen_adv + \
                         self.hyperparameters['sty_w'] * loss_gen_sty + \
                         self.hyperparameters['rec_w'] * loss_gen_rec + \
                         self.hyperparameters['sparse_w'] * loss_sparsity + \
                         self.hyperparameters['dis_w'] * loss_disentanglement
        

        loss_gen_total.backward()

        return loss_gen_adv, loss_gen_sty, loss_gen_rec, loss_sparsity, loss_disentanglement, \
        x_trg.detach(), x_cyc.detach(), alpha.detach(), alpha_trg.detach()

    def dis_losses(self, x, x_trg, x_cyc, alpha, alpha_trg, i, j, j_trg):

        loss_dis_adv = self.dis.calc_dis_loss_real(x, alpha, i, j) + \
                       self.dis.calc_dis_loss_fake_trg(x_trg, alpha_trg, i, j_trg) + \
                       self.dis.calc_dis_loss_fake_cyc(x_cyc, alpha, i, j) 
        loss_dis_adv.backward()

        return loss_dis_adv

class HiSD_Trainer(nn.Module):
    def __init__(self, hyperparameters, multi_gpus=False):
        super(HiSD_Trainer, self).__init__()
        # Initiate the networks
        self.multi_gpus = multi_gpus
        self.models = HiSD(hyperparameters)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        
        
        self.dis_opt = torch.optim.Adam(self.models.dis.parameters(),
                                        lr=hyperparameters['lr_dis'], betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])
        
        self.gen_opt = torch.optim.Adam([{'params': self.models.gen.encoder.parameters()},
                                         {'params': self.models.gen.direction_matrix},
                                         #{'params': self.models.gen.masks.parameters()},
                                         {'params': self.models.gen.decoder.parameters()},
                                         {'params': self.models.gen.vectorization_unit.parameters()},
                                         # Different LR for mappers.
                                         {'params': self.models.gen.mappers.parameters(), 'lr': hyperparameters['lr_gen_mappers']},
                                        ],
                                        lr=hyperparameters['lr_gen_others'], betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])
        
        self.apply(weights_init(hyperparameters['init']))

        # For historical average version of the generators
        self.models.gen_test = copy.deepcopy(self.models.gen)

    def update(self, x, i, j, j_trg):

        this_model = self.models.module if self.multi_gpus else self.models

        # gen 
        for p in this_model.dis.parameters():
            p.requires_grad = False
        for p in this_model.gen.parameters():
            p.requires_grad = True

        self.gen_opt.zero_grad()
        #self.dis_2_opt.zero_grad()

        self.loss_gen_adv, self.loss_gen_sty, self.loss_gen_rec, self.loss_sparsity, self.loss_disentanglement, \
        x_trg, x_cyc, s, s_trg = self.models((x, i, j, j_trg), mode='gen')

        self.loss_gen_adv = self.loss_gen_adv.mean()
        self.loss_gen_sty = self.loss_gen_sty.mean()
        self.loss_gen_rec = self.loss_gen_rec.mean()
        self.loss_disentanglement = self.loss_disentanglement.mean()
        #self.loss_shift_cls = self.loss_shift_cls.mean()
        #self.loss_shift_alpha = self.loss_shift_alpha.mean()
        self.loss_sparsity = self.loss_sparsity.mean()

        
        nn.utils.clip_grad_norm_(this_model.gen.parameters(), 100)
        self.gen_opt.step()

        # New step
        #self.dis_2_opt.step()

        # dis
        for p in this_model.dis.parameters():
            p.requires_grad = True
        for p in this_model.gen.parameters():
            p.requires_grad = False
    
    
        self.dis_opt.zero_grad()

        self.loss_dis_adv = self.models((x, x_trg, x_cyc, s, s_trg, i, j, j_trg), mode='dis')
        self.loss_dis_adv = self.loss_dis_adv.mean()

        nn.utils.clip_grad_norm_(this_model.dis.parameters(), 100)
        self.dis_opt.step()

        update_average(this_model.gen_test, this_model.gen)

        return self.loss_gen_adv.item(), \
               self.loss_gen_sty.item(), \
               self.loss_gen_rec.item(), \
               self.loss_dis_adv.item()


    def sample(self, x, x_trg, j, j_trg, i):
        this_model = self.models.module if self.multi_gpus else self.models
        if True:
            gen = this_model.gen_test
        else:
            gen = this_model.gen

        out = [x]
        with torch.no_grad():

            e, s = gen.encode(x)
            x_alpha = gen.extract(e, i)

            # Latent-guided 1
            alpha = torch.rand(1,1).cuda().repeat(x.size(0), 1)
            alpha_trg = gen.map(alpha, i, j_trg) - x_alpha
            style_l_1 = gen.translate(e, i, alpha_trg)
            x_trg_ = gen.decode(style_l_1, s)
            out += [x_trg_]
            # Latent-guided 2
            alpha = torch.rand(1,1).cuda().repeat(x.size(0), 1)
            alpha_trg = gen.map(alpha, i, j_trg) - x_alpha
            style_l_2 = gen.translate(e, i, alpha_trg)
            x_trg_ = gen.decode(style_l_2, s)
            out += [x_trg_]
    
            e_trg, _ = gen.encode(x_trg)
            alpha_trg = gen.extract(e_trg, i) - x_alpha
            # Reference-guided 1: use x_trg[0, 1, ..., n] as reference
            x_trg_ = gen.decode(gen.translate(e, i, alpha_trg), s)
            out += [x_trg, x_trg_]
            # Reference-guided 2: use x_trg[n, ..., 1, 0] as reference
            x_trg_ = gen.decode(gen.translate(e, i, alpha_trg.flip([0])), s)
            out += [x_trg.flip([0]), x_trg_]

        return out

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.models.gen.load_state_dict(state_dict['gen'])
        self.models.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = os.path.join(checkpoint_dir, "dis_"+ last_model_name[-11:])
        state_dict = torch.load(last_model_name)
        self.models.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        print('Resume from iteration %d' % iterations)
        return iterations
    

    def save(self, snapshot_dir, iterations):
        this_model = self.models.module if self.multi_gpus else self.models
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': this_model.gen.state_dict(), 'gen_test': this_model.gen_test.state_dict()}, gen_name)
        torch.save({'dis': this_model.dis.state_dict()}, dis_name)
        torch.save({'dis': self.dis_opt.state_dict(),
                    'gen': self.gen_opt.state_dict()}, opt_name)
