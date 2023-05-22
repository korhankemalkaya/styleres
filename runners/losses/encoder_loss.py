# python3.7
"""Defines loss functions for encoder training."""
import torch
import torch.nn.functional as F
import torch.distributed as dist

from models import build_perceptual, build_arcface

__all__ = ['EncoderLoss']


class EncoderLoss(object):
    """Contains the class to compute logistic GAN loss."""

    def __init__(self,
                 runner,
                 d_loss_kwargs=None,
                 e_loss_kwargs=None,
                 perceptual_kwargs=None,
                 ratio_loss_kwargs=None,
                 ld_loss_kwargs=None, 
                 d2_loss_kwargs=None,
                 same_style_loss_kwargs=None,
                 hfgi_loss_kwargs =None,
                 id_loss_kwargs= None):
        """Initializes with models and arguments for computing losses."""
        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.e_loss_kwargs = e_loss_kwargs or dict()
        self.ld_loss_kwargs = ld_loss_kwargs or dict()
        self.d2_loss_kwargs = d2_loss_kwargs or dict()
        self.id_loss_kwargs = id_loss_kwargs or dict()
        self.same_style_loss = same_style_loss_kwargs or dict()
        self.hfgi_loss_kwargs = hfgi_loss_kwargs or dict()

        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        self.r2_gamma = self.d_loss_kwargs.get('r2_gamma', 0.0)
        self.r1_gamma2 = self.d2_loss_kwargs.get('r1_gamma', 0.0)
        self.r2_gamma2 = self.d2_loss_kwargs.get('r2_gamma', 0.0)
        self.perceptual_lw = self.e_loss_kwargs.get('perceptual_lw', 5e-5)
        self.adv_lw = self.e_loss_kwargs.get('adv_lw', 0.08)
        self.pixel_lw = self.e_loss_kwargs.get('pixel_lw', 1.0)
        self.ratio_lw = self.e_loss_kwargs.get('ratio_lw', 0.1)
        self.latent_adv_lw = self.e_loss_kwargs.get('latent_adv_lw', 0.0)
        self.adv2_lw = self.e_loss_kwargs.get('adv2_lw', 0.0)
        self.meanw_lw = self.e_loss_kwargs.get('meanw_lw', 0.05)
        self.cycle_lw = self.e_loss_kwargs.get('cycle_lw', 0.1)
        self.diver_image_lw = self.e_loss_kwargs.get('diver_image_lw', 0.0)
        self.diver_latent_lw = self.e_loss_kwargs.get('diver_latent_lw', 0.0)
        self.use_same_style_loss = self.same_style_loss.get('use', False)
        self.same_style_pixel_lw = self.same_style_loss.get('pixel_lw', 0.0)
        self.consistency_lw = self.same_style_loss.get('consistency_lw', 0.0)
        self.use_ada_loss = self.hfgi_loss_kwargs.get('use', False)
        self.ada_lw = self.hfgi_loss_kwargs.get('ada_lw', 0.0)
        self.f_lw = self.e_loss_kwargs.get('f_lw', 1.0)
        self.align_lw = self.e_loss_kwargs.get('align_lw', 0.1)
        self.id_lw = self.e_loss_kwargs.get('id_lw', 0.1)
        self.g_lw = self.e_loss_kwargs.get('g_lw', 0.1)
        self.id_loss_edits = self.id_loss_kwargs.get('id_loss_edits', False)

        runner.running_stats.add(f'id_loss', log_format='.3f', log_strategy='AVERAGE')
        #runner.running_stats.add(f'recons_fr', log_format='.3f', log_strategy='AVERAGE')

        runner.running_stats.add(f'f_regularizer', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(f'g_regularizer', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(f'align_loss', log_format='.3f', log_strategy='AVERAGE')


        #runner.running_stats.add(f'recons_fw', log_format='.3f', log_strategy='AVERAGE')

        self.arcface = build_arcface(path='/media/hdd2/adundar/hamza/genforce/checkpoints/arcface.pth').cuda()
        self.arcface.eval()
        #self.k = torch.log(ratio_loss_kwargs.k).cuda()
        for param in self.arcface.parameters():
            param.requires_grad = False

        if self.adv2_lw == 0:
            self.r1_gamma2 = 0.0
            self.r2_gamma2 = 0.0

        if (self.ratio_lw > 0):
            self.arcface = build_arcface(path=ratio_loss_kwargs.path).cuda()
            self.arcface.eval()
            #self.k = torch.log(ratio_loss_kwargs.k).cuda()
            for param in self.arcface.parameters():
                param.requires_grad = False
        
        runner.space_of_latent = runner.config.space_of_latent

        if self.pixel_lw != 0 or self.perceptual_lw != 0:
            runner.running_stats.add(
                f'recon_loss', log_format='.3f', log_strategy='AVERAGE')
            runner.running_stats.add(
                f'vgg_loss', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'adv_loss', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'loss_fake', log_format='.3f', log_strategy='AVERAGE')
        # runner.running_stats.add(
        #     f'loss_fake_edit', log_format='.3f', log_strategy='AVERAGE')
        # runner.running_stats.add(
        # f'adv_edit_loss', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'loss_real', log_format='.3f', log_strategy='AVERAGE')
        
        if (self.ratio_lw != 0):
            runner.running_stats.add(
                f'ratio_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.r1_gamma != 0:
            runner.running_stats.add(
                f'real_grad_penalty', log_format='.3f', log_strategy='AVERAGE')
        if self.r2_gamma != 0:
            runner.running_stats.add(
                f'fake_grad_penalty', log_format='.3f', log_strategy='AVERAGE')
        if self.r1_gamma2 != 0:
            runner.running_stats.add(
                f'd2_real_penalty', log_format='.3f', log_strategy='AVERAGE')
        if self.r2_gamma2 != 0:
            runner.running_stats.add(
                f'd2_fake_penalty', log_format='.3f', log_strategy='AVERAGE')
        if self.latent_adv_lw != 0:
            runner.running_stats.add(
                    f'loss_ld_fake', log_format='.3f', log_strategy='AVERAGE')
            runner.running_stats.add(
                    f'loss_ld_real', log_format='.3f', log_strategy='AVERAGE')
            runner.running_stats.add(
                    f'adv_latent_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.adv2_lw != 0:
            runner.running_stats.add(
                    f'loss_d2_fake', log_format='.3f', log_strategy='AVERAGE')
            runner.running_stats.add(
                    f'loss_d2_real', log_format='.3f', log_strategy='AVERAGE')
            runner.running_stats.add(
                    f'adv2_loss', log_format='.3f', log_strategy='AVERAGE')

        if self.meanw_lw != 0:
             runner.running_stats.add(
                f'meanw_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.cycle_lw != 0:
            runner.running_stats.add(f'cycle_loss_pix', log_format='.3f', log_strategy='AVERAGE')
            runner.running_stats.add(f'cycle_loss_vgg', log_format='.3f', log_strategy='AVERAGE')
            runner.running_stats.add(f'cycle_loss_id', log_format='.3f', log_strategy='AVERAGE')
            
        if self.diver_image_lw != 0:
            runner.running_stats.add(
                f'diver_image_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.diver_latent_lw != 0:
            runner.running_stats.add(
                f'diver_latent_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.use_same_style_loss:
            if self.same_style_pixel_lw != 0:
                runner.running_stats.add(
                f'same_style_pixel_loss', log_format='.3f', log_strategy='AVERAGE')
            if self.consistency_lw != 0:
                runner.running_stats.add(
                f'consistency_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.use_ada_loss:
            if self.ada_lw > 0:
                runner.running_stats.add(f'ada_loss',log_format='.3f', log_strategy='AVERAGE' )
    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        image_grad = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=images,
            create_graph=True,
            retain_graph=True)[0].view(images.shape[0], -1)
        penalty = image_grad.pow(2).sum(dim=1).mean()
        return penalty

    def d_loss(self, runner, data):
        """Computes loss for discriminator."""
        D = runner.models['discriminator']

        reals = data['image']
        # z_rand = data['z_rand']
        with torch.no_grad():
            return_dict = runner.train_forward(data)
            fakes = return_dict['fakes']
            #gouts = return_dict[ 'gouts']
            # eouts = return_dict[ 'eouts']
            # wp_mixed = return_dict[ 'wp_mixed']

            # edit_wp = wp_mixed + (data['factor'] * data['direction']).unsqueeze(1)
            # edit_img, _ = runner.runG(edit_wp, 'synthesis', highres_outs=eouts)
            # edit_img = F.adaptive_avg_pool2d(edit_img, 256)
            #final_out = reals*valids + fakes*(1.0-valids) #Final output is the mixed one.
        reals.requires_grad = True #To calculate grad penalty

        real_scores = D(reals, **runner.D_kwargs_train)
        fake_scores = D(fakes, **runner.D_kwargs_train)
        #fake_scores_edit = D(edit_img, **runner.D_kwargs_train) / 2
        loss_fake = F.softplus(fake_scores).mean()
        loss_real = F.softplus(-real_scores).mean()
        #loss_fake_edit = F.softplus(fake_scores_edit).mean()
        d_loss = loss_fake + loss_real #+ loss_fake_edit

        runner.running_stats.update({'loss_fake': loss_fake.item()})
        runner.running_stats.update({'loss_real': loss_real.item()})
        #runner.running_stats.update({'loss_fake_edit': loss_fake_edit.item()})


        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        if self.r1_gamma:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
            runner.running_stats.update(
                {'real_grad_penalty': real_grad_penalty.item()})
        if self.r2_gamma:
            fake_grad_penalty = self.compute_grad_penalty(
                fakes, fake_scores)
            runner.running_stats.update(
                {'fake_grad_penalty': fake_grad_penalty.item()})
        return (d_loss +
                real_grad_penalty * (self.r1_gamma * 0.5) +
                fake_grad_penalty * (self.r2_gamma * 0.5))

    def ld_loss(self, runner, data):
        """Computes loss for discriminator."""
        if 'generator_smooth' in runner.models:
            G = runner.get_module(runner.models['generator_smooth'])
        else:
            G = runner.get_module(runner.models['generator'])
        G.eval()
        E = runner.models['encoder']
        LD = runner.models['latent_disc']
        reals = data['image']
        valids = data['valid']
        with torch.no_grad():
            z_rand = torch.randn((reals.shape[0]* 14, 512), device=reals.device)
            w_rand = runner.runG(G, z_rand, 'getw').view(reals.shape[0],14, -1)
            fake_latents, wp_enc, blender = E(reals, valids, w_rand, mix_space = 'w')
            real_latents = w_rand.view(z_rand.size())

        ld_fake = LD(fake_latents) #Decrease
        ld_real = LD(real_latents) #Increase
        loss_ld_fake = F.softplus(ld_fake).mean()
        loss_ld_real = F.softplus(-ld_real).mean()
        ld_loss = loss_ld_fake + loss_ld_real
        
        runner.running_stats.update({'loss_ld_fake': loss_ld_fake.item()})
        runner.running_stats.update({'loss_ld_real': loss_ld_real.item()})
        
        return ld_loss, wp_enc, blender

    def d2_loss(self, runner, data):
        """Computes loss for fake discriminator."""
        D2 = runner.models['discriminator2']

        reals = data['image']
        valids = data['valid']
        z_rand = data['z_rand']
        reals.requires_grad = True #To calculate grad penalty
        with torch.no_grad():
            wp_rand = runner.runM(z_rand)

            #Use the same latents and blender that is used in previous Discriminators, if possible
            try:
                wp_enc = data['wp_enc']
                blender = data['blender']
            except KeyError:
                wp_enc, blender = runner.runE(reals, valids, wp_rand)
            wp_mixed = runner.mix(wp_enc, wp_rand, blender)
            fakes = runner.runG(wp_mixed, 'synthesis')

        real_scores = D2(reals, **runner.D2_kwargs_train)
        fake_scores = D2(fakes, **runner.D2_kwargs_train)
        loss_fake = F.softplus(fake_scores).mean()
        loss_real = F.softplus(-real_scores).mean()
        d2_loss = loss_fake + loss_real

        real_grad_penalty = torch.zeros_like(d2_loss)
        fake_grad_penalty = torch.zeros_like(d2_loss)
        if self.r1_gamma2:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
            runner.running_stats.update(
                {'d2_real_penalty': real_grad_penalty.item()})
        if self.r2_gamma2:
            fake_grad_penalty = self.compute_grad_penalty(
                fakes, fake_scores)
            runner.running_stats.update(
                {'d2_fake_penalty': fake_grad_penalty.item()})

        return_val = (d2_loss +
                real_grad_penalty * (self.r1_gamma2 * 0.5) +
                fake_grad_penalty * (self.r2_gamma2 * 0.5))

        runner.running_stats.update({'loss_d2_real': loss_real.item()})
        runner.running_stats.update({'loss_d2_fake': loss_fake.item()})

        return return_val, wp_enc, blender

    def e_loss(self, runner, data):
        """Computes loss for generator."""
        # if 'generator_smooth' in runner.models:
        #     G = runner.get_module(runner.models['generator_smooth'])
        # else:
        #     G = runner.get_module(runner.models['generator'])
        # G.eval()
        D = runner.models['discriminator']
        #E = runner.models['encoder']
        if (self.latent_adv_lw != 0):
            LD = runner.models['latent_disc']
        if (self.adv2_lw != 0):
            D2 = runner.models['discriminator2']
        if (self.perceptual_lw != 0):
            P = runner.perceptual_model
        #if self.ratio_lw != 0:
        ARC = self.arcface

        # Fetch data
        reals = data['image']
        reals.requires_grad = False
        #valids = data['valid']

        if self.cycle_lw > 0:
            iscycle = True
        else: 
            iscycle = False
        return_dict = runner.train_forward(data, iscycle=iscycle)
        #wp_enc = return_dict['wp_enc']
        fakes = return_dict['fakes']
        gouts = return_dict[ 'gouts']
        #eouts = return_dict[ 'eouts']

        #fakes2 = return_dict['fakes2']
        #Pixel + vgg loss
        loss_pix = 0
        # factor_idx = (data['factor'] == 0).view(-1)
        # direction_idx =  torch.sum(data['direction'], dim=1) == 0
        # slice_idx = factor_idx | direction_idx
        slice_idx = (data['factor'] == 0).view(-1)
        isempty = True if (slice_idx == False).all() else False
        if self.pixel_lw != 0:
            if not isempty:
                loss_pix = F.mse_loss(fakes[slice_idx], reals[slice_idx], reduction='mean')
            
            # all_pix =  torch.mean((fakes - reals)**2, dim=(1,2,3))  # F.mse_loss(fakes, reals, reduction='mean') #self.pixel_lw 
            
            # for idx in range(all_pix.shape[0]):
            #     is_normal = slice_idx[idx].item()
            #     if is_normal == True:
            #         loss_pix += all_pix[idx] * self.pixel_lw 
            #     else: 
            #         loss_pix += all_pix[idx] * (self.pixel_lw /ratio)

        loss_feat = 0
        if self.perceptual_lw != 0:
            if not isempty:
                loss_feat = self.perceptual_lw * F.mse_loss(
                    P(fakes[slice_idx]), P(reals[slice_idx]), reduction='mean')

            # all_feats =  torch.mean((P(fakes) - P(reals))**2, dim=1)

            # for idx in range(all_pix.shape[0]):
            #     is_normal = slice_idx[idx].item()
            #     if is_normal == True:
            #         loss_feat += all_feats[idx] * self.perceptual_lw 
            #     else: 
            #         loss_feat += all_feats[idx] * (self.perceptual_lw  /ratio)
        loss_rec = loss_pix + loss_feat

        
                
        fake_scores = D(fakes, **runner.D_kwargs_train)
        adv_loss = self.adv_lw * F.softplus(-fake_scores).mean()

        # if not isempty:
        #     adv_loss = self.adv_lw * F.softplus(-fake_scores[slice_idx]).mean()
        # else:
        #     adv_loss = self.adv_lw * 10 * F.softplus(-fake_scores).mean()
        # fake_scores = D(fakes, **runner.D_kwargs_train)
        # fake_scores = F.softplus(-fake_scores).view(-1)
        # adv_loss = 0
        # for idx in range(all_pix.shape[0]):
        #     is_normal = slice_idx[idx].item()
        #     if is_normal == True:
        #         adv_loss += fake_scores[idx] *  self.adv_lw
        #     else: 
        #         adv_loss += fake_scores[idx] * ( self.adv_lw)

        id_loss = 0
        if self.id_lw > 0:
            if self.id_loss_edits:
                cosine_sim = ARC(reals, fakes)
                id_loss =  (1-cosine_sim)
                id_loss = self.id_lw * id_loss.mean()
            else:
                if not isempty:
                    cosine_sim = ARC(reals[slice_idx], fakes[slice_idx])
                    id_loss =  (1-cosine_sim)
                    id_loss = self.id_lw * id_loss.mean()

        # cosine_sim = ARC(reals, fakes)
        # id_loss_all =  (1-cosine_sim)
        # id_loss = 0
        # for idx in range(all_pix.shape[0]):
        #     is_normal = slice_idx[idx].item()
        #     if is_normal == True:
        #         id_loss += id_loss_all[idx] * self.id_lw 
        #     else: 
        #         id_loss += id_loss_all[idx] * (self.id_lw /ratio)
        

        # skip_penalty = 0
        # for i in range (1):
        #     gate =  gouts['gates']
        #     n,c,h,w =gate.shape[0], gate.shape[1], gate.shape[2], gate.shape[3]
        #     #skip_penalty = skip_penalty + torch.sum(highres_outs['gates'][i], dim=(1,2,3))
        #     skip_penalty = skip_penalty + torch.sum(torch.abs(gouts['gates'])) / (n*c*w*h)
        # skip_penalty =  skip_penalty * 0.2

        f_regularizer = 0
        if self.f_lw != 0:
            f_regularizer = self.f_lw * torch.mean(torch.abs(gouts['additions']))
            
        g_regularizer = 0
        if self.g_lw != 0:
            g_regularizer = self.g_lw * torch.mean(gouts['gates'])
        
        cycle_loss = 0
        if self.cycle_lw > 0:
            #cycle_pix= self.pixel_lw * F.mse_loss(reals, return_dict['cycle'], reduction='mean')
            cycle_vgg = self.perceptual_lw * F.mse_loss(
                    P(return_dict['cycle']), P(reals), reduction='mean')
            cycle_id = ARC(reals, return_dict['cycle'])
            cycle_id =  (1-cycle_id)
            cycle_id =  cycle_id.mean()
            cycle_loss = 0.1 * (cycle_vgg + cycle_id) #+ cycle_vgg + cycle_id)

        align_loss = 0
        if self.align_lw != 0:
            #align_loss = self.align_lw * gouts['aligned_loss']
            slice_inv = torch.logical_not(slice_idx)
            is_empty_inv = True if (slice_inv == False).all() else False
            if not is_empty_inv:
                align_loss = -1 * self.align_lw * F.mse_loss(reals[slice_inv], fakes[slice_inv])

        #Full Loss
        # e_loss = loss_pix + loss_feat + adv_loss + ratio_loss + ld_loss + meanw_loss 
        # + cycle_loss - diversification_loss + adv2_loss + ada_loss
        e_loss = loss_rec  + id_loss + f_regularizer + g_regularizer  + adv_loss + align_loss + cycle_loss #+ adv_edit_loss # + meanw_loss + adv_loss skip_penalty

        if not isempty:
            runner.running_stats.update({'id_loss': id_loss.item()})
        if self.g_lw != 0:
            runner.running_stats.update({'g_regularizer': g_regularizer.item()})
        if self.f_lw != 0:
            runner.running_stats.update({'f_regularizer': f_regularizer.item()})
        if self.align_lw != 0:
            if not is_empty_inv:
                runner.running_stats.update({'align_loss': align_loss.item()})
        if self.cycle_lw != 0:
            #runner.running_stats.update({'cycle_loss_pix': cycle_pix.item()})
            runner.running_stats.update({'cycle_loss_vgg': cycle_vgg.item()})
            runner.running_stats.update({'cycle_loss_id': cycle_id.item()})


        #runner.running_stats.update({'recons_fr': loss_pix_fr.item()})
        #runner.running_stats.update({'recons_fw': loss_pix_fw.item()})

        #runner.running_stats.update({'wp_recons': wp_recons.item()})

        #Statistics
        if (self.pixel_lw != 0 or self.perceptual_lw != 0):
            if not isempty:
                runner.running_stats.update({'recon_loss': loss_pix.item()})
                runner.running_stats.update({'vgg_loss': loss_feat.item()})

        runner.running_stats.update({'adv_loss': adv_loss.item()})
        #runner.running_stats.update({'adv_edit_loss': adv_edit_loss.item()})
        # if (self.meanw_lw != 0):
        #     runner.running_stats.update({'meanw_loss': meanw_loss.item()})
    
        return e_loss
