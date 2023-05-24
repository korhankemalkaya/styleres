# python3.7
"""Contains the base class for Encoder (GAN Inversion) runner."""

import os
import shutil

import torch
import torch.distributed as dist
import torchvision.transforms as T
from utils.visualizer import postprocess_image

from utils.visualizer import postprocess_tensor

import numpy as np
from .base_runner import BaseRunner
from datasets import BaseDataset
from torch.utils.data import DataLoader
from PIL import Image
from runners.controllers.summary_writer import log_image
import torchvision
from editings.latent_editor import LatentEditor
from editings.styleclip.edit_hfgi import styleclip_edit, load_stylegan_generator,load_direction_calculator
from editings.GradCtrl.manipulate import main as gradctrl
import torch.nn.functional as F
import time
from fid import InceptionV3, frechet_distance
from core.data_loader import get_eval_loader
from tqdm import tqdm
from easydict import EasyDict

__all__ = ['BaseEncoderRunner']


class BaseEncoderRunner(BaseRunner):
    """Defines the base class for Encoder runner."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.inception_model = None

    def build_models(self):
        super().build_models()
        assert 'encoder' in self.models
        assert 'generator_smooth' in self.models
        assert 'discriminator' in self.models

        self.resolution = self.models['generator_smooth'].resolution
        self.G_kwargs_train = self.config.modules['generator_smooth'].get(
            'kwargs_train', dict())
        self.G_kwargs_val = self.config.modules['generator_smooth'].get(
            'kwargs_val', dict())
        self.D_kwargs_train = self.config.modules['discriminator'].get(
            'kwargs_train', dict())
        self.D_kwargs_val = self.config.modules['discriminator'].get(
            'kwargs_val', dict())
        if self.config.use_disc2:
            self.D2_kwargs_train = self.config.modules['discriminator2'].get(
                'kwargs_train', dict())
            self.D2_kwargs_val = self.config.modules['discriminator2'].get(
                'kwargs_val', dict())
        if self.config.mapping_method != 'pretrained':
            self.M_kwargs_train = self.config.modules['mapping'].get(
                'kwargs_train', dict())
            self.M_kwargs_val = self.config.modules['mapping'].get(
                'kwargs_val', dict())
        if self.config.create_mixing_network:
            self.MIX_kwargs_train = self.config.modules['mixer'].get(
                'kwargs_train', dict())
            self.MIX_kwargs_val = self.config.modules['mixer'].get(
                'kwargs_val', dict())

        

    def train_step(self, data, **train_kwargs):
        raise NotImplementedError('Should be implemented in derived class.')

    @torch.no_grad()
    def fid(self):
        """Computes the FID metric."""
        self.set_mode('val')
        dist.barrier()
        if self.rank != 0:
            return 0

        if self.inception_model is None:
            self.inception_model = InceptionV3().eval().to('cuda')
            self.logger.info(f'Finish building inception model.')

        # Log Images

        interface_options = EasyDict()
        interfacem_options = EasyDict()
        options = EasyDict()

        # Log real smiles and fake smiles
        interface_options.dataset = r"/media/hdd2/adundar/hamza/genforce/data/temp/smile_with_original"
        interface_options.output = os.path.join(self.work_dir, 'logs', 'smile_3')
        interface_options.edit = "smile"
        interface_options.factor = 3        
        interface_options.method = "interfacegan"
        paths_smiles = [interface_options.dataset, interface_options.output]

        # Log real non-smiles and fake non-smiles
        interfacem_options.dataset = r"/media/hdd2/adundar/hamza/genforce/data/temp/smile_without_original"
        interfacem_options.output = os.path.join(self.work_dir, 'logs', 'smile_minus_3')
        interfacem_options.edit = "smile"
        interfacem_options.factor = -3
        paths_non_smiles = [interfacem_options.dataset, interfacem_options.output]
        interfacem_options.method = "interfacegan"
        # Log real images and inverted images
        options.dataset = r"/media/hdd2/adundar/hamza/genforce/data/temp/images256"
        options.output = os.path.join(self.work_dir, 'logs', 'inversion')
        options.method = "inversion"
        paths_images = [options.dataset, options.output]

        #? inversion_opts['method'] = 'inversion' 
        self.save_edited_images(interface_options)
        self.save_edited_images(interfacem_options)
        self.save_edited_images(options)

        # FID
        paths = paths_smiles + paths_non_smiles + paths_images # Reals and fakes
        
        print("printing paths now:")

        print(paths)
        loaders = [get_eval_loader(path, self.resolution, self.val_batch_size) for path in paths]
        print("printing loaders now:")
        print(loaders)
        mu, cov = [], []
        all_activations = {}
        i = 0
        for loader in loaders:
            actvs = []
            for x in tqdm(loader, total=len(loader)):
                actv = self.inception_model(x.to('cuda'))
                actvs.append(actv)
            actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
            all_activations[i] = actvs
            mu.append(np.mean(actvs, axis=0))
            cov.append(np.cov(actvs, rowvar=False))
            i += 1
        fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])

        return fid_value
 
    

    def val(self, **val_kwargs):
        self.synthesize(**val_kwargs)

    def synthesize(self,
                   num,
                   html_name=None,
                   save_raw_synthesis=False):
        """Synthesizes images.

        Args:
            num: Number of images to synthesize.
            z: Latent codes used for generation. If not specified, this function
                will sample latent codes randomly. (default: None)
            html_name: Name of the output html page for visualization. If not
                specified, no visualization page will be saved. (default: None)
            save_raw_synthesis: Whether to save raw synthesis on the disk.
                (default: False)
        """

        dist.barrier()
        if self.rank != 0:
            return
            
        if not html_name and not save_raw_synthesis:
            return

        self.set_mode('val')

        if self.val_loader is None:
            self.build_dataset('val')

        # temp_dir = os.path.join(self.work_dir, 'synthesize_results')
        # os.makedirs(temp_dir, exist_ok=True)

        if not num:
            return
        # if num % self.val_batch_size != 0:
        #     num =  (num //self.val_batch_size +1)*self.val_batch_size
        # TODO: Use same z during the entire training process.
        
        self.logger.init_pbar()
        task = self.logger.add_pbar_task('Synthesis', total=num)
        for i in range(num):
            data = next(self.val_loader)
            for key in data:
                if key != 'name':
                    data[key] = data[key].cuda(
                        torch.cuda.current_device(), non_blocking=True)

            with torch.no_grad():
                real_images = data['image']
                return_dict = self.forward_pass(data, return_vals='all')
                fakes = return_dict['fakes']
                wp_mixed = return_dict['wp_mixed']
                eouts = return_dict['eouts']

                log_list_gpu = {"real": real_images, "fake": fakes}

                # Add editings to log_list
                editings = ['age', 'pose', 'smile']
                for edit in editings:
                    direction = torch.load(f'editings/interfacegan_directions/{edit}.pt').cuda()
                    factors = [+3, -3]
                    for factor in factors:
                        name = f"{edit}_{factor}"
                        edit_wp = wp_mixed + factor * direction
                        edited_images, _ = self.runG(edit_wp, "synthesis", highres_outs=eouts)
                        # if edit == 'smile' and factor == -3:
                        #     res = gouts['gates'].shape[-1]
                        #     log_list_gpu[f'smile_-3_gate'] = ( torch.mean((gouts_edits['gates']) , dim=1, keepdim=True), 0)
                        #edited_images = F.adaptive_avg_pool2d(edited_images, 256)
                        log_list_gpu[name] = edited_images
                        #log_list_gpu[f'{name}_gate'] = ( torch.mean((temp['gates']) , dim=1, keepdim=True), 0)

                #Add gate to log_list
                # res = gouts['gates'].shape[-1]
                # log_list_gpu[f'gate{res}x{res}'] = ( torch.mean((gouts['gates']) , dim=1, keepdim=True), 0)
            
                #Log images
                for log_name, log_val in log_list_gpu.items():
                    log_im = log_val[0] if type(log_val) is tuple else log_val
                    min_val = log_val[1] if type(log_val) is tuple else -1
                    cpu_img = postprocess_tensor(log_im.detach().cpu(), min_val=min_val)
                    grid = torchvision.utils.make_grid(cpu_img, nrow=5)
                    log_image( name = f"image/{log_name}", grid=grid, iter=self.iter)
            self.logger.update_pbar(task, 1)
        self.logger.close_pbar()

    def save_edited_images(self, opts):
        dist.barrier()
        if self.rank != 0:
            return
        self.set_mode('val')

        if opts.method  == 'inversion':
            pass
        elif opts.method == 'interfacegan':
            direction = torch.load(f'editings/interfacegan_directions/{opts.edit}.pt').cuda()
        elif opts.method == 'ganspace':
            ganspace_pca = torch.load('editings/ganspace_pca/ffhq_pca.pt')
            direction = {
                'eye_openness':            (54,  7,  8,  20),
                'smile':                   (46,  4,  5, -20),
                'beard':                   (58,  7,  9,  -20),
                'white_hair':              (57,  7, 10, -24),
                'lipstick':                (34, 10, 11,  20)
            }
            editor = LatentEditor()
        elif opts.method == 'styleclip':
            #model_path = '/media/hdd2/adundar/hamza/hyperstyle/pretrained_models/stylegan2-ffhq-config-f.pt'
            # calculator_args = {
		    # 'delta_i_c': 'editings/styleclip/global_directions/ffhq/fs3.npy',
		    # 's_statistics': 'editings/styleclip/global_directions/ffhq/S_mean_std',
		    # 'text_prompt_templates': 'editings/styleclip/global_directions/templates.txt'
	        #  }
            stylegan_model = load_stylegan_generator(opts.model_path)
            global_direction_calculator = load_direction_calculator(opts.calculator_args)
            #Eyeglasses 5, bangs 2, bobcut 5
            # edit_args = {'alpha_min': 2, 'alpha_max': 2, 'num_alphas':1, 'beta_min':0.11, 'beta_max':0.11, 'num_betas': 1,
            #     'neutral_text':'face', 'target_text': 'face with bangs'}
    
        
        self.config.data['val']['root_dir'] = opts.dataset

        dataset = BaseDataset(**self.config.data['val'])
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        temp_dir = os.path.join(self.work_dir, opts.output)
        os.makedirs(temp_dir, exist_ok=True)
 

        self.logger.init_pbar()
        task = self.logger.add_pbar_task('Synthesis', total=len(val_loader))
        global_i = 0
        all_latents = {}
        for idx, data in enumerate(val_loader):
            for key in data:
                if key != 'name':
                    data[key] = data[key].cuda(
                        torch.cuda.current_device(), non_blocking=True)
            with torch.no_grad():
                return_dict = self.forward_pass(data, return_vals='all', only_enc=True)
                wp_mixed = return_dict['wp_mixed']
                eouts = return_dict['eouts']
                #fakes = return_dict['fakes']
                factors = np.linspace(0, 3, 100)
                global_i = 0

                # for factor in factors:
                if opts.method == 'interfacegan':
                    wp_mixed = wp_mixed + opts.factor * direction
                if opts.method == 'ganspace':
                    #interpolate_dir = direction[opts.edit][0:3] + (factor,) 
                    wp_mixed = editor.apply_ganspace_(wp_mixed, ganspace_pca, [direction[opts.edit]])
                    #wp_edit = editor.apply_ganspace_(wp_mixed, ganspace_pca, [interpolate_dir])


                # z = torch.randn((1,self.config.latent_dim), device=torch.cuda.current_device())
                # z = self.runM(z)
                # diff = z - wp_mixed
                # edit = (diff * 3.5) / 10
                # wp_mixed = wp_mixed + edit
                edited_images, gouts_edits = self.runG(wp_mixed, "synthesis", highres_outs=eouts, resize=False)
                if opts.method == 'styleclip':
                    # opts.edit_args['alpha_min'] = factor
                    edited_images = styleclip_edit(wp_mixed, gouts_edits['additions'], stylegan_model, global_direction_calculator, opts.edit_args)
                edited_images = T.Resize((256,256))(edited_images)
                edited_images = postprocess_image(edited_images.detach().cpu().numpy())
                for j in range(edited_images.shape[0]):
                    # dir_name = data['name'][j][:-4]
                    # os.makedirs(os.path.join(temp_dir, dir_name), exist_ok=True)
                    # save_name = f'{global_i:03d}_' + data['name'][j]
                    save_name = data['name'][j]
                    pil_img = Image.fromarray(edited_images[j]) #.resize((256,256))
                    #pil_img.save(os.path.join(temp_dir, dir_name,  save_name ))
                    pil_img.save(os.path.join(temp_dir,  save_name ))
                    global_i += 1
                # if global_i >= 1000:
                #     break
                # if global_i % 100 == 0:
                #     print(f"{global_i}/1000")

            self.logger.update_pbar(task, 1)
        self.logger.close_pbar()
    


    def interface_generate(self, num, edit, factor):
        direction = torch.load(f'editings/interfacegan_directions/{edit}.pt').cuda()
        indices = list(range(self.rank, num, self.world_size))
        gt_path = os.path.join(self.work_dir, f'interfacegan_gt')
        smile_add_path = os.path.join(self.work_dir, f'interfacegan_{edit}_{factor}')
        smile_rm_path = os.path.join(self.work_dir, f'interfacegan_{edit}_-{factor}')
        if self.rank == 0:
            os.makedirs(gt_path, exist_ok=True)
            os.makedirs(smile_add_path, exist_ok=True)
            os.makedirs(smile_rm_path, exist_ok=True)
        dist.barrier()

        self.logger.init_pbar()
        task = self.logger.add_pbar_task('Interfacegan', total=num)


        for batch_idx in range(0, len(indices), self.val_batch_size):
            sub_indices = indices[batch_idx:batch_idx + self.val_batch_size]
            batch_size = len(sub_indices)

            z = torch.randn((batch_size,512), device=torch.cuda.current_device())
            w_r = self.runM(z, repeat_w=True)
            gt_imgs,_ = self.runG(w_r, resize=False)
            gt_imgs = postprocess_image(gt_imgs.detach().cpu().numpy())
            for i in range(gt_imgs.shape[0]):
                save_name = str(sub_indices[i]) + ".png"
                pil_img = Image.fromarray(gt_imgs[i]).resize((256,256))
                pil_img.save(os.path.join(gt_path, save_name ))
            
            smile_added, _ = self.runG(w_r + factor*direction, resize=False)
            smile_added = postprocess_image(smile_added.detach().cpu().numpy())
            for i in range(gt_imgs.shape[0]):
                save_name = str(sub_indices[i]) + ".png"
                pil_img = Image.fromarray(smile_added[i]).resize((256,256))
                pil_img.save(os.path.join(smile_add_path, save_name ))

            smile_removed, _= self.runG(w_r - factor*direction, resize=False)
            smile_removed = postprocess_image(smile_removed.detach().cpu().numpy())
            for i in range(gt_imgs.shape[0]):
                save_name = str(sub_indices[i]) + ".png"
                pil_img = Image.fromarray(smile_removed[i]).resize((256,256))
                pil_img.save(os.path.join(smile_rm_path, save_name ))

            self.logger.update_pbar(task, batch_size * self.world_size)
        self.logger.close_pbar()

    def grad_edit(self, edit, factor, dataset=None):
        dist.barrier()
        if self.rank != 0:
            return
        self.set_mode('val')
        edit_name = edit
        edit = 'val'
        self.config.data[edit]['root_dir'] = dataset

        dataset = BaseDataset(**self.config.data[edit])
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        temp_dir = os.path.join(self.work_dir, f'fakes_{edit_name}_{factor}')
        os.makedirs(temp_dir, exist_ok=True)

        self.logger.init_pbar()
        task = self.logger.add_pbar_task('Synthesis', total=len(val_loader))
        global_i = 0
        args = {'model': 'ffhq', 'model_dir': '/media/hdd2/adundar/hamza/genforce/editings/GradCtrl/model_ffhq',
                'attribute': edit_name, 'exclude': 'default', 'top_channels': 'default', 'layerwise': 'default' }
        for idx, data in enumerate(val_loader):
            for key in data:
                if key != 'name':
                    data[key] = data[key].cuda(
                        torch.cuda.current_device(), non_blocking=True)
            with torch.no_grad():
                return_dict = self.forward_pass(data, return_vals='all', only_enc=True)
                wp_mixed = return_dict['wp_mixed']
                eouts = return_dict['eouts']

            #fakes = return_dict['fakes']
            edit_wp = gradctrl(args, wp_mixed, factor)
            edited_images, gouts_edits = self.runG(edit_wp, "synthesis", highres_outs=eouts, resize=False)
            #edited_images, gouts_edits = self.runG(wp_mixed, "synthesis", highres_outs=eouts, resize=False)
            edited_images = postprocess_image(edited_images.detach().cpu().numpy())
            for j in range(edited_images.shape[0]):
                save_name = data['name'][j]
                pil_img = Image.fromarray(edited_images[j]).resize((256,256))
                pil_img.save(os.path.join(temp_dir, save_name ))
                global_i += 1

            self.logger.update_pbar(task, 1)
        self.logger.close_pbar()

    def measure_time(self, edit, factor, dataset=None, save_latents=False):
            dist.barrier()
            if self.rank != 0:
                return
            self.set_mode('val')
            edit_name = edit
            if dataset is not None:
                edit = 'val'
                self.config.data[edit]['root_dir'] = dataset

            dataset = BaseDataset(**self.config.data[edit])
            val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            global_i = 0
            time_list = []
            for idx, data in enumerate(val_loader):
                for key in data:
                    if key != 'name':
                        data[key] = data[key].cuda(
                            torch.cuda.current_device(), non_blocking=True)
                with torch.no_grad():
                    start = time.time()
                    return_dict = self.forward_pass(data, return_vals='all', only_enc=True)
                    wp_mixed = return_dict['wp_mixed']
                    eouts = return_dict['eouts']
                    edited_images, gouts_edits = self.runG(wp_mixed, "synthesis", highres_outs=eouts, resize=False)
                    end = time.time()
                    time_list.append(end-start)
                    print(np.mean(time_list))
            print(np.mean(time_list[1:]))

            