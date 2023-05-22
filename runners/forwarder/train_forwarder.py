import torch

class TrainForwarder():
    def forward(self, runner, data, iscycle=False):
        encoder_type = runner.config.encoder_type
        forward_func = getattr(self, f'{encoder_type}_train_forward')
        return forward_func( runner, data, iscycle)

    def styleres_train_forward(self, runner, data, iscycle, wp, eouts, query=None):
        direction = data['direction']
        edit_name = data['edit_name']
        factor = data['factor']
        E = runner.models['encoder']
        edit = torch.zeros_like(wp)
        for i in range (edit.shape[0]):
            if edit_name[i] is None:
                edit[i] = 0
            elif edit_name[i] == 'randw':
                diff = direction[i] - wp[i]
                # one_hot = [1] * 8 + [0] * 10
                # one_hot = torch.tensor(one_hot, device=diff.device).unsqueeze(1)
                # diff = diff * one_hot   
                #norm = torch.linalg.norm(diff, dim=1, keepdim=True)
                edit[i] = (diff * factor[i]) / 10
            elif edit_name[i] == 'interface':
                edit[i] = (factor[i] * direction[i])

        # # Debug
        # with torch.no_grad():
        #     fakes,_ =runner.runG(wp, 'synthesis', highres_outs=None)
        #     fakes = postprocess_image(fakes.detach().cpu().numpy())
        #     for i in range(fakes.shape[0]):
        #         pil_img = Image.fromarray(fakes[i]).resize((256,256))
        #         pil_img.save(f'{runner.iter}_orig.png')

        #     fakes,_ =runner.runG(wp+edit, 'synthesis', highres_outs=None)
        #     fakes = postprocess_image(fakes.detach().cpu().numpy())
        #     for i in range(fakes.shape[0]):
        #         pil_img = Image.fromarray(fakes[i]).resize((256,256))
        #         pil_img.save(f'{runner.iter}_edit.png')

        #     fakes,_ =runner.runG(direction.unsqueeze(1).repeat(1,18,1), 'synthesis', highres_outs=None)
        #     fakes = postprocess_image(fakes.detach().cpu().numpy())
        #     for i in range(fakes.shape[0]):
        #         pil_img = Image.fromarray(fakes[i]).resize((256,256))
        #         pil_img.save(f'{runner.iter}_rand.png')

        with torch.no_grad():
            eouts['inversion'] = runner.runG(wp, 'synthesis', highres_outs=None, return_f=True)
        wp = wp + edit
        fakes, gouts = runner.runG(wp, 'synthesis', highres_outs=eouts)
        #fakes = F.adaptive_avg_pool2d(fakes, (256,256))
        fakes_cycle = None
        if iscycle:
            # wp_cycle = wp_cycle + runner.meanw.repeat(reals.shape[0], 1, 1)
            with torch.no_grad():
                if query != None:
                    wp_cycle, eout_cycle = E(fakes, query)
                else:
                    wp_cycle, eout_cycle = E(fakes)
                eout_cycle['inversion'] = runner.runG(wp_cycle, 'synthesis', highres_outs=None, return_f=True)

            #wp_cycle = wp_cycle - edit
            wp_cycle = wp_cycle - edit
            #wp_cycle = wp_cycle - (data['factor'] * data['direction']).unsqueeze(1)
            fakes_cycle, _ = runner.runG(wp_cycle, 'synthesis', highres_outs=eout_cycle)
            #fakes_cycle = F.adaptive_avg_pool2d(fakes, (256,256))
            #cycle = F.mse_loss(fakes_cycle, reals, reduction='mean')
        return_dict = {'fakes': fakes, 'wp_mixed':wp, 'gouts':gouts, 'eouts': eouts, 'cycle': fakes_cycle}
        return return_dict

    def base_train_forward(self, runner, data, iscycle):
        reals = data['image']
        E = runner.models['encoder']
        with torch.no_grad():
            wp, eouts = E(reals)
        return self.styleres_train_forward(runner, data, iscycle, wp, eouts)
        
    def e4e_train_forward(self, runner, data, iscycle):
        return self.base_train_forward(runner, data, iscycle)

    def pSp_train_forward(self, runner, data, iscycle):
        return self.base_train_forward(runner, data, iscycle)

    def hyperstyle_train_forward(self, runner, data, iscycle):
        return self.base_train_forward(runner, data, iscycle)

    def styletransformer_train_forward(self, runner, data, iscycle):
        reals = data['image']
        E = runner.models['encoder']
        with torch.no_grad():
            z = E.basic_encoder.z
            n, c = z.shape[1], z.shape[2]
            b = reals.shape[0]
            z = z.expand(b, n, c).flatten(0, 1)
            query = runner.runM(z).reshape(b, n, c)
            wp, eouts = E(reals, query)
        return self.styleres_train_forward(runner, data, iscycle, wp, eouts, query)
        
