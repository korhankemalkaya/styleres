class EvalForwarder():
    def forward(self, runner, data, only_enc=False):
        encoder_type = runner.config.encoder_type
        forward_func = getattr(self, f'{encoder_type}_forward')
        return forward_func(runner, data,only_enc)

    def base_forward(self, runner, only_enc, wp_mixed, eouts):
        eouts['inversion'] = runner.runG(wp_mixed, 'synthesis', highres_outs=None, return_f=True)
        if only_enc:
            return_dict = {'wp_mixed':wp_mixed,'eouts': eouts}
            return return_dict
        fakes, gouts = runner.runG(wp_mixed, 'synthesis', highres_outs=eouts)
        return_dict = {'fakes': fakes, 'wp_mixed':wp_mixed, 'gouts':gouts, 'eouts': eouts}
        return return_dict

    def e4e_forward(self, runner, data, only_enc):
        reals = data['image']
        E = runner.models['encoder']
        wp_mixed, eouts = E(reals)
        return self.base_forward( runner, only_enc, wp_mixed, eouts)

    def pSp_forward(self, runner, data, only_enc):
        reals = data['image']
        E = runner.models['encoder']
        wp_mixed, eouts = E(reals)
        return self.base_forward( runner, only_enc, wp_mixed, eouts)
    
    def hyperstyle_forward(self, runner, data, only_enc):
        reals = data['image']
        E = runner.models['encoder']
        wp_mixed, eouts = E(reals)
        return self.base_forward( runner, only_enc, wp_mixed, eouts)

    def styletransformer_forward(self, runner, data, only_enc):
        reals = data['image']
        E = runner.models['encoder']
        z = E.basic_encoder.z
        n, c = z.shape[1], z.shape[2]
        b = reals.shape[0]
        z = z.expand(b, n, c).flatten(0, 1)
        query = runner.runM(z).reshape(b, n, c)
        wp_mixed, eouts = E(reals, query)
        return self.base_forward( runner, only_enc, wp_mixed, eouts)