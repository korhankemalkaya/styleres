
# Inversion: images256
# Smile+: smile_with_original
# Smile-: smile_without_original
dataset = '/media/hdd2/adundar/hamza/genforce/data/temp/images256'  #Input Dataset

#inversion
#interfacegan
method = 'inversion'
output = 'inversion'     #Output Directory

if method == 'interfacegan':
    factor = -3
    edit = 'smile'

elif method == 'ganspace':
    edit = 'eye_openness'

elif method == 'styleclip':
    model_path = '/media/hdd2/adundar/hamza/hyperstyle/pretrained_models/stylegan2-ffhq-config-f.pt'
    calculator_args = {
		    'delta_i_c': 'editings/styleclip/global_directions/ffhq/fs3.npy',
		    's_statistics': 'editings/styleclip/global_directions/ffhq/S_mean_std',
		    'text_prompt_templates': 'editings/styleclip/global_directions/templates.txt'
	         }
    # edit_args = {'alpha_min': 15, 'alpha_max': 1, 'num_alphas':1, 'beta_min':0.5, 'beta_max':0.8, 'num_betas': 1,
    #             'neutral_text':'face with hair', 'target_text': 'face with purple hair'}

    # Bangs
    edit_args = {'alpha_min': 1, 'alpha_max': 1, 'num_alphas':1, 'beta_min':0.11, 'beta_max':0.11, 'num_betas': 1,
                'neutral_text':'a face', 'target_text': 'a face with bangs'}

    #Glasses
    # edit_args = {'alpha_min': 5, 'alpha_max': 5, 'num_alphas':1, 'beta_min':0.11, 'beta_max':0.16, 'num_betas': 1,
    #             'neutral_text':'a face', 'target_text': 'a face with glasses'}

     #Bobcut
    # edit_args = {'alpha_min': 3, 'alpha_max': 1, 'num_alphas':1, 'beta_min':0.11, 'beta_max':0.16, 'num_betas': 1,
    #             'neutral_text':'a face', 'target_text': 'a face with bobcut'}



