# python3.7
"""Configuration for training StyleGAN Encoder on FF-HQ (256) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""
comment = "Train edits after 64x64 resolution. Small E1."
#------------------------------------------------------------------------------------------
# General Settings
#------------------------------------------------------------------------------------------

#checkpoints/stylegan_ffhq256.pth or checkpoints/stylegan2_official_ffhq256.pth 
gan_model_path = '/media/hdd2/adundar/hamza/genforce/checkpoints/stylegan2_official_ffhq1024.pth'
perceptual_model_path = '/media/hdd2/adundar/hamza/genforce/checkpoints/vgg16.pth'
meanw_path = '/media/hdd2/adundar/hamza/genforce/checkpoints/meanw.npy'
arcface_path = '/media/hdd2/adundar/hamza/genforce/checkpoints/arcface.pth'

runner_type = 'EncoderRunner'
gan_type = 'stylegan2_official' #Can be 'stylegan2_official_reduced' or 'stylegan2_official'
encoder_type = 'e4e' #Can be 'idinvert', 'pSp', 'styletransformer', 'e4e', 'hfgi', 'hyperstyle', 'highres'.
resolution = 256
gan_res = 1024
disc_res = 256
batch_size = 1
val_batch_size = 5
total_img = 12000_000 #12000_000
space_of_latent = 'wp'
mixing_method = None # How to combine wp_enc and wp_rand? Can be 'blender', 'fc', 'encoder_in' or 'attention'. Attention not finished yet.
mapping_method = 'pretrained' #How to map z to w? 'pretrained' or 'train_fc'
latent_dim = 512 #Style vector dimension
repeat_w = True #Broadcast w in Mapping
test_time_optims = 0  # None -> 0, Optimize Latents -> 1 (like IDinvert), Optimize Latents and Noise -> 2 (like Image2Stylegan++)
wrand_edit_factor = 4
edit_prob = 0.5 #0.5
randw_prob = 1 #0.5
interface_prob = 0.0
embed_res = 64      #For now 32 or 64

#------------------------------------------------------------------------------------------
# Data Settings
#------------------------------------------------------------------------------------------

# Training dataset is repeated at the beginning to avoid loading dataset
# repeatedly at the end of each epoch. This can save some I/O time.
data = dict(
    num_workers=0,
    repeat=500,
    train=dict(root_dir='/media/hdd2/adundar/hamza/genforce/data/ffhq256_train.zip', data_format='zip', resolution=resolution, mirror=0.5, masked_ratio=[0.0,0.0]),
    val=dict(root_dir='/media/hdd2/adundar/hamza/genforce/data/temp/images256', data_format='dir', resolution=resolution, masked_ratio=[0.0,0.0]),
    smile_p = dict(root_dir= '/media/hdd2/adundar/hamza/genforce/ata/CelebA-HQ-img', ann_dir= 'data/CelebAMask-HQ-attribute-anno.txt', data_format='dir', resolution=resolution, attribute_tag='smile', ispositive=True),
    smile_n = dict(root_dir='/media/hdd2/adundar/hamza/genforce/data/CelebA-HQ-img', ann_dir= 'data/CelebAMask-HQ-attribute-anno.txt', data_format='dir', resolution=resolution, attribute_tag='smile', ispositive=False),
    smile=dict(data_format='dir', resolution=resolution)
    # train=dict(root_dir='data/', data_format='list',
    #            image_list_path='data/ffhq/ffhq_train_list.txt',
    #            resolution=resolution, mirror=0.5, masked_ratio=[0.0,1.0]),
    # val=dict(root_dir='data/', data_format='list',
    #          image_list_path='./data/ffhq/ffhq_val_list.txt',
    #          resolution=resolution, masked_ratio=[0.0, 1.0]),
)

#------------------------------------------------------------------------------------------
# Contoroller Settings
#------------------------------------------------------------------------------------------

controllers = dict(
    RunningLogger=dict(every_n_iters=50),
    Snapshoter=dict(every_n_iters=1250, first_iter=False, num=1), #num does nothing, number of output samples always equal to val_batch_size
    Checkpointer=dict(every_n_iters=100000, first_iter=False, initial_fid = 1000), 
    FIDEvaluator=dict(every_n_iters=2500, first_iter=True, num=2000), #'auto' means evaluate metrics for entire val set.
)

#------------------------------------------------------------------------------------------
#Model Settings
#------------------------------------------------------------------------------------------
"""
LR_SCHED OPTIONS:
lr_type='FIXED' ->  Fixed Learning Rate
lr_type='STEP', decay_factor=0.5, decay_step =50000 -> Update rule: After each 50000 steps, lr = lr * (decay_factor)
lr_type='ExpSTEP', decay_factor=0.8, decay_step=50000 -> Update rule : Within each decay_step, lr exponentially decays decay_factor amount.
"""

#-----------------------Generator and Discriminator ---------------------------#
G_kwargs_model = dict(type=gan_type, resolution=gan_res)
G_kwargs_val = dict()
D_kwargs_model = dict(type=gan_type, resolution=disc_res)
D_kwargs_train = dict()
D_kwargs_val = dict()
if (gan_type == 'stylegan'):
    G_kwargs_model['repeat_w'] = repeat_w
    G_kwargs_val['randomize_noise'] = False
elif (gan_type == 'stylegan2_official' or 'stylegan2_official_reduced'):
    G_kwargs_model['fused_modconv_default'] = 'inference_only' # Speed up training by using regular convolutions instead of grouped
    if gan_res == 256:
        G_kwargs_model['channel_base'] = 16384
    G_kwargs_model['embed_res'] = embed_res
    G_kwargs_val['noise_mode'] = 'const'
    G_kwargs_val['force_fp32'] = True
    if disc_res == 256:
        D_kwargs_model['channel_base'] = 16384
    D_kwargs_train['c'] = None #No class input
    D_kwargs_val['c'] = None #No class input
else:
    raise "Unsupported Generator type!"
#----------------------------------------------------------#
#-----------------------Encoder ---------------------------#
if encoder_type == 'idinvert':
    E_kwargs_model = dict(type=encoder_type, resolution=resolution, network_depth=18, #depth can be 18,34
                        latent_dim = [512]*14,#[1024] * 8 + [512, 512, 256, 256, 128, 128]
                        num_latents_per_head=[4, 4, 6],
                        use_fpn=True,
                        fpn_channels=512,
                        use_sam=True,
                        sam_channels=512,
                        norm_fn = None, #Can be None or 'sync' or 'style'.
                        activation_fn = 'lrelu', #Can be lrelu or relu
                        downsample_fn = 'max_pool', #Can be 'max_pool', 'avg_pool' or 'strided'
                        use_blender=mixing_method, # If it is False, blender is returned as None
                        attention=False, #Whether to use attention module
                    ) 
elif encoder_type == 'pSp':
    E_kwargs_model = dict(type=encoder_type, resolution=resolution, 
                        psp_path='/media/hdd2/adundar/hamza/genforce/checkpoints/psp_ffhq_encode.pt',
                        num_layers=50, #ResNet depth. It can be 50,100 or 152
                        mode='ir_se',  #Mode can be 'ir' or 'ir_se'.
                    )
elif encoder_type == 'styletransformer':

    E_kwargs_model = dict(type=encoder_type, resolution=gan_res, basic_enc_path='/media/hdd2/adundar/hamza/styletrans/ckpts/style_transformer_ffhq.pt', 
                            num_layers = 50,
                            mode='ir_se', #Can be 'ir' or 'ir_se'
                            out_res = embed_res
                        )

elif encoder_type == 'e4e':
    E_kwargs_model = dict(type=encoder_type, resolution=gan_res, e4e_path='/media/hdd2/adundar/hamza/genforce/checkpoints/e4e_ffhq_encode.pt', 
                            num_layers = 50,
                            mode='ir_se', #Can be 'ir' or 'ir_se'
                            out_res = embed_res
                        )
elif encoder_type == 'hfgi':
    E_kwargs_model = dict(type=encoder_type, resolution=resolution, basic_encoder='pSp', #Basic Encoder can be pSp or e4e
                        basic_encoder_path='checkpoints/psp.pth', #Where to find pretrained basic encoder
                        num_layers=50, norm_fn=None, use_blender=mixing_method, #These should be same with pretrained basic encoder
                        distortion_scale=0.15, aug_rate=0.9, #Augmentation parameters for ADA
                        ) 
elif encoder_type == 'hyperstyle':
     E_kwargs_model =  dict(type=encoder_type, resolution=gan_res, path='/media/hdd2/adundar/hamza/hyperstyle/pretrained_models/faces_w_encoder.pt', 
                            num_layers = 50,
                            mode='ir_se', #Can be 'ir' or 'ir_se'
                            out_res = embed_res
                        ) 
#----------------------------------------------------------#
#-----------------------Mapping ---------------------------#
M_kwargs_model = dict()
M_kwargs_train = dict()
M_kwargs_val = dict()
if mapping_method == 'train_fc':
    M_kwargs_model = dict(type="fc", z_dim=512, c_dim=0, w_dim=512, num_ws= 14 if repeat_w else None) #No class label
    M_kwargs_train = dict(c=0, truncation_psi=1, truncation_cutoff=None, update_emas=False, repeat_w = repeat_w) #No truncation is set. Different w values used.
    M_kwargs_val = dict(c=0, truncation_psi=1, truncation_cutoff=None, update_emas=False, repeat_w = repeat_w) #No truncation is set. Different w values used.

#--------------------------MIX--------------------------------#
MIX_kwargs_model = dict()
if mixing_method == 'fc':
    MIX_kwargs_model = dict(type='fc_layers', latent_size=512)
#-------------------------------------------------------------#

modules = dict(
    discriminator=dict(
        model=D_kwargs_model,
        lr = dict(lr_type='STEP', decay_factor=0.5, decay_step=[5000, 10000, 15000]),  #decay_step=[5000, 10000, 15000, 20000, 25000]
        opt=dict(opt_type='Adam', base_lr=1e-4, betas=(0.9, 0.99)), 
        kwargs_train=D_kwargs_train,
        kwargs_val=D_kwargs_val,
    ),
    discriminator2=dict(
        model=D_kwargs_model,
        lr=dict(lr_type='STEP', decay_factor=0.5, decay_step =50000),  
        opt=dict(opt_type='Adam', base_lr=1e-4, betas=(0.9, 0.99)),
        kwargs_train=D_kwargs_train,
        kwargs_val=D_kwargs_val,
    ),
    generator_smooth=dict(
        model=G_kwargs_model,
        lr = dict(lr_type='STEP', decay_factor=0.5, decay_step=[5000, 10000, 15000]), #decay_step=[5000, 10000, 15000, 20000, 25000]
        opt=dict(opt_type='Adam', base_lr=1e-4, betas=(0.9, 0.99)),
        kwargs_val=G_kwargs_val,
    ),
    latent_disc=dict(
        model = dict(type=gan_type, resolution=resolution, latent_size = 512),
        lr = dict(lr_type='STEP', decay_factor=0.5, decay_step=50000),
        opt = dict(opt_type='Adam', base_lr=1e-4, betas=(0.9, 0.99)),
    ),
    encoder=dict(
        model=E_kwargs_model,
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-4, betas=(0.9, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    mixer = dict(
        model=MIX_kwargs_model,
        lr = None, #Same as encoder
        opt = None,
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    mapping=dict(
        model=M_kwargs_model,
        lr= None, #Same as encoder
        opt= None,
        kwargs_train=M_kwargs_train,
        kwargs_val=M_kwargs_val,

    )
)

#------------------------------------------------------------------------------------------
#Loss Settings
#------------------------------------------------------------------------------------------
meanw_settings = dict(meanw_path=meanw_path, n_samples=1e5)

loss = dict(
    type='EncoderLoss',
    d_loss_kwargs=dict(r1_gamma=10.0, r2_gamma=0.0), #Discriminator for real and final
    e_loss_kwargs=dict(adv_lw=0.1,
    pixel_lw = 1.0, perceptual_lw=1e-3, f_lw = 5.0, align_lw=0.0, id_lw = 0.1 , g_lw = 0, cycle_lw=1.0,
                       latent_adv_lw=0.0, adv2_lw=0.0, 
                       ratio_lw=0.0, meanw_lw = 0.01, 
                       iver_image_lw = 0.0, diver_latent_lw = 0.00),
    d2_loss_kwargs = dict(r1_gamma=10.0, r2_gamma=0.0), #Discriminator for real and fake
    same_style_loss_kwargs = dict(use = False, pixel_lw=1.0, consistency_lw=1.0),  #We can add more losses like perceptual etc. Set use=False to not train with this.
    ld_loss_kwargs=dict(),
    perceptual_kwargs=dict(output_layer_idx=23,
                           pretrained_weight_path=perceptual_model_path),
    ratio_loss_kwargs = dict(path=arcface_path, k=1), #blender/cosine = k, ln(blender) - ln(cosine) = ln(k)
    hfgi_loss_kwargs = dict(ada_lw=0.1), #Supervised Training of ADA
    id_loss_kwargs = dict(id_loss_edits = False)
)

"""
LOSS EXPLANATIONS:
0 to disable specific loss
r1_gamma: Grad penalty calculated with real samples
r2_gamma: Grad penalty calculated with fake samples
adv_lw: Adversarial loss weigth for the Encoder where the fake image is R * M + F * (1- M)
adv2_lw: Adversarial loss weigth for the Encoder where the fake image is generator output.
pixel_lw: Pixel loss weight on the unmasked regions.
perceptual_lw: VGG loss weight on the unmasked regions.
latent_adv_lw: Latent adversarial loss weight for the Encoder
ratio_lw: Ratio loss between blender and valid area.
meanw_lw: Mean w regularizer loss weigth.
cycle_lw: Cycle loss between w_mixed and w_enc_hat. 
diver_image_lw: Diversification loss weight between |final 1 - final 2|
diver_latent_lw: Diversification loss weight between |wp_mixed1 - w_mixed2|
"""
#------------------------------------------------------------------------------------------

#Latent Discriminator is not Created When Latent Adv Loss is 0
use_latent_disc = loss['e_loss_kwargs']['latent_adv_lw'] > 0
use_disc2 = loss['e_loss_kwargs']['adv2_lw'] > 0
