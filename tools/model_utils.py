def get_arch(opt, isp):
    from models.UNet import UNet
    from models.Uformer import Uformer

    arch = opt.net_type

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(opt, isp)
    elif arch == 'Uformer':
        model_restoration = Uformer(opt,isp, img_size=opt.crop_size,embed_dim=opt.dim,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_T':
        model_restoration = Uformer(opt,isp, img_size=opt.crop_size,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_S':
        model_restoration = Uformer(opt,isp, img_size=opt.crop_size,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_S_noshift':
        model_restoration = Uformer(opt,isp, img_size=opt.crop_size,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True,
            shift_flag=False)
    elif arch == 'Uformer_B_fastleff':
        model_restoration = Uformer(opt,isp, img_size=opt.crop_size,embed_dim=32,win_size=8,token_projection='linear',token_mlp='fastleff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True)  
    elif arch == 'Uformer_B':
        model_restoration = Uformer(opt,isp, img_size=opt.crop_size,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=opt.dd_in)  
    else:
        raise Exception("Arch error!")

    return model_restoration