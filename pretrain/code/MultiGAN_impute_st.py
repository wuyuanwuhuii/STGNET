import torch
import torch.optim as optim
import time
import sys
import os
from utiles import CriticUpdater_st, CriticUpdater, mask_norm, mkdir, mask_data,one_hot,GeneratorLoss
from pathlib import Path
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor

#data_gene = imputer2

def scMultiGAN_impute(args, imputer_2, imputer_1,
                   impu_critic_st, impu_critic_sc,
                  data_loader):
    impute_dir = args.output_dir
    checkpoint = args.checkpoint
    mkdir(Path(impute_dir)/'model_impute')
    model_dir = Path(os.path.join(impute_dir,"model_impute"))
    n_critic = args.n_critic
    lambd = args.lambd
    batch_size = args.batch_size
    nz = args.latent_dim
    epochs = args.epoch
    save_model_interval = args.save_interval
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    tau = args.tau
    img_size = args.img_size
    update_all_networks = not args.imputeronly

    n_batch = len(data_loader)
    data_noise = torch.FloatTensor(batch_size, nz).to(device)
    impu_noise = torch.FloatTensor(batch_size, args.channels,img_size,img_size).to(device)
    eps = torch.FloatTensor(batch_size, 1, 1, 1).to(device)
    ones = torch.ones(batch_size).to(device)
    #lrate = 1e-4
    imputer_lrate = args.lr 

    generator_loss = GeneratorLoss()
    imputer_optimizer = optim.Adam(
        imputer_2.parameters(), lr=imputer_lrate)
    impu_critic_optimizer = optim.Adam(
        impu_critic_st.parameters(), lr=imputer_lrate)
    update_impu_critic_st = CriticUpdater_st( 
        impu_critic_st, impu_critic_optimizer, eps, ones, lambd)  #  CriticUpdater  -> CriticUpdater_st
        
    #update_impu_critic_sc = CriticUpdater(impu_critic_sc, impu_critic_optimizer, eps, ones, lambd)

    start_epoch = 0
    critic_updates = 0

    pretrain = torch.load(args.pretrain, map_location=device)
    #data_gen.load_state_dict(pretrain['data_gen'])
    
    imputer_1.load_state_dict(pretrain['imputer'])  #old
    #ckpt = torch.load(args.pretrain, map_location="cpu")
    #print("Loaded checkpoint type:", type(ckpt))
    #if isinstance(ckpt, dict):
    #    print("Checkpoint keys:", ckpt.keys())
    
    if checkpoint:
        print("Using pretained model {}".format(checkpoint))
        checkpoint = torch.load(Path(checkpoint))
        imputer_2.load_state_dict(checkpoint['imputer'])
        impu_critic.load_state_dict(checkpoint['impu_critic'])
        start_epoch = checkpoint['epoch']-1
        critic_updates = checkpoint['critic_updates']


    def save_model(path, epoch, critic_updates=0):
        torch.save({
            'imputer': imputer_2.state_dict(),
            'impu_critic': impu_critic_st.state_dict(),
            'epoch': epoch + 1,
            'critic_updates': critic_updates,
            'args': args,
        }, str(path))
    start = time.time()
    epoch_start = start
    
    file_path = 'training_impute_log.txt'
    if os.path.exists(file_path):
        os.remove(file_path)
    #update_impu_critic = CriticUpdater(impu_critic, impu_critic_optimizer, eps, ones, lambd)
    with open('training_impute_log.txt', 'a') as file:
        tt = []
      
        for epoch in range(start_epoch, epochs):
            #print('start train')
            sum_data_loss, sum_mask_loss, sum_impu_loss = 0, 0, 0
            for i,real_sample in enumerate(data_loader):
                real_data = real_sample['real_data'].to(device)
                real_mask = real_sample['real_mask'].to(device)
                label = real_sample['label'].to(device)
                #print(real_sample.keys())
                
                cord = real_sample['coords'].to(device)
                
                label_hot = one_hot((label).type(torch.LongTensor),args.ncls).type(Tensor).to(device)
                
                data_noise.normal_()
                impu_noise.uniform_()
                fake_data = imputer_2(real_data, real_mask, impu_noise,label_hot, cord)
                #print("*********")
                #fake_data = data_gen(data_noise,label_hot)
    
                
                imputed_data = imputer_1(real_data,real_mask, impu_noise,label_hot )
                masked_imputed_data = mask_data(real_data, real_mask, imputed_data)
                #masked_imputed_data = imputed_data
    
                update_impu_critic_st(fake_data, masked_imputed_data,real_mask, label_hot, cord)
                sum_impu_loss += update_impu_critic_st.loss_value
                critic_updates += 1
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D_impute loss: %f]" % (
                    epoch + 1, args.epoch, i + 1, len(data_loader),
                    sum_impu_loss))
                sys.stdout.flush()
                if critic_updates == n_critic:
                    critic_updates = 0
                    for p in impu_critic_st.parameters():
                        p.requires_grad_(False)
                    impu_noise.uniform_()
                    imputed_data = imputer_1(real_data, real_mask, impu_noise,label_hot)
                    masked_imputed_data = mask_data(real_data, real_mask, imputed_data)
                    
                    #masked_imputed_data = imputed_data
                    impu_loss = 1-impu_critic_st(masked_imputed_data,label_hot, cord).mean()  # changed
                    impu_loss = generator_loss(impu_loss,fake_data,masked_imputed_data)
                    imputer_2.zero_grad()
                    if gamma > 0:
                        imputer_mismatch_loss = mask_norm(
                            (imputed_data - real_data) ** 2, real_mask)
                        (impu_loss + imputer_mismatch_loss * gamma).backward()
                    else:
                        impu_loss.backward()
                    imputer_optimizer.step()
                    for p in impu_critic_st.parameters():
                        p.requires_grad_(True)
                    print("\r[Epoch %d/%d] [Batch %d/%d] [G_impute loss: %f]" % (
                        epoch + 1, args.epoch, i + 1, len(data_loader),
                        impu_loss.detach().cpu().numpy().item()))
            mean_impu_loss = sum_impu_loss / n_batch
            print("epoch {} data_loss is {}".format(epoch + 1, mean_impu_loss))
            log_message = "Epoch {} data loss is {}\n".format(epoch + 1, mean_impu_loss)
            print(log_message.strip())
            file.write(log_message)  # data write
            #print("epoch {} data_loss is {}".format(epoch + 1, mean_impu_loss))
    
            if save_model_interval > 0 and (epoch + 1) % save_model_interval == 0 and epoch > 480:
                save_model(os.path.join(model_dir , f'{epoch:04d}_12_5.pth'), epoch, critic_updates)
            epoch_end = time.time()
            time_elapsed = epoch_end - start
            epoch_time = epoch_end - epoch_start
            epoch_start = epoch_end
