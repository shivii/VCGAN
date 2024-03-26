import torch
from dataset import A_B_Dataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from discriminator_model import Discriminator
from generator_model import Generator
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.utils.data import Sampler,SequentialSampler

def topKloss(tensor1, tensor2):
  diff = torch.abs(tensor1 - tensor2)
  diff_flattened = torch.flatten(diff)
  k = int(0.1 * torch.numel(diff_flattened))
  diff_k, i = torch.topk(diff_flattened, k, sorted = True)
  topk_loss = torch.mean((torch.tensor(diff_k)).clone().detach())
  
  return topk_loss

def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    A_reals = 0
    A_fakes = 0
    loop = tqdm(loader, leave=True)
    #print("\n loop:", loop)

    for idx, (B, A) in enumerate(loop):
        #print("idx: ", idx)
        B = B.to(config.DEVICE)
        A = A.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # put it togethor
            D_loss = (D_A_loss + D_B_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            # cycle loss
            cycle_B = gen_B(fake_A)
            cycle_A = gen_A(fake_B)
            cycle_B_loss = l1(B, cycle_B)
            cycle_A_loss = l1(A, cycle_A)
            
            #agg_loss
            agg_loss_B = topKloss(B, cycle_B)
            agg_loss_A = topKloss(A, cycle_A)
            
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_B = gen_B(B)
            identity_A = gen_A(A)
            identity_B_loss = l1(B, identity_B)
            identity_A_loss = l1(A, identity_A)

            '''
            Version 1: adv_loss + cycle_loss*LAMBDA_CYCLE + agg_loss
            Version 2: adv_loss + agg_loss*LAMBDA_CYCLE
            '''
            
            # add all togethor
            G_loss = (
                loss_G_B
                + loss_G_A
               # + cycle_B_loss * config.LAMBDA_CYCLE
               # + cycle_A_loss * config.LAMBDA_CYCLE
                + identity_A_loss * config.LAMBDA_IDENTITY
                + identity_B_loss * config.LAMBDA_IDENTITY
                + agg_loss_A * config.LAMBDA_CYCLE
                + agg_loss_B * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 80 == 0:
            img_A = 'agg_1/A_' + str(idx) + '_'+ str(epoch) + '.png'
            img_B = 'agg_1/B_' + str(idx) + '_'+ str(epoch) + '.png'
            #save_image(fake_A*0.5+0.5, img_A)
            #save_image(fake_B*0.5+0.5, img_B)
            
            # generate samples A->B, B->A
            fakeB = gen_B(A)*0.5+0.5
            fakeA = gen_A(B)*0.5+0.5
            
            new_im1 = torch.cat([A, fakeB], dim = 0)
            new_im2 = torch.cat([B, fakeA], dim = 0)
            
            grid_img1 = make_grid(new_im1, nrow=2)
            grid_img2 = make_grid(new_im2, nrow=2)
            save_image( grid_img1, img_A)
            save_image( grid_img2 , img_B)
            
            #save_image(fake_A*0.5+0.5, f"saved_images/A_{idx}.png")
            #save_image(fake_B*0.5+0.5, f"saved_images/B_{idx}.png")

        loop.set_postfix(A_real=A_reals/(idx+1), A_fake=A_fakes/(idx+1))
    
    cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo = cycle_A_loss.detach(), cycle_B_loss.detach(), D_A_loss.detach(), D_B_loss.detach(), loss_G_A.detach(), loss_G_B.detach() 
    agg_A, agg_B = agg_loss_A, agg_loss_B
    
    return cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, agg_A, agg_B

def main():
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE,
        )

    
    dataset = A_B_Dataset(
            root_A=config.TRAIN_DIR+"A", root_B=config.TRAIN_DIR+"B", transform=config.transforms
    )
    val_dataset = A_B_Dataset(
            root_A=config.VAL_DIR+"A", root_B=config.VAL_DIR+"B", transform=config.transforms
    )
    
    dataset_seq = SequentialSampler(dataset)
    val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
    )
    loader = DataLoader(
            dataset,
            sampler = dataset_seq,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
    )

    print("Image Shape: {}".format(dataset[0][0].numpy().shape), end = '\n\n')
    print("Training Set:   {} samples".format(len(dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)))
    print("epochs:", config.NUM_EPOCHS)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    cl_A = []
    cl_B = []
    d_A = []
    d_B = []
    g_A = []
    g_B = []
    a_A = []
    a_B = []

    for epoch in range(config.NUM_EPOCHS):
        print("\n For epoch :", epoch)
        cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, agg_A, agg_B = train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)
        print("Loss: ", cl_Ao, cl_Bo, d_Ao, d_Bo, g_Ao, g_Bo, agg_A, agg_B)
        cl_A.append(cl_Ao)
        cl_B.append(cl_Bo)
        d_A.append(d_Ao)
        d_B.append(d_Bo)
        g_A.append(g_Ao)
        g_B.append(g_Bo)
        a_A.append(agg_A)
        a_B.append(agg_B)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

    # plots
    plt.figure(figsize=(10, 7))
    plt.plot(cl_A, color='green', linestyle='-', label='Cycle_loss_A')
    plt.plot(cl_B, color='yellow', linestyle='-', label='Cycle_loss_B')
    plt.plot(d_A, color='blue', linestyle='-', label='Disc A')
    plt.plot(d_B, color='red', linestyle='-', label='Disc B')
    plt.plot(g_A, color='orange', linestyle='-', label='Gen A')
    plt.plot(g_B, color='black', linestyle='-', label='Gen B')
    plt.plot(a_A, color='orange', linestyle='-', label='Agg A')
    plt.plot(a_B, color='black', linestyle='-', label='Agg B')    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Agg_loss.jpg')
    plt.show()



if __name__ == "__main__":
    main()
