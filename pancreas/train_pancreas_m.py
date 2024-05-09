from asyncore import write
from audioop import avg
from cgi import test
import imp
from multiprocessing import reduction
from turtle import pd
from unittest import loader, result

from yaml import load
import torch
import os
import pdb
import torch.nn as nn

from tqdm import tqdm as tqdm_load
from pancreas_utils import *
from test_util import *
from losses import DiceLoss, softmax_mse_loss, mix_loss
from dataloaders import get_ema_model_and_dataloader
from dataloaders import get_ema_model_and_dataloader_ex


"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
seed_test = 2020
seed_reproducer(seed = seed_test)

data_root, split_name = '/data/fazhan/BCP-main/code/data/pancreas/processed_h5', 'pancreas'
result_dir = 'result/cutmix/'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 120, 600
pretrain_save_step, st_save_step, pred_step = 20, 10, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = 20
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64
patch_size_ex = 32


logger = None
overall_log = 'cutmix_log.txt'

def pretrain(net1, optimizer, lab_loader_a, labe_loader_b, test_loader):
    """pretrain image- & patch-aware network"""

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain_SGD'
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = cutmix_config_log(save_path, tensorboard=True)
    logger.info("cutmix Pretrain, patch_size: {}, save path: {}".format(patch_size, str(save_path)))

    max_dice = 0
    measures = CutPreMeasures(writer, logger)
    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        measures.reset()
        """Testing"""
        if epoch % pretrain_save_step == 0:
            avg_metric1,_= test_calculate_metric(net1, test_loader.dataset)
            logger.info('average metric is : {}'.format(avg_metric1))
            val_dice = avg_metric1[0]

            if val_dice > max_dice:
                save_net_opt(net1, optimizer, save_path / f'best_ema{val_dice}_pre.pth', epoch)
                save_net_opt(net1, optimizer, save_path / f'best_ema{label_percent}_pre.pth', epoch)
                
                max_dice = val_dice
            
            writer.add_scalar('test_dice', val_dice, epoch)
            logger.info('Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice, max_dice))
            save_net_opt(net1, optimizer, save_path / ('%d.pth' % epoch), epoch)
        
        """Training"""
        net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b  = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, patch_size)   

            img = img_a * img_mask + img_b * (1 - img_mask)
            lab = lab_a * img_mask + lab_b * (1 - img_mask)

            out = net1(img)[0]
            ce_loss1 = F.cross_entropy(out, lab)
            dice_loss1 = DICE(out, lab)
            loss = (ce_loss1 + dice_loss1) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            measures.update(out, lab, ce_loss1, dice_loss1, loss)
            measures.log(epoch, epoch * len(lab_loader_a) + step)
        writer.flush()
    return max_dice

def ema_cutmix(net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader):
    """Create Path"""
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger 
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    """Load Model"""
    pretrained_path = Path(result_dir) / 'pretrain'
    load_net_opt(net, optimizer, pretrained_path / f'best_ema{label_percent}_pre.pth')
    load_net_opt(ema_net, optimizer, pretrained_path / f'best_ema{label_percent}_pre.pth')
    logger.info('Loaded from {}'.format(pretrained_path))

    max_dice = 0
    max_list = None
    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        measures.reset()
        logger.info('')

        """Testing"""
        if epoch % st_save_step == 0:
            avg_metric,_= test_calculate_metric(net, test_loader.dataset)
            
            logger.info('average metric is : {}'.format(avg_metric))
            val_dice = avg_metric[0]
            writer.add_scalar('val_dice', val_dice, epoch)

            """Save Model"""
            if val_dice > max_dice:
                save_net(net, str(save_path / f'best_ema_{val_dice}_self.pth'))
                max_dice = val_dice
                max_list = avg_metric

            logger.info('Evaluation: val_dice: %.4f, val_maxdice: %.4f' % (val_dice, max_dice))
            
        """Training"""
        net.train()
        ema_net.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            """Generate Pseudo Label"""
            with torch.no_grad():
                unimg_a_out = ema_net(unimg_a)[0]
                unimg_b_out = ema_net(unimg_b)[0]
                uimg_a_plab = get_cut_mask(unimg_a_out, nms=True, connect_mode=connect_mode)
                uimg_b_plab = get_cut_mask(unimg_b_out, nms=True, connect_mode=connect_mode)
                img_mask, loss_mask = generate_mask(img_a, patch_size)     
            
            # """Mix input"""
            net3_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            net3_input_unlab = img_a * img_mask + unimg_b * (1 - img_mask)

            """Supervised Loss"""
            mix_output_l = net(net3_input_l)[0]
            loss_1 = mix_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_output_2 = net(net3_input_unlab)[0]
            loss_2 = mix_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask)

            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(net, ema_net, alpha)

            measures.update(loss_1, loss_2, loss)  
            measures.log(epoch, epoch*len(lab_loader_a) + step)

        if epoch ==  self_training_epochs:
            save_net(net, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
        writer.flush()
    return max_dice, max_list


def ema_cutmix_ex(net, ema_net,ema_net_copy, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader):
    """Create Path"""
    save_path = Path(result_dir) / 'self_train_ens'
    save_path.mkdir(exist_ok=True)
    
    shutil.copy('/data/fazhan/BCP-main/code/pancreas/train_pancreas_m.py', save_path)
    

    """Create logger and measures"""
    global logger 
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    """Load Model"""
    pretrained_path = '/data/fazhan/BCP-main/code/pancreas/result/cutmix/pretrain_SGD/best_ema0.7450176900184577_pre.pth'
    load_net_opt(net, optimizer, pretrained_path)
    load_net_opt(ema_net, optimizer, pretrained_path)
    load_net_opt(ema_net_copy, optimizer, pretrained_path)
    
    logger.info('Loaded from {}'.format(pretrained_path))

    max_dice = 0
    count=0
    max_list = None
    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        measures.reset()
        logger.info('')

        """Testing"""
        
        """Training"""
        net.train()
        ema_net.train()
        ema_net_copy.train()
        
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            """Generate Pseudo Label"""
            flag=0
            
            with torch.no_grad():
                random_number = random.uniform(0, 1)
                if random_number>0.5:
                    flag=1
                    
                # if count==0:
                #     count = random.randint(1, 5)
                #     flag=1-flag
                # else:
                #     count=count-1    
                # if flag==1:
                #     unimg_a_out = ema_net(unimg_a)[0]
                #     unimg_b_out = ema_net(unimg_b)[0]
                # else:
                #     unimg_a_out = ema_net_copy(unimg_a)[0]
                
                #     unimg_b_out = ema_net_copy(unimg_b)[0]
                                
                unimg_a_out = ema_net(unimg_a)[0]
                unimg_b_out = ema_net(unimg_b)[0]
                
                unimg_a_out_ex = ema_net_copy(unimg_a)[0]
                unimg_b_out_ex = ema_net_copy(unimg_b)[0]

                uimg_a_plab = get_cut_mask_ens(unimg_a_out,unimg_a_out_ex, nms=True, connect_mode=connect_mode)
                uimg_b_plab = get_cut_mask_ens(unimg_b_out,unimg_b_out_ex, nms=True, connect_mode=connect_mode)
                img_mask, loss_mask = generate_mask(img_a, patch_size)  
                img_mask_a, loss_mask_a = generate_mask(img_a, patch_size_ex)     
                img_mask_b, loss_mask_b = generate_mask(img_a, patch_size_ex)     
            if flag==0:
                
                
                
                    mixla_img = img_a * img_mask_a + unimg_a * (1 - img_mask_a)
                    mixla_img_x = unimg_a * img_mask_a + img_a * (1 - img_mask_a)
                    mixla_lab = lab_a * img_mask_a + uimg_a_plab * (1 - img_mask_a)
                    mixla_lab_x = uimg_a_plab * img_mask_a + lab_a * (1 - img_mask_a)

            
            else:
                    mixla_img = img_a
                    mixla_img_x = unimg_a
                    mixla_lab = lab_a
                    mixla_lab_x = uimg_a_plab
            if flag==1:
                
                    
                mixlb_img = img_b * img_mask_b + unimg_b * (1 - img_mask_b)
                mixlb_img_x = unimg_b * img_mask_b + img_b * (1 - img_mask_b)

                mixlb_lab = lab_b * img_mask_b + uimg_b_plab * (1 - img_mask_b)
                mixlb_lab_x = uimg_b_plab * img_mask_b + lab_b * (1 - img_mask_b)

                
            else:
                mixlb_img = img_b
                mixlb_img_x = unimg_b
                mixlb_lab = lab_b
                mixlb_lab_x = uimg_b_plab
            
            # """Mix input"""
            # net3_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            # net3_input_unlab = img_a * img_mask + unimg_b * (1 - img_mask)
            mixl_img = mixla_img * img_mask + mixlb_img_x * (1 - img_mask)
            mixu_img = mixla_img_x * img_mask + mixlb_img * (1 - img_mask)
            """Supervised Loss"""
            mix_output_l = net(mixl_img)[0]
            #loss_1 = mix_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_output_2 = net(mixu_img)[0]
            if flag==0:
                loss_mask_ex=loss_mask_a*loss_mask
            else:
                loss_mask_ex=(1-loss_mask_b)|loss_mask
            #loss_2 = mix_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask)
            loss_1 = mix_loss(mix_output_l, mixla_lab.long(), mixlb_lab_x.long(), loss_mask_ex)
            loss_2 = mix_loss(mix_output_2, mixla_lab_x.long(), mixlb_lab.long(), loss_mask_ex,unlab=True)
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if flag==1:
                update_ema_variables(net, ema_net, 0.99)
            else:
                update_ema_variables(net, ema_net_copy, 0.99)
            # update_ema_variables(net, ema_net, alpha)

            measures.update(loss_1, loss_2, loss)  
            measures.log(epoch, epoch*len(lab_loader_a) + step)
        if epoch % st_save_step == 0 and epoch!=0:
            avg_metric,_= test_calculate_metric(net, test_loader.dataset)
            
            logger.info('average metric is : {}'.format(avg_metric))
            val_dice = avg_metric[0]
            writer.add_scalar('val_dice', val_dice, epoch)

            """Save Model"""

            if val_dice > max_dice:
                save_net(net, str(save_path / f'best_ema_{round(val_dice, 4)}_epoch{epoch}_self.pth'))
                max_dice = val_dice
                max_list = avg_metric
            # elif val_dice > max_dice-0.003:
            #     save_net(net, str(save_path / f'best_ema_{round(val_dice, 4)}_epoch{epoch}_self.pth'))
                
                

            logger.info('Evaluation: val_dice: %.4f, val_maxdice: %.4f' % (val_dice, max_dice))
            
        if epoch ==  self_training_epochs:
            save_net(net, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
        writer.flush()
    return max_dice, max_list

def test_model(net, test_loader):
    load_path = Path(result_dir) / self_train_name
    load_path='/data/fazhan/BCP-main/code/pancreas/result/cutmix/self_train_ens/best_ema_0.8368_epoch380_self.pth'
    load_net(net, load_path)
    print('Successful Loaded')
    avg_metric, m_list = test_calculate_metric(net, test_loader.dataset, s_xy=16, s_z=4)
    test_dice = avg_metric[0]
    print(avg_metric)
    return avg_metric, m_list


if __name__ == '__main__':
    try:
        net, ema_net,ema_net_copy, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader_ex(data_root, split_name, batch_size, lr, labelp=label_percent)
        
        #net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent)
        #pretrain(net, optimizer, lab_loader_a, lab_loader_b, test_loader)
        #ema_cutmix(net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader)
        #ema_cutmix_ex(net, ema_net,ema_net_copy, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader)
        
        avg_metric, m_list = test_model(net, test_loader)
    except Exception as e:
        logger.exception("BUG FOUNDED ! ! !")


