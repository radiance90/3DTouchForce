import numpy as np
import pandas as pd
from net_small_2feature import Net
import torch
from dataset_large_mask_loss import ForceDataset
from torch.utils.data import DataLoader
import time
import os
from logger_new import get_logger
from sklearn.model_selection import train_test_split
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


if __name__ == '__main__':
    log_path = 'model/2frame_relative_moving_small_press_data_coestimate/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log = get_logger(log_path + 'training.log')
    train_path_press = 'training_lists/train_list_press.csv'
    # train_path_shear = 'training_lists/train_list_shear.csv'
    train_list = pd.read_csv(train_path_press).values
    # items2 = pd.read_csv(train_path_shear).values
    # train_list = np.concatenate((items1, items2))

    test_path_press = 'training_lists/test_list_press.csv'
    # test_path_shear = 'training_lists/test_list_shear.csv'
    test_list = pd.read_csv(test_path_press).values
    # items2t = pd.read_csv(test_path_shear).values
    # test_list = np.concatenate((items1t, items2t))

    phase = 'train'
    pin_memory = False
    num_workers = 8
    batch_size = 512

    model = Net().cuda()
    # model.load_state_dict(torch.load('model/2frame_more_relative_to_first_all_train_large/600.pth'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_x = torch.nn.MSELoss().cuda()
    loss_y = torch.nn.MSELoss().cuda()
    loss_z = torch.nn.MSELoss().cuda()

    training_dataset = ForceDataset(train_list, phase)
    testing_dataset = ForceDataset(test_list, 'test')
    data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=True)

    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory)
    losses = []
    for epoch in range(1001):
        model.train()
        train_time_sp = time.time()
        running_loss = 0.0
        for batch_id, batch_data in enumerate(data_loader):
            img, force_x, force_y, force_z, action = batch_data
            # image, x_moved, y_moved = batch_data
            img = img.cuda()
            force_x = force_x.cuda()
            force_y = force_y.cuda()
            force_z = force_z.cuda()

            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_x(outputs[0].view(-1), force_x) + loss_y(outputs[1].view(-1), force_y) + loss_z(outputs[2].view(-1), force_z)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - train_time_sp
        losses.append(running_loss)
        log.info('Epoch: {}, avg_batch_loss = {:.9f}, epoch_time = {:.2f}'
                 .format(epoch, running_loss / len(data_loader), epoch_time))

        scheduler.step()
        if epoch % 10 == 0:# and epoch > 0:
            if epoch % 10 == 0 and epoch > 0:
                torch.save(model.state_dict(), log_path+str(epoch)+'.pth')
            # torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict()},
            #            train_path.replace('list.csv', 'model'+str(epoch))+'.pth.tar')
            # loss_data = pd.DataFrame(losses)
            # loss_data.to_csv(train_path.replace('list.csv', 'loss' + str(epoch) + '.csv'), index=False)
            count = 0
            x_diff_total = 0
            y_diff_total = 0
            z_diff_total = 0

            model.eval()
            with torch.no_grad():
                for batch_id, batch_data in enumerate(test_loader):
                    img, force_x, force_y, force_z, action = batch_data
                    # image, x_moved, y_moved = batch_data
                    img = img.cuda()
                    force_x = force_x.numpy()
                    force_y = force_y.numpy()
                    force_z = force_z.numpy()
                    # action = action.numpy()
                    outputs = model(img)
                    pred_x = outputs[0].view(-1).cpu().numpy()
                    pred_y = outputs[1].view(-1).cpu().numpy()
                    pred_z = outputs[2].view(-1).cpu().numpy()
                    x_diff = np.abs(pred_x - force_x)
                    y_diff = np.abs(pred_y - force_y)
                    z_diff = np.abs(pred_z - force_z)

                    x_diff_total += sum(x_diff)
                    y_diff_total += sum(y_diff)
                    z_diff_total += sum(z_diff)
                    count += force_x.shape[0]
            log.info('average x_diff = {:.3f} , average y_diff = {:.3f}, average z_diff = {:.3f}'.format(x_diff_total / count, y_diff_total / count, z_diff_total / count))
            

            


    log.info('Finished Training')


