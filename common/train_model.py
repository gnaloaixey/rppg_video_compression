import copy
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from singleton_pattern.load_config import get_config
from importlib import import_module
from common.cache import Cache,CacheType
from common.import_tqdm import tqdm
from common.cuda_info import get_device
import matplotlib.pyplot as plt

def run(model:nn.Module,train_dataloader:DataLoader,test_dataloader:DataLoader):
    start_time = datetime.datetime.now()
    try:
        train_config = get_config().get('train',{})
        num_epochs = train_config.get('num_epochs',10)
        min_test_loss = train_config.get('',0.01)
        min_train_loss = train_config.get('',0.01)

        optim_config = train_config.get('optim',{})
        optim_package = optim_config.get('package','torch.optim')
        optim_name = optim_config.get('name','Adam')
        optim_params = optim_config.get('params',{})

        loss_config = train_config.get('loss',{})
        loss_package = loss_config.get('package','torch.nn')
        loss_name = loss_config.get('name','MSE')

        # optimizer
        optim_model_file = import_module(optim_package)
        Optim = getattr(optim_model_file,optim_name)
        optimizer = Optim(params = model.parameters(), **optim_params)
        # criterion
        loss_model_file = import_module(loss_package)
        Loss = getattr(loss_model_file,loss_name)
        criterion = Loss()

        print(f'optimizer:{optimizer}\ncriterion:{criterion}')

        gpu_device = get_device()
        model.to(gpu_device)
        progress_bar = tqdm(range(num_epochs), desc="Progress")
        all_loss = list()
        all_test_loss = list()
        best_loss = 1
        best_test_loss = 1
        best_epoch = 0
        cache = Cache(CacheType.MODEL)
        for epoch in progress_bar:
            epoch_loss = 0
            test_loss = 0
            model.train()
            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.to(gpu_device)
                batch_y = batch_y.to(gpu_device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = epoch_loss/len(train_dataloader)
            all_loss.append(avg_loss)
            model.eval()
            for batch_X, batch_y in test_dataloader:
                batch_X = batch_X.to(gpu_device)
                batch_y = batch_y.to(gpu_device)
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
            avg_test_loss = test_loss/len(test_dataloader)
            all_test_loss.append(avg_test_loss)
            if avg_loss <= best_loss and avg_test_loss <= best_test_loss:
                best_loss = avg_loss
                best_test_loss = avg_test_loss
                best_epoch = epoch + 1
                temp_model = copy.deepcopy(model)
                temp_model.eval()
                temp_model.to('cpu')
                cache.save_model(temp_model)
            print(f'Epoch [{epoch + 1}/{num_epochs}],Train Loss: {avg_loss:.4f},Test Loss: {avg_test_loss:.4f}')
            if avg_loss < min_train_loss or avg_test_loss < min_test_loss:
                break
        model.eval()
        model.to('cpu')
        __plot_loss(all_loss,all_test_loss)
        print(f'train end: best loss {best_loss:.4f}, best test loss {best_test_loss:.4f}, Epoch {best_epoch}')
        __print_used_time(start_time)
    except KeyboardInterrupt:
        print(f'training is forcibly terminated: best loss {best_loss:.4f}, best test loss {best_test_loss:.4f}, Epoch {best_epoch}')
        __print_used_time(start_time)
        __plot_loss(all_loss,all_test_loss)
    pass
def __plot_loss(all_loss,all_test_loss):
    plt.title("trend of loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(all_loss,label="train")
    plt.plot(all_test_loss,label="test")
    plt.legend()
    plt.show()
def __print_used_time(start_time):
    runtime = datetime.datetime.now() - start_time
    hours, remainder = divmod(runtime.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"train total time:  {int(hours)}h:{int(minutes)}m:{int(seconds)}s")
