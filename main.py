# %%
from util.torch_info import print_info
from singleton_pattern import load_config
load_config.init_config('GREEN_UBFC-PHYs_UBFC-PHys_10.yaml')
print_info()

# %%
from singleton_pattern import load_dataset_loader,load_model,load_data_generator

non_dnn_method_list = load_config.get_non_dnn_method_list()

config = load_config.get_config()
model = load_model.get_model()
tensor_data_generator = load_data_generator.get_tensor_data_generator()
print(f'Method and Preprocessing Name: {config["method"]}')
print(f'Model:\n------------------\n{model}\n------------------')
test_loader = load_dataset_loader.get_test_loader()
print(f'Test Loader: {test_loader.loader_name}')

# %%
import datetime
def train():
    if config['method'] in non_dnn_method_list: 
        print('non train')
        return
    start_time = datetime.datetime.now()
    train_loader = load_dataset_loader.get_train_loader()
    print(f'train_loader: {train_loader.loader_name}')
    model.train(train_loader.load_data())
    runtime = datetime.datetime.now() - start_time
    hours, remainder = divmod(runtime.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"train total time:  {int(hours)}h:{int(minutes)}m:{int(seconds)}s")
# run train
train()

# %%
# test
model.eval()

data = test_loader.load_data()
X,y = tensor_data_generator.generate_tensor_data(data)
print(f'Test Video Size: {data[0].shape}')

# X,y = model.preprocessing(test_df)
# pred_y = model.forward(X)

# 评价指标



