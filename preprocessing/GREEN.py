


def generate_tensor_data(df_loaded_data):
    b_size = 1
    
    for index, row in df_loaded_data.iterrows():
        
        print(f'Row {index + 1}: Name={row["Name"]}, Age={row["Age"]}')
    # 压缩

    # 特征提取
    # 线性插值
    # 归一化
    # 返回：
    #   x:tensor,y:tensor
    pass