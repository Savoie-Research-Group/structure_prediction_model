def retrieve_drop(drop_path):
    return_list = []
    with open(drop_path) as f:
        for line in f:
            return_list.append(int(float(line.rstrip())))

    return return_list

def retrieve_array(ms_drop_path, ir_drop_path, nmr_drop_path, input_array):
    ms_drop_list = retrieve_drop(ms_drop_path)
    ir_drop_list = retrieve_drop(ir_drop_path)
    nmr_drop_list = retrieve_drop(nmr_drop_path)
    print(ms_drop_list)
    input_array[ms_drop_list,0:999] = 43
    input_array[ir_drop_list,999:1899] = 144
    input_array[nmr_drop_list,1899:] = 244

    return input_array
