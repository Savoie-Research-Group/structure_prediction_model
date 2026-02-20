import numpy as np
def random_choice(array_input, drop_rate):
    ms_drop_list = []
    ir_drop_list = []
    nmr_drop_list = []
    for i in range(array_input.shape[0]):
        rand1 = float(np.random.rand(1))
        rand2 = float(np.random.rand(1))
        rand3 = float(np.random.rand(1))
        if i % 3 == 0:
            if rand1 < drop_rate:
                if rand2 < drop_rate:
                    array_input[i,0:999] = np.full((999),43)
                    array_input[i,999:1899] = np.full((900),144)
                    ms_drop_list.append(i)
                    ir_drop_list.append(i)
                else:
                    if rand3 < drop_rate:
                        array_input[i,1899:] = np.full((993),244)
                        nmr_drop_list.append(i)
            else:
                if rand2 < drop_rate:
                    array_input[i,999:1899] = np.full((900),144)
                    ir_drop_list.append(i)
                else:
                    continue
                if rand3 < drop_rate:
                    array_input[i,1899:] = np.full((993),244)
                    nmr_drop_list.append(i)
                else:
                    continue
        if i % 3 == 1:
            if rand1 < drop_rate:
                if rand2 < drop_rate:
                    array_input[i,1899:] = np.full((993),244)
                    array_input[i,999:1899] = np.full((900),144)
                    nmr_drop_list.append(i)
                    ir_drop_list.append(i)
                else:
                    if rand3 < drop_rate:
                        array_input[i,0:999] = np.full((999),43)
                        ms_drop_list.append(i)
            else:
                if rand2 < drop_rate:
                    array_input[i,1899:] = np.full((993),244)
                    nmr_drop_list.append(i)
                else:
                    continue
                if rand3 < drop_rate:
                    array_input[i,0:999] = np.full((999),43)
                    ms_drop_list.append(i)
                else:
                    continue
        if i % 3 == 2:
            if rand1 < drop_rate:
                if rand2 < drop_rate:
                    array_input[i,1899:] = np.full((993),244)
                    array_input[i,0:999] = np.full((999),43)
                    nmr_drop_list.append(i)
                    ms_drop_list.append(i)
                else:
                    if rand3 < drop_rate:
                        array_input[i,999:1899] = np.full((900),144)
                        ir_drop_list.append(i)
            else:
                if rand2 < drop_rate:
                    array_input[i,0:999] = np.full((999),43)
                    ms_drop_list.append(i)
                else:
                    continue
                if rand3 < drop_rate:
                    array_input[i,999:1899] = np.full((900),144)
                    ir_drop_list.append(i)
                else:
                    continue
    return ms_drop_list, ir_drop_list, nmr_drop_list, array_input
