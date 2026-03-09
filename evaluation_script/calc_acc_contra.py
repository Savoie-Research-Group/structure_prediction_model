from rdkit import Chem
from beam_search import beam_search

def bs_accuracy(batch, model, tgt_maxlen, idx_to_ch_dict, target_start_token_idx, target_end_token_idx):
    count1_top1 = 0
    count1_top10 = 0
    count2_top1 = 0
    count2_top10 = 0
    beam_size = 10
    source, target = batch
    bs = source.size(dim=0)
    preds = beam_search(model, source, beam_size, tgt_maxlen, target_start_token_idx, target_end_token_idx)
    for i in range(bs):
        target_text1 = "".join([idx_to_ch_dict[int(_)] for _ in target[i, 0:67]])
        target_text1 = target_text1.replace("PAD_WORD","")
        target_text1 = target_text1.replace("<","")
        target_text1 = target_text1.replace("$","")
        target_text2 = "".join([idx_to_ch_dict[int(_)] for _ in target[i, 67:]])
        target_text2 = target_text2.replace("PAD_WORD","")
        target_text2 = target_text2.replace("<","")
        target_text2 = target_text2.replace("$","")
# target_text2: contradictory input

        preds[i].sort(key = lambda x:x[1])
        pred_dict = {}
        for j in range(beam_size):
            pred_dict["prediction_{}".format(j)] = ""
            for idx in preds[i][-1-j][0]:
                pred_dict["prediction_{}".format(j)] += idx_to_ch_dict[idx]
                if idx == target_end_token_idx:
                    break
            pred_dict["prediction_{}".format(j)] = pred_dict["prediction_{}".format(j)].replace("<","")
            pred_dict["prediction_{}".format(j)] = pred_dict["prediction_{}".format(j)].replace("$","")
            try:
                pred_dict["prediction_{}".format(j)] = Chem.CanonSmiles(pred_dict["prediction_{}".format(j)])
            except:
                pred_dict["prediction_{}".format(j)] = "XXX"

        prediction_top_1 = pred_dict["prediction_0"]
        prediction_top_10 = []
        for k in range(10):
            prediction_top_10 += [pred_dict["prediction_{}".format(k)]]
        if prediction_top_1 == target_text1:
            count1_top1 = count1_top1 + 1
        if target_text1 in prediction_top_10:
            count1_top10 = count1_top10 + 1
        if prediction_top_1 == target_text2:
            count2_top1 = count2_top1 + 1
        if target_text2 in prediction_top_10:
            count2_top10 = count2_top10 + 1
# uncomment it if you want to print out smiles results
#        with open('test-smiles.txt','a') as f:
 #           for mol in prediction_top_10:
  #              f.write(mol)
   #             f.write(';')
    #        f.write(target_text)
     #       f.write('\n')
    return count1_top1, count1_top10, count2_top1, count2_top10

