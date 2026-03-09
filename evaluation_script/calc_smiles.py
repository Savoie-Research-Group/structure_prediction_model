from rdkit import Chem
from beam_search import beam_search

def bs_accuracy(batch, model, tgt_maxlen, idx_to_ch_dict, target_start_token_idx, target_end_token_idx):
    count_top1 = 0
    count_top10 = 0
    beam_size = 10
    source, target = batch
    bs = source.size(dim=0)
    preds = beam_search(model, source, beam_size, tgt_maxlen, target_start_token_idx, target_end_token_idx)
    batch_pred_10 = []
    for i in range(bs):
        target_text = "".join([idx_to_ch_dict[int(_)] for _ in target[i, :]])
        target_text = target_text.replace("PAD_WORD","")
        target_text = target_text.replace("<","")
        target_text = target_text.replace("$","")
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
        if prediction_top_1 == target_text:
            count_top1 = count_top1 + 1
        if target_text in prediction_top_10:
            count_top10 = count_top10 + 1
        prediction_top_10.append(target_text)
        batch_pred_10.append(prediction_top_10)

    return count_top1, count_top10, batch_pred_10

