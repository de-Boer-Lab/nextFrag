import torch, csv, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
from models.model_utils import load_model
from .dataloader import prepare_dataloader
from .utils import _forward
from nextFrag.config import get_project_root

MODULE_DIR = Path(__file__).parent

def load_ground_truth(filename: str | Path) -> np.ndarray:
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    expressions = [float(line[1]) for line in lines]
    return np.array(expressions)

def average_fwd_rev_pred(data: np.ndarray) -> np.ndarray:
    num_samples=len(data)//2
    return (data[:num_samples]+data[num_samples:])/2

def load_promoter_class_indices(file_path):
    df = pd.read_csv(file_path)
    if 'pos' in df.columns:
        return np.unique(np.array(df['pos']))
    elif 'alt_pos' in df.columns and 'ref_pos' in df.columns:
        SNVs_alt = list(df['alt_pos'])
        SNVs_ref = list(df['ref_pos'])
        return list(set(list(zip(SNVs_alt, SNVs_ref))))

def calculate_correlations(index_list, expressions, GROUND_TRUTH_EXP):
    PRED_DATA = OrderedDict()
    GROUND_TRUTH = OrderedDict()

    for j in index_list:
        PRED_DATA[str(j)] = float(expressions[j])
        GROUND_TRUTH[str(j)] = float(GROUND_TRUTH_EXP[j])

    pearson = pearsonr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]
    spearman = spearmanr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]

    return pearson, spearman

def calculate_diff_correlations(pair_list, expressions, GROUND_TRUTH_EXP):
    Y_pred_selected = []
    expressions_selected = []

    for pair in pair_list:
        ref, alt = pair[0], pair[1]
        Y_pred_selected.append(expressions[alt] - expressions[ref])
        expressions_selected.append(GROUND_TRUTH_EXP[alt] - GROUND_TRUTH_EXP[ref])

    Y_pred_selected = np.array(Y_pred_selected)
    expressions_selected = np.array(expressions_selected)

    pearson = pearsonr(expressions_selected, Y_pred_selected)[0]
    spearman = spearmanr(expressions_selected, Y_pred_selected)[0]

    return pearson, spearman

def evaluate_yeast_predictions(expressions, result_file: str | Path):
    expressions = np.array(expressions)
    data_dir = MODULE_DIR.parent / 'data' / 'yeast'
    subset_ids_dir = data_dir / 'test_subset_ids'

    GROUND_TRUTH_EXP = load_ground_truth(data_dir / 'test.txt')
    # Load indices for different promoter classes
    high = load_promoter_class_indices(subset_ids_dir / 'high_exp_seqs.csv')
    low = load_promoter_class_indices(subset_ids_dir / 'low_exp_seqs.csv')
    yeast = load_promoter_class_indices(subset_ids_dir / 'yeast_seqs.csv')
    random = load_promoter_class_indices(subset_ids_dir / 'all_random_seqs.csv')
    challenging = load_promoter_class_indices(subset_ids_dir / 'challenging_seqs.csv')
    SNVs = load_promoter_class_indices(subset_ids_dir / 'all_SNVs_seqs.csv')
    motif_perturbation = load_promoter_class_indices(subset_ids_dir / 'motif_perturbation_seqs.csv')
    motif_tiling = load_promoter_class_indices(subset_ids_dir / 'motif_tiling_seqs.csv')

    final_all = list(range(len(GROUND_TRUTH_EXP)))

    # Calculate correlations
    pearson, spearman = calculate_correlations(final_all, expressions, GROUND_TRUTH_EXP)
    high_pearson, high_spearman = calculate_correlations(high, expressions, GROUND_TRUTH_EXP)
    low_pearson, low_spearman = calculate_correlations(low, expressions, GROUND_TRUTH_EXP)
    yeast_pearson, yeast_spearman = calculate_correlations(yeast, expressions, GROUND_TRUTH_EXP)
    random_pearson, random_spearman = calculate_correlations(random, expressions, GROUND_TRUTH_EXP)
    challenging_pearson, challenging_spearman = calculate_correlations(challenging, expressions, GROUND_TRUTH_EXP)

    # Calculate difference correlations
    SNVs_pearson, SNVs_spearman = calculate_diff_correlations(SNVs, expressions, GROUND_TRUTH_EXP)
    motif_perturbation_pearson, motif_perturbation_spearman = calculate_diff_correlations(motif_perturbation, expressions, GROUND_TRUTH_EXP)
    motif_tiling_pearson, motif_tiling_spearman = calculate_diff_correlations(motif_tiling, expressions, GROUND_TRUTH_EXP)

    # Calculate scores
    pearsons_score = (pearson**2 + 0.3 * high_pearson**2 + 0.3 * low_pearson**2 + 0.3 * yeast_pearson**2 + 
                    0.3 * random_pearson**2 + 0.5 * challenging_pearson**2 + 1.25 * SNVs_pearson**2 + 
                    0.3 * motif_perturbation_pearson**2 + 0.4 * motif_tiling_pearson**2) / 4.65


    spearmans_score = (spearman + 0.3 * high_spearman + 0.3 * low_spearman + 0.3 * yeast_spearman 
                    + 0.3 * random_spearman + 0.5 * challenging_spearman + 1.25 * SNVs_spearman
                    + 0.3 * motif_perturbation_spearman + 0.4 * motif_tiling_spearman) / 4.65

    # Write results
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        f.write(f'Pearson Score\t{pearsons_score}\n')
        f.write(f'all r\t{pearson}\n')
        f.write(f'high r\t{high_pearson}\n')
        f.write(f'low r\t{low_pearson}\n')
        f.write(f'yeast r\t{yeast_pearson}\n')
        f.write(f'random r\t{random_pearson}\n')
        f.write(f'challenging r\t{challenging_pearson}\n')
        f.write(f'SNVs r\t{SNVs_pearson}\n')
        f.write(f'motif perturbation r\t{motif_perturbation_pearson}\n')
        f.write(f'motif tiling r\t{motif_tiling_pearson}\n')

def eval_human_model(arch: str, 
                     model_path: str | Path = None, 
                     out_file: str | Path = None, 
                     al_strategy: str = None,
                     round_num: int = None, 
                     seed: int = None, 
                     batch_size: int=2048):
    test_path_ID = "data/human/demo_test.txt"
    test_path_OOD = "data/human/demo_test.txt"
    test_path_SNV = "data/human/demo_test_snv.txt"
    seqsize = 200
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ID_dl = prepare_dataloader(test_path_ID, 
                                    seqsize=seqsize, 
                                    dset='human',
                                    batch_size=batch_size,
                                    shuffle=False)
    test_OOD_dl = prepare_dataloader(test_path_OOD, 
                                    seqsize=seqsize, 
                                    dset='human',
                                    batch_size=batch_size,
                                    shuffle=False)
    test_SNV_dl = prepare_dataloader(test_path_SNV, 
                                    seqsize=seqsize, 
                                    dset='human',
                                    batch_size=batch_size,
                                    shuffle=False)
    if model_path is not None:
        model=load_model(path=model_path,dataset='human',arch=arch)
    else: # model path inferred from arch, al_strategy, etc
        model=load_model(dataset='human',
                         arch=arch,
                         al_strategy=al_strategy,
                         seed=seed,
                         round_num=round_num)
    model.to(device).eval()

    all_model_predictions=[]
    final_result=[]
    for dl in [test_ID_dl, test_OOD_dl, test_SNV_dl]:
        with torch.inference_mode():
            predictions=[]
            for batch in dl:
                X = batch["x"].to(device)
                predictions.append(_forward(model, X).cpu().numpy())
        all_preds=np.concatenate(predictions,axis=0)
        all_preds=np.squeeze(all_preds)
        all_preds = average_fwd_rev_pred(data=all_preds)
        all_model_predictions.append(all_preds)
    
    for i, gt_path in enumerate([test_path_ID, test_path_OOD]):
        gt=load_ground_truth(gt_path)
        final_result.append(pearsonr(all_model_predictions[i],gt)[0])
    
    snv_df = pd.read_csv(test_path_SNV,sep='\t',header=None)
    snv_length = len(snv_df)//2

    snv_expr = np.array(snv_df[1]).reshape((snv_length,2))
    preds=all_model_predictions[2].reshape((snv_length,2))
    snv_gt=snv_expr[:,1]-snv_expr[:,0]
    snv_pred = preds[:,1]-preds[:,0]
    final_result.append(pearsonr(snv_pred,snv_gt)[0])

    if out_file is not None:
        result_file = out_file
    else:
        result_file = get_project_root() / 'human' / f'round_{round_num}' / al_strategy / f'{arch}_{seed}' / 'model' / 'results.txt'
    with open(result_file, 'w') as f:
        f.write(f"ID\t{final_result[0]}\n")
        f.write(f"OOD\t{final_result[1]}\n")
        f.write(f"SNV\t{final_result[2]}\n")

def eval_yeast_model(arch: str,
                     model_path: str | Path = None, 
                     out_file: str | Path = None, 
                     al_strategy: str = None, 
                     seed: int = None, 
                     round_num: int = None, 
                     batch_size: int=2048):
    test_data_path = "data/yeast/test.txt" 
    seqsize = 150
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dl = prepare_dataloader(test_data_path, 
                                seqsize=seqsize, 
                                dset='yeast',
                                batch_size=batch_size,
                                shuffle=False)
    
    if model_path is not None:
        model=load_model(dataset='yeast',arch=arch,path=model_path)
    else: # model path inferred from arch, al_strategy, etc
        model=load_model(dataset='yeast',
                        arch=arch,
                        al_strategy=al_strategy,
                        seed=seed,
                        round_num=round_num)
    model.to(device).eval()

    with torch.inference_mode():
        predictions=[]
        for batch in test_dl:
            X = batch["x"].to(device)
            predictions.append(_forward(model, X).cpu().numpy())
    all_preds=np.concatenate(predictions,axis=0)
    all_preds=np.squeeze(all_preds)
    result = average_fwd_rev_pred(data=all_preds)

    if out_file is not None:
        result_file = out_file
    else:
        result_file = get_project_root() / 'yeast' / f'round_{round_num}' / al_strategy / f'{arch}_{seed}' / 'model' / 'results.txt'
    evaluate_yeast_predictions(result,result_file=result_file)

def eval_model(dataset: str, 
               arch: str, 
               model_path: str | Path = None,
               out_file: str | Path = None,
               al_strategy: str = None, 
               round_num: int = None, 
               seed: int = None, 
               batch_size: int=2048):
    '''
    expects either model_path or (al_strategy AND round_num AND seed)
    '''
    match dataset:
        case 'human':
            eval = eval_human_model
        case 'yeast':
            eval = eval_yeast_model

    if model_path is not None:
        return eval(arch=arch,model_path=model_path,out_file=out_file,batch_size=batch_size)
    else: # infer model path from other params
        return eval(arch=arch,al_strategy=al_strategy,round_num=round_num,seed=seed,batch_size=batch_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("--al_strategy",type=str)
    parser.add_argument("--round",type=int)
    parser.add_argument("--seed",type=int)
    args = parser.parse_args()


    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
    model_dir = get_project_root() / args.dataset / f'round_{args.round}' / args.al_strategy / f'{args.arch}_{args.seed}' / 'model'
    return eval_model(model_path=model_dir / 'model_best.pth',
                      out_file= model_dir / 'results.txt',
                      dataset=args.dataset,
                      arch=args.arch)
   
if __name__=="__main__":
    main()
