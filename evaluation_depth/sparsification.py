import torch


uncertainty_metrics = ["abs_rel", "rmse", "d1"]
def compute_eigen_errors_v2(gt, pred, metrics=uncertainty_metrics, mask=None, reduce_mean=False):
    """Revised compute_eigen_errors function used for uncertainty metrics, with optional reduce_mean argument and (1-d1) computation
    """
    results = []
    
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    
    if "abs_rel" in metrics:
        abs_rel = (torch.abs(gt - pred) / gt)
        if reduce_mean:
            abs_rel = abs_rel.mean()
        results.append(abs_rel)

    if "rmse" in metrics:
        rmse = (gt - pred) ** 2
        if reduce_mean:
            rmse = torch.sqrt(rmse.mean())
        results.append(rmse)

    if "d1" in metrics:
        d1 = torch.maximum((gt / pred), (pred / gt))
        if reduce_mean:
        
            # invert to get outliers
            d1 = (d1 >= 1.25).to(torch.float).mean()
        results.append(d1)

    return results

def compute_aucs(gt, pred, uncert, intervals=100, device = 'cuda'):
    """Computation of auc metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gt = gt.to(device)
    pred = pred.to(device)
    uncert = uncert.to(device)

    # results dictionaries
    AUSE = {"abs_rel":0, "rmse":0, "d1":0}
    AURG = {"abs_rel":0, "rmse":0, "d1":0}

    # revert order (high uncertainty first)
    uncert = -uncert
    true_uncert = compute_eigen_errors_v2(gt,pred)
    true_uncert = {"abs_rel":-true_uncert[0],"rmse":-true_uncert[1],"d1":-true_uncert[2]}

    # prepare subsets for sampling and for area computation
    quants = [100./intervals*t for t in range(0,intervals)]
    quants = torch.tensor(quants).to(device)/100.
    plotx = [1./intervals*t for t in range(0,intervals+1)]
    plotx = torch.tensor(plotx).to(device)

    # get percentiles for sampling and corresponding subsets
    thresholds = [torch.quantile(uncert, q) for q in quants]
    subs = [(uncert >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    sparse_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=sub,reduce_mean=True)[0] for sub in subs]+[0] for m in uncertainty_metrics }

    # human-readable call
    '''
    sparse_curve =  {"rmse":[compute_eigen_errors_v2(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                     "d1":[compute_eigen_errors_v2(gt,pred,metrics=["d1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                     "abs_rel":[compute_eigen_errors_v2(gt,pred,metrics=["abs_rel"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
    '''
    
    # get percentiles for optimal sampling and corresponding subsets
    opt_thresholds = {m:[torch.quantile(true_uncert[m], q) for q in quants] for m in uncertainty_metrics}
    opt_subs = {m:[(true_uncert[m] >= o) for o in opt_thresholds[m]] for m in uncertainty_metrics}

    # compute sparsification curves for optimal sampling (add 0 for final sampling)
    opt_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=opt_sub,reduce_mean=True)[0] for opt_sub in opt_subs[m]]+[0] for m in uncertainty_metrics}

    # compute metrics for random sampling (equal for each sampling)
    rnd_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=None,reduce_mean=True)[0] for t in range(intervals+1)] for m in uncertainty_metrics}    

    # compute error and gain metrics
    for m in uncertainty_metrics:

        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = torch.trapz(torch.tensor(sparse_curve[m]).to(device), x=plotx) - torch.trapz(torch.tensor(opt_curve[m]).to(device), x=plotx)
        
        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        AURG[m] = rnd_curve[m][0].to(device) - torch.trapz(torch.tensor(sparse_curve[m]).to(device), x=plotx)

    # returns a dictionary with AUSE and AURG for each metric
    # return {m:[AUSE[m], AURG[m]] for m in uncertainty_metrics}
    return AUSE["abs_rel"], AUSE["rmse"], AUSE["d1"], AURG["abs_rel"], AURG["rmse"], AURG["d1"]