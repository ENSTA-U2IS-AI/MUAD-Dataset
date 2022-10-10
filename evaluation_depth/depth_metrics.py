import torch

def compute_errors(gt, pred):
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = torch.sqrt(rms.mean())

    log_rms = (torch.log(gt) - torch.log(pred)) ** 2
    log_rms = torch.sqrt(log_rms.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    err = torch.abs(torch.log10(pred) - torch.log10(gt))
    log10 = torch.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]