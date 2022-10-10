import os
import numpy as np
import argparse

from stream_metrics import StreamSegMetrics, _ECELoss, eval_ood_measure
import av_corrected

import torch
import torch.nn as nn
from torch.utils import data
    


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--gt_root", type=str, default='./datasets/data',
                        help="path to the folder you put the ground truth maps")
    parser.add_argument("--prediction_root", type=str, default='./datasets/data',
                        help="path to the folder you save the predictions")
    parser.add_argument("--val_file_root", type=str, default='val',
                        help="path to the json file that include the information for validation set, check the example file val.odgt")

    return parser



def get_dataset(opts):
    """ Dataset 
    """
    if True:
        val_dst = av_corrected.dataset(gt_dataset=opts.gt_root, pred_dataset=opts.prediction_root, 
                                    root_odgt=opts.val_file_root)
    return  val_dst




def validate(loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ece=[]
    auroc_list=[]
    aupr_list=[]
    fpr_list=[]

    with torch.no_grad():
        for dico, labels, image_pth in loader:

            labels = torch.min(labels, torch.tensor([18], dtype=torch.uint8))
            conf = dico['conf'].to(device)
            pred = dico['pred'].to(device)
            conf = conf.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            targets = labels.numpy()
            metrics.update(targets, pred.numpy())
            
            auroc, aupr, fpr = 0, 0, 0
            auroc, aupr, fpr = eval_ood_measure(conf, labels, image_pth, mask=None)
            '''try:
                auroc, aupr, fpr = eval_ood_measure(conf, labels,image_pth, mask=None)
            except:
                print('pb with image name_img =',image_pth)'''

            ece_out =  ECE.forward(conf, pred, labels)
            ece.append(ece_out.item())
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            fpr_list.append(fpr)

    score = metrics.get_results_iou() # you can also try get_results() to see the other evaluation metrics for semantic segmentation 
    return score, ece, auroc_list, aupr_list, fpr_list


val_batch_size = 1
num_classes = 19

ECE = _ECELoss()


def main():

    opts = get_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using the Device: %s" % device)


    # Setup dataloader
    val_dst = get_dataset(opts) 

    val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=False, drop_last=False)

    # Set up metrics
    metrics = StreamSegMetrics(num_classes)

    val_score, ece, auroc_list, aupr_list, fpr_list  = validate(loader=val_loader, device=device, metrics=metrics)

    print('--------------------------------------------------------------------')
    print(metrics.to_str(val_score))
    print('mean ECE = ', np.mean(np.asarray(ece)))
    print('mean AUROC = ', np.mean(np.asarray(auroc_list)))
    print('mean AUPR = ', np.mean(np.asarray(aupr_list)))
    print('mean FPR = ', np.mean(np.asarray(fpr_list)))

    with open(os.path.join('scores.txt'), 'w') as output_file:
        output_file.write("Mean_IoU: {0}\n".format(val_score['Mean IoU']))
        output_file.write("mECE: {0}\n".format(np.mean(np.asarray(ece))))
        output_file.write("mAUROC: {0}\n".format(np.mean(np.asarray(auroc_list))))
        output_file.write("mAUPR: {0}\n".format(np.mean(np.asarray(aupr_list))))
        output_file.write("mFPR: {0}\n".format(np.mean(np.asarray(fpr_list))))


if __name__ == '__main__':
    main()

