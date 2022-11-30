import torch.nn.functional as F
import torch
import math

def soft_assignment(x,y,weights):
    miu = torch.mean(x)
    sigma = torch.std(x)
    x_hat = miu + weights*sigma
    x_hat = x_hat.repeat(x.size(0),1)
    x = x[:,None]
    y = y[:,None]
    x_dist = 10*torch.exp(-(x_hat-x)**2/5)
    x_p = F.softmax(x_dist,dim=1)
    x_p = torch.sum(x_p,dim=0)/len(x)
    y_dist = 10 * torch.exp(-(x_hat - y) ** 2 / 5)
    y_p = F.softmax(y_dist, dim=1)
    y_p = torch.sum(y_p,dim=0)/len(x)
    return x_p,y_p

def model_loss(output, disp_gt, mask,args):
    weights = [0.5, 0.5, 0.7, 1.0]

    if args.loss_type == 'smooth_l1':
        losses = smooth_l1(output, disp_gt, weights,mask)
    if args.loss_type == 'KG':
        losses = KG_loss(output, disp_gt, weights,mask,args)
    if args.loss_type == 'UC':
        losses = UC_loss(output, disp_gt, weights,mask,args)
    return losses


def smooth_l1(output, disp_gt, weights,mask):
    disp_ests = output['disp']

    scale = 0
    total_loss = 0
    losses = {}

    for disp_est, weight in zip(disp_ests,weights):
        disp_loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduce=False) # (must small than 1 after norm)
        total_loss += weight * disp_loss.mean()

        losses["avg_epe/{}".format(scale)] = disp_loss.mean()
        losses["std_epe/{}".format(scale)] = torch.std(disp_loss)

        scale += 1

    losses["loss"] = total_loss

    return losses

def KG_loss(output, disp_gt, weights,mask,args):
    disp_ests = output['disp']
    uncert_ests = output['uncert']

    scale = 0
    total_loss = 0
    losses = {}


    for disp_est, uncert_est, weight in zip(disp_ests, uncert_ests, weights):

        disp_loss = torch.abs(disp_est[mask] - disp_gt[mask])
        uncert_loss = torch.exp(uncert_est[mask])  # act: relu, uncert_est is log(sigma)

        mdist_loss = disp_loss / uncert_loss
        log_s = uncert_est[mask]

        if args.inliers > 0:
            if args.mask in ['soft']:
                epe_std = torch.std(disp_loss)
                epe_miu = torch.mean(disp_loss)
                dist_to_miu = torch.abs(disp_loss - epe_miu)
                inliers = (dist_to_miu < args.inliers * epe_std)
            else:
                inliers = (disp_loss < args.inliers)
            pct = torch.mean(inliers.float())*100
            losses["% of inliers/{}".format(scale)] = pct

        else:
            inliers = torch.ones(disp_loss.size(), dtype=torch.bool)


        all_loss = mdist_loss + log_s

        total_loss += weight*(all_loss[inliers].mean())

        losses["std_epe/{}".format(scale)] = torch.std(disp_loss)
        losses["std_s/{}".format(scale)] = torch.std(uncert_loss)
        losses["loss/{}".format(scale)] = all_loss.mean()
        losses["loss_mdist/{}".format(scale)] = mdist_loss.mean()
        losses["loss_logs/{}".format(scale)] = log_s.mean()

        if args.inliers > 0:
            losses["std_epe/inliers/{}".format(scale)] = torch.std(disp_loss[inliers])
            losses["std_s/inliers/{}".format(scale)] = torch.std(uncert_loss[inliers])
            losses["loss_mdist/inliers/{}".format(scale)] = mdist_loss[inliers].mean()
            losses["loss_logs/inliers/{}".format(scale)] = log_s[inliers].mean()

        del all_loss, disp_loss, uncert_loss, mdist_loss, log_s

        scale += 1

    losses["loss"] = total_loss

    del uncert_ests,disp_ests

    return losses

def UC_loss(output, disp_gt, weights,mask,args):
    if args.bin_scale == 'log':
        mark = math.log(6)/math.log(10)
        bins = torch.logspace(0,mark,args.n_bins)-1
    else:
        bins = torch.linspace(0, 5, args.n_bins)

    disp_ests = output['disp']
    uncert_ests = output['uncert']

    scale = 0
    total_loss = 0
    total_inliers = 0
    losses = {}

    for disp_est, uncert_est, weight in zip(disp_ests, uncert_ests, weights):

        disp_loss = torch.abs(disp_est[mask] - disp_gt[mask])
        uncert_loss = torch.exp(uncert_est[mask])  # act: relu, uncert_est is log(sigma)

        mdist_loss = disp_loss / uncert_loss
        log_s = uncert_est[mask]
        kl_loss = torch.zeros(1).float().cuda()

        if disp_est[mask].numel():
            disp_p,uncert_p = soft_assignment(disp_loss,uncert_loss,bins.cuda())
            kl_loss = F.kl_div(uncert_p.log(),disp_p, reduction='sum')

            epe_std = torch.std(disp_loss)
            epe_miu = torch.mean(disp_loss)
            dist_to_miu = torch.abs(disp_loss - epe_miu)

            if args.inliers > 0:
                if args.mask in ['soft']:
                    inliers = (dist_to_miu < args.inliers * epe_std)
                else:
                    inliers = (disp_loss < args.inliers)

            else:
                inliers = torch.ones(disp_loss.size(), dtype=torch.bool)

            pct = torch.mean(inliers.float()) * 100
            total_inliers += pct
            losses["% of inliers/{}".format(scale)] = pct

            if disp_est[mask][inliers].numel():
                losses["std_epe/{}".format(scale)] = torch.std(disp_loss)
                losses["std_epe/inlier/{}".format(scale)] = torch.std(disp_loss[inliers])

                losses["std_s/{}".format(scale)] = torch.std(uncert_loss)
                losses["std_s/inlier/{}".format(scale)] = torch.std(uncert_loss[inliers])

                all_loss = mdist_loss + log_s + kl_loss

                total_loss += weight*(all_loss[inliers].mean())


                losses["loss/{}".format(scale)] = all_loss.mean()
                losses["loss/inlier/{}".format(scale)] = all_loss[inliers].mean()

                losses["loss_mdist/{}".format(scale)] = mdist_loss.mean()
                losses["loss_mdist/inlier/{}".format(scale)] = mdist_loss[inliers].mean()

                losses["loss_logs/{}".format(scale)] = log_s.mean()
                losses["loss_logs/inlier/{}".format(scale)] = log_s[inliers].mean()

                losses["loss_kl/{}".format(scale)] = kl_loss.mean()
                losses["loss_kl/inlier/{}".format(scale)] = kl_loss.mean()

            else:
                print('Lack of inliers!')
                all_loss = mdist_loss + log_s
                total_loss += weight * (all_loss.mean())
                losses["loss/{}".format(scale)] = all_loss.mean()
                losses["loss_mdist/{}".format(scale)] = mdist_loss.mean()
                losses["loss_logs/{}".format(scale)] = log_s.mean()
        else:
            print('Lack of valid pixles!')
            all_loss = mdist_loss + log_s
            total_loss += weight * (all_loss.mean())
            losses["loss/{}".format(scale)] = all_loss.mean()
            losses["loss_mdist/{}".format(scale)] = mdist_loss.mean()
            losses["loss_logs/{}".format(scale)] = log_s.mean()

        del all_loss, disp_loss, uncert_loss, kl_loss, mdist_loss, log_s

        scale += 1

    losses["loss"] = total_loss
    losses['inliers'] = total_inliers/scale

    del uncert_ests,disp_ests

    return losses