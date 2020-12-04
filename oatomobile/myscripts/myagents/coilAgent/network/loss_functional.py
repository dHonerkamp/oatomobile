from torch.nn import functional as F
import torch

from oatomobile.datasets.carla import DirectionsEnum


def normalize(x, dim):
    x_normed = x / x.max(dim, keepdim=True)[0]
    return x_normed



def weight_decay_l1(loss, model, intention_factors, alpha, gating):

    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(torch.abs(w)), wdecay)

    if intention_factors is not None:

        intention, _ = torch.min(intention_factors, 1)
        intention = (1. > intention).float()
        if gating == 'hard':
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0]/torch.sum(intention)

        elif gating == 'easy':
            wdecay = wdecay * torch.sum(intention)/intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss


def weight_decay_l2(loss, model, intention_factors, alpha, gating):

    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(w**2), wdecay)

    if intention_factors is not None:

        intention, _ = torch.min(intention_factors, 1)
        intention = (1. > intention).float()
        if gating == 'hard':
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0]/torch.sum(intention)

        elif gating == 'easy':
            wdecay = wdecay * torch.sum(intention)/intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss


def compute_branches_masks(controls, number_targets):
    """
        Args
            controls
            the control values that have the following structure
            command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go straight
            size of targets:
            How many targets is produced by the network so we can produce the masks properly
        Returns
            a mask to have the loss function applied
            only on over the correct branch.
    """

    """ A vector with a mask for each of the control branches"""
    controls_masks = []

    # when command = 2, branch 1 (follow lane) is activated
    for command in DirectionsEnum:
        controls_bool = (controls == command)
        controls_bool = torch.tensor(controls_bool, dtype=torch.float32).cuda()
        controls_bool = torch.cat([controls_bool] * number_targets, 1)
        controls_masks.append(controls_bool)
    return controls_masks

def l2_loss(params):
    """
        Functional LOSS L2
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) - 1):
        loss_branches_vec.append(((params['branches'][i] - params['targets']) ** 2
                                  * params['controls_mask'][i])
                                  * params['branch_weights'][i])
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append((params['branches'][-1] - params['inputs']) ** 2
                             * params['branch_weights'][-1])
    return loss_branches_vec, {}


def l1_loss(params):
    """
        Functional LOSS L1
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) -1):
        loss_branches_vec.append(torch.abs((params['branches'][i] - params['targets'])
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append(torch.abs(params['branches'][-1] - params['inputs'])
                             * params['branch_weights'][-1])
    return loss_branches_vec, {}

