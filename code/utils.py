import torch
import matplotlib.pyplot as plt


def accuracy_1(pred, targ):
    """[Used to calculate trajectory links acc@1]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@1]
    """
    pred = torch.max(torch.log_softmax(pred, dim=1), 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac


def accuracy_5(pred, targ):
    """[Used to calculate trajectory links acc@5]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@5]
    """
    pred = torch.topk(torch.log_softmax(pred, dim=1), k=5,
                      dim=1, largest=True, sorted=True)[1]
    ac = (torch.tensor([t in p for p, t in zip(pred, targ)]
                       ).float()).sum().item() / targ.size()[0]
    return ac


def loss_plot(train_loss, val_loss):
    """[Function used to plot the loss curve]

    Args:
        train_loss ([list]): [Loss list of training sets]
        val_loss ([list]): [Loss list of Validation sets]
    """
    plt.switch_backend('agg')
    plt.plot(range(len(train_loss)), train_loss, label='Train loss', linewidth=2,
             color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_loss)), val_loss, label='Validation loss',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss-fig')
    plt.legend()
    plt.savefig('../log/Loss.png')


def loss_with_earlystop_plot(avg_train_losses, avg_valid_losses):
    """[Function used to plot the loss curve and early stop line]

    Args:
        train_loss ([list]): [Loss list of training sets]
        val_loss ([list]): [Loss list of Validation sets]
    """
    # visualize the loss as the network trained
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(avg_train_losses)+1),
             avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1),
             avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')

    # plt.ylim(0, 10) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('../log/early_stop_loss.png')


def acc_plot(train_acc1, val_acc1, train_acc5, val_acc5):
    """[Accuracy curve drawing function]

    Args:
        train_acc1 ([list]): [Training set acc@1 list]
        val_acc1 ([list]): [Validation set acc@1 list]
        train_acc5 ([list]): [Training set acc@5 list]
        val_acc5 ([list]): [Validation set acc@5 list]
    """
    plt.switch_backend('agg')
    plt.figure(figsize=(30, 10), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(range(len(train_acc1)), train_acc1, label='Train Acc@1',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_acc1)), val_acc1, label='Validation Acc@1',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Acc1-fig')
    plt.legend()
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(range(len(train_acc5)), train_acc5, label='Train Acc@5',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_acc5)), val_acc5, label='Validation Acc@5',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Acc5-fig')
    plt.legend()
    plt.savefig('../log/Acc.png')


def macro_plot(train_macro_p, val_macro_p, train_macro_r, val_macro_r, train_macro_f1, val_macro_f1):
    """[macro-p, macro-r and macro-f1 curve drawing of training set and verification set]

    Args:
        train_macro_p ([list]): [Training set macro_p list]
        val_macro_p ([list]): [Validation set macro_p list]
        train_macro_r ([list]): [Training set macro_r list]
        val_macro_r ([list]): [Validation set macro_r list]
        train_macro_f1 ([list]): [Training set macro_f1 list]
        val_macro_f1 ([list]): [Validation set macro_f1 list]
    """
    plt.switch_backend('agg')
    plt.figure(figsize=(45, 10), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(131)
    plt.plot(range(len(train_macro_p)), train_macro_p, label='Train Acc@1',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_macro_p)), val_macro_p, label='Validation Acc@1',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Macro-P value')
    plt.title('Macro-P fig')
    plt.legend()
    ax2 = plt.subplot(132)
    plt.plot(range(len(train_macro_r)), train_macro_r, label='Train Acc@5',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_macro_r)), val_macro_r, label='Validation Acc@5',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Macro-R value')
    plt.title('Macro-R fig')
    plt.legend()
    ax3 = plt.subplot(133)
    plt.plot(range(len(train_macro_f1)), train_macro_f1, label='Train macro-F1',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_macro_f1)), val_macro_f1, label='Validation macro-F1',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1 value')
    plt.title('Macro_F1 fig')
    plt.legend()
    plt.savefig('../log/Macro.png')
