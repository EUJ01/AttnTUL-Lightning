import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.metrics import accuracy_1, accuracy_5


class MolLightningModule(pl.LightningModule):
    """LightningModule for MolNet"""

    def __init__(
        self,
        LocalGcnModel,
        GlobalGcnModel,
        MolModel,
        local_feature,
        local_adj,
        global_feature,
        global_adj,
    ):
        super().__init__()

        self.automatic_optimization = False
        # manual optimization in training step due to having multiple optimisers

        self.LocalGcnModel = LocalGcnModel
        self.GlobalGcnModel = GlobalGcnModel
        self.MolModel = MolModel
        self.local_feature = local_feature
        self.local_adj = local_adj
        self.global_feature = global_feature
        self.global_adj = global_adj
        self.grid_emb = None
        self.traj_emb = None
        self.sum_train_step_losses = 0

    def on_train_epoch_start(self):
        self.sum_train_step_losses = 0

        self.grid_emb = self.LocalGcnModel(
            self.local_feature.to(self.device), self.local_adj.to(self.device)
        )

        self.traj_emb = self.GlobalGcnModel(
            self.global_feature.to(self.device), self.global_adj.to(self.device)
        )

    def on_validation_epoch_start(self):
        self.grid_emb = self.LocalGcnModel(
            self.local_feature.to(self.device), self.local_adj.to(self.device)
        )

        self.traj_emb = self.GlobalGcnModel(
            self.global_feature.to(self.device), self.global_adj.to(self.device)
        )

    def forward(self, input_seq, time_seq, state_seq, input_index):
        return self.MolModel(
            self.grid_emb.to(self.device),
            self.traj_emb.to(self.device),
            input_seq.to(self.device),
            time_seq.to(self.device),
            state_seq.to(self.device),
            input_index.to(self.device),
        )

    def training_step(self, batch, batch_idx):
        input_seq, time_seq, state_seq, input_index, y_true = batch
        y_predict = self(input_seq, time_seq, state_seq, input_index)

        loss = F.nll_loss(y_predict, y_true)
        acc1 = accuracy_1(y_predict, y_true)
        acc5 = accuracy_5(y_predict, y_true)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("acc1_train", acc1, on_epoch=True, on_step=False, prog_bar=True)
        self.log("acc5_train", acc5, on_epoch=True, on_step=False, prog_bar=True)

        self.sum_train_step_losses += loss

        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, time_seq, state_seq, input_index, y_true = batch
        y_predict = self(input_seq, time_seq, state_seq, input_index)

        loss = F.nll_loss(y_predict, y_true)
        acc1 = accuracy_1(y_predict, y_true)
        acc5 = accuracy_5(y_predict, y_true)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("acc1_val", acc1, on_epoch=True, prog_bar=True)
        self.log("acc5_val", acc5, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_seq, time_seq, state_seq, input_index, y_true = batch
        y_predict = self(input_seq, time_seq, state_seq, input_index)

        self.outputs = []
        self.outputs.append({"y_predict": y_predict, "y_true": y_true})

        # return {'y_predict': y_predict, 'y_true': y_true}

    def on_train_epoch_end(self):
        opt1, opt2, opt3 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()
        opt3.zero_grad()

        self.manual_backward(self.sum_train_step_losses)

        opt1.step()
        opt2.step()
        opt3.step()

    def on_test_epoch_end(self):
        y_true_list = []
        y_pred_list = []

        for output in self.outputs:
            y_true = output["y_true"]
            y_pred = output["y_predict"]
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)

        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)

        acc1 = accuracy_1(y_pred, y_true)
        acc5 = accuracy_5(y_pred, y_true)
        self.log("acc1", acc1, on_epoch=True, prog_bar=True)
        self.log("acc5", acc5, on_epoch=True, prog_bar=True)

        y_true = y_true.cpu().numpy()
        y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

        p = precision_score(y_true, y_pred, average="macro")
        r = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        self.log("precision", p, prog_bar=True)
        self.log("recall", r, prog_bar=True)
        self.log("f1 score", f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer_localgcn = torch.optim.Adam(
            self.LocalGcnModel.parameters(), lr=0.001, weight_decay=5e-4
        )
        optimizer_globalgcn = torch.optim.Adam(
            self.GlobalGcnModel.parameters(), lr=0.001, weight_decay=5e-4
        )
        optimizer_mol = torch.optim.Adam(self.MolModel.parameters(), lr=0.001)

        return [optimizer_localgcn, optimizer_globalgcn, optimizer_mol]
