import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

MODEL_FILE = "model.pt"
LOG_FILE = "log.csv"


class Trainer:
    def __init__(
        self,
        dataset,
        val_loader,
        test_loader,
        batch_size,
        model,
        optimizer,
        scheduler,
        result_dir,
    ):
        y1a1_loader = DataLoader(
            dataset,
            batch_size=batch_size // 4,
            num_workers=1,
            sampler=torch.utils.data.SubsetRandomSampler(dataset.idx_y1a1),
        )
        y1a0_loader = DataLoader(
            dataset,
            batch_size=batch_size // 4,
            num_workers=1,
            sampler=torch.utils.data.SubsetRandomSampler(dataset.idx_y1a0),
        )
        y0a1_loader = DataLoader(
            dataset,
            batch_size=batch_size // 4,
            num_workers=1,
            sampler=torch.utils.data.SubsetRandomSampler(dataset.idx_y0a1),
        )
        y0a0_loader = DataLoader(
            dataset,
            batch_size=batch_size // 4,
            num_workers=1,
            sampler=torch.utils.data.SubsetRandomSampler(dataset.idx_y0a0),
        )
        self.minimum = min(
            min(len(y1a1_loader), len(y1a0_loader)),
            min(len(y0a1_loader), len(y0a0_loader)),
        )
        self.batch_size = batch_size
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.dic = {
            0: {0: y0a0_loader, 1: y0a1_loader},
            1: {0: y1a0_loader, 1: y1a1_loader},
        }
        self.loaders = {
            0: {0: y0a0_loader.__iter__(), 1: y0a1_loader.__iter__()},
            1: {0: y1a0_loader.__iter__(), 1: y1a1_loader.__iter__()},
        }

        self.q = torch.zeros(2, 2) + 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_f = nn.CrossEntropyLoss()

        self.result_dir = result_dir
        self.log = pd.DataFrame([], columns=["epoch", "split", "loss", "acc", "dca"])

    def train(self, epochs, rho, test_per_epoch=10, save_per_epoch=10):
        for epoch in range(epochs):
            eta = 1 - (epoch + 1) / epochs
            loss_01 = torch.zeros([2, 2])
            dcas = torch.zeros([2, 2])
            count = torch.zeros([2, 2])

            for _ in range(self.minimum):
                total_bce_loss = 0.0
                imgs, labels, senss = None, None, None
                for y, a in itertools.product([0, 1], [0, 1]):
                    try:
                        img, sens, label = next(self.loaders[y][a])
                    except Exception:
                        self.loaders[y][a] = self.dic[y][a].__iter__()
                        img, sens, label = next(self.loaders[y][a])
                    img, sens, label = (
                        img.to(self.device),
                        sens.to(self.device),
                        label.to(self.device),
                    )
                    imgs = torch.cat([imgs, img]) if imgs is not None else img
                    labels = torch.cat([labels, label]) if labels is not None else label
                    senss = torch.cat([senss, sens]) if senss is not None else sens

                out = self.model(imgs)

                for y, a in itertools.product([0, 1], [0, 1]):
                    mask = (senss == a).view(-1) & (labels == y)
                    loss = self.loss_f(out[mask, :], labels[mask].to(torch.int64))
                    total_bce_loss += self.q[y][1 - a] * loss
                    loss_01[y][a] += sum(
                        torch.argmax(out[mask], 1) == labels[mask]
                    ).item()
                    count[y][a] += sum(mask.cpu())

                    dcas[y][a] += sum(torch.argmax(out[mask], 1).cpu() == y)

                total_bce_loss /= 2  # / |Y|

                self.optimizer.zero_grad()
                total_bce_loss.backward()
                self.optimizer.step()

            loss_01 /= count
            dcas /= count
            q_star = self.update_q_star(loss_01, rho)
            self.q = (1 - eta) * self.q + eta * q_star
            acc = sum(sum(loss_01)) / 4
            dca = (abs(dcas[1][0] - dcas[1][1]) + abs(dcas[0][0] - dcas[0][1])) / 2

            self.log = self.log.append(
                {
                    "epoch": epoch + 1,
                    "split": "train",
                    "loss": total_bce_loss.item(),
                    "acc": acc.item(),
                    "dca": dca.item(),
                },
                ignore_index=True,
            )

            self.model.eval()
            val_loss, val_acc, val_dca = self.eval(self.val_loader)

            print(
                f"Epoch {epoch+1}: Loss={total_bce_loss.item():.3f} | "
                + f"Train Acc: {acc*100:.2f} | Train DCA: {dca:.3f} | "
                + f"Val Acc: {val_acc*100:.2f} | Val DCA: {val_dca:.3f}"
            )

            self.log = self.log.append(
                {
                    "epoch": epoch + 1,
                    "split": "val",
                    "loss": val_loss,
                    "acc": val_acc,
                    "dca": val_dca,
                },
                ignore_index=True,
            )

            if (epoch + 1) % test_per_epoch == 0:
                test_loss, test_acc, test_dca = self.eval(self.test_loader)
                self.log = self.log.append(
                    {
                        "epoch": epoch + 1,
                        "split": "test",
                        "loss": test_loss,
                        "acc": test_acc,
                        "dca": test_dca,
                    },
                    ignore_index=True,
                )

            self.scheduler.step()

            if (epoch + 1) % save_per_epoch == 0:
                self.model.cpu()
                save_path = os.path.join(self.result_dir, f"model-{epoch+1}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    save_path,
                )
                self.model.to(self.device)
                self.log.to_csv(os.path.join(self.result_dir, LOG_FILE), index=False)

    def eval(self, data_loader):
        self.model.eval()
        val_loss, val_acc, val_dcas, val_count = (
            0.0,
            0.0,
            torch.zeros([2, 2]),
            torch.zeros([2, 2]),
        )
        with torch.no_grad():
            for img, sens, label in data_loader:
                img, sens, label = (
                    img.to(self.device),
                    sens.to(self.device),
                    label.to(self.device),
                )
                sens = sens.squeeze()
                out = self.model(img)
                pred = torch.argmax(out, 1)
                for y, a in itertools.product([0, 1], [0, 1]):
                    mask = (sens == a) & (label == y)
                    val_count[y][a] += sum(mask.cpu())
                    val_dcas[y][a] += sum(pred[mask].cpu() == y)
                val_loss += self.loss_f(out, label.to(torch.int64))
                val_acc += sum(pred == label) / len(pred)
            val_acc /= len(data_loader)
            val_dcas /= val_count
            val_loss /= len(data_loader)
            val_dca = (
                abs(val_dcas[1][0] - val_dcas[1][1])
                + abs(val_dcas[0][0] - val_dcas[0][1])
            ) / 2

        self.model.train()

        return val_loss.item(), val_acc.item(), val_dca.item()

    def update_q_star(self, loss_01, rho):
        q_star = torch.zeros([2, 2])
        for y, a in itertools.product([0, 1], [0, 1]):
            mu = torch.mean(loss_01[y])
            var = torch.mean((loss_01[y] - mu) ** 2)
            q_star[y][a] = 1 / 2 + np.sqrt(rho / 2) * (
                1e-20 + loss_01[y][a] - loss_01[y].sum() / 2
            ) / (1e-20 + torch.sqrt(2 * var))
        return q_star
