import argparse
import os

from training.model import ModelFactory
from training.train import Trainer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim
from utils.datasets import get_dataset
from utils.helpers import create_result_directory, find_latest_checkpoint, set_seed


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name")
    parser.add_argument("--arch", type=str, help="Model name")
    parser.add_argument("-s", "--seed", type=int, help="Training seed")
    parser.add_argument("--y", type=str, help="Target label")
    parser.add_argument("--name", type=str, help="Seed and n_label of FixMatch")
    parser.add_argument("--method", type=str, help="Training method")
    parser.add_argument("--label_dir", type=str, help="Training method")  # "../FFVAE/label/celeba/new/"
    parser.add_argument("--root", type=str, help="Training method")  # "../FFVAE/data/CelebA/img_align_celeba/"
    parser.add_argument("--rho", type=float, help="Hyper Parameter")
    parser.add_argument("-e", "--epochs", type=int, help="Total training epoch")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--wd", type=float, help="Weight decay")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size in training")
    parser.add_argument("--res", type=str, help="Result directory")
    parser.add_argument("--save_per_epoch", type=int, help="Save model per X epoch")
    parser.add_argument("--test_per_epoch", type=int, help="Test model per X epoch")
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)

    dataset = get_dataset(args.dataset, "train", args.y, name=args.name, method=args.method, label_dir=args.label_dir, root=args.root)
    val_dataset = get_dataset(args.dataset, "val", args.y, name=args.name, method=args.method, label_dir=args.label_dir, root=args.root)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1)
    test_dataset = get_dataset(args.dataset, "test", args.y, name=args.name, method=args.method, label_dir=args.label_dir, root=args.root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1)

    model = ModelFactory().get_model(args.arch, img_size=(64, 64)).float()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    result_dir = create_result_directory(args.res, args)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Result Directory: {result_dir}")
    
    trainer = Trainer(dataset, val_loader, test_loader, args.batch_size, model, optimizer, scheduler, result_dir)
    trainer.train(args.epochs, args.rho, test_per_epoch=args.test_per_epoch, save_per_epoch=args.save_per_epoch)


if __name__ == "__main__":
    args = argparser()
    main(args)
