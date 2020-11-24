import argparse
from functools import partial
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import toml
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as samplers
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tacotron import BucketBatchSampler, Tacotron, TTSDataset, pad_collate

matplotlib.use("Agg")


def save_checkpoint(tacotron, optimizer, scaler, scheduler, step, checkpoint_dir):
    state = {
        "tacotron": tacotron.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(tacotron, optimizer, scaler, scheduler, load_path):
    print(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path)
    tacotron.load_state_dict(checkpoint["tacotron"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["step"]


def log_alignment(alpha, y, cfg, writer, global_step):
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(alpha, vmin=0, vmax=0.6, origin="lower")
    plt.xlabel("Decoder steps")
    plt.ylabel("Encoder steps")
    writer.add_figure("alignment", fig, global_step)

    fig, ax = plt.subplots(figsize=(20, 4))
    librosa.display.specshow(
        cfg["top_db"] * y + cfg["ref_db"],
        x_axis="time",
        y_axis="mel",
        sr=cfg["sr"],
        hop_length=cfg["hop_length"],
        cmap="viridis",
        ax=ax,
    )
    writer.add_figure("mel", fig, global_step)


def train_model(args):
    with open("tacotron/config.toml") as file:
        cfg = toml.load(file)

    tensorboard_path = Path("tensorboard") / args.checkpoint_dir
    checkpoint_dir = Path(args.checkpoint_dir)
    writer = SummaryWriter(tensorboard_path)

    tacotron = Tacotron(**cfg["model"]).cuda()
    optimizer = optim.Adam(tacotron.parameters(), lr=cfg["train"]["optimizer"]["lr"])
    scaler = amp.GradScaler()
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg["train"]["scheduler"]["milestones"],
        gamma=cfg["train"]["scheduler"]["gamma"],
    )

    if args.resume is not None:
        global_step = load_checkpoint(
            tacotron=tacotron,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            load_path=args.resume,
        )
    else:
        global_step = 0

    root_path = Path(args.dataset_dir)
    text_path = Path(args.text_path)

    dataset = TTSDataset(root_path, text_path)
    sampler = samplers.RandomSampler(dataset)
    batch_sampler = BucketBatchSampler(
        sampler=sampler,
        batch_size=cfg["train"]["batch_size"],
        drop_last=True,
        sort_key=dataset.sort_key,
        bucket_size_multiplier=cfg["train"]["bucket_size_multiplier"],
    )
    collate_fn = partial(
        pad_collate, reduction_factor=cfg["model"]["decoder"]["reduction_factor"]
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=cfg["train"]["n_workers"],
        pin_memory=True,
    )

    n_epochs = cfg["train"]["n_steps"] // len(loader) + 1
    start_epoch = global_step // len(loader) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_loss = 0

        for i, (mels, texts, mel_lengths, text_lengths, attn_flag) in enumerate(
            tqdm(loader), 1
        ):
            mels, texts = mels.cuda(), texts.cuda()

            optimizer.zero_grad()

            with amp.autocast():
                ys, alphas = tacotron(texts, mels)
                loss = F.l1_loss(ys, mels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(tacotron.parameters(), cfg["train"]["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            average_loss += (loss.item() - average_loss) / i

            if global_step % cfg["train"]["checkpoint_interval"] == 0:
                save_checkpoint(
                    tacotron=tacotron,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    step=global_step,
                    checkpoint_dir=checkpoint_dir,
                )

            if attn_flag:
                index = attn_flag[0]
                alpha = alphas[index, : text_lengths[index], : mel_lengths[index] // 2]
                alpha = alpha.detach().cpu().numpy()

                y = ys[index, :, :].detach().cpu().numpy()
                log_alignment(alpha, y, cfg["preprocess"], writer, global_step)

        writer.add_scalar("loss", average_loss, global_step)
        print(f"epoch {epoch} : loss {average_loss:.4f} : {scheduler.get_last_lr()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Tacotron with dynamic convolution attention."
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Path to the directory where model checkpoints will be saved",
    )
    parser.add_argument(
        "text_path",
        help="Path to the dataset transcripts",
    )
    parser.add_argument(
        "dataset_dir",
        help="Path to the preprocessed data directory",
    )
    parser.add_argument(
        "--resume",
        help="Path to the checkpoint to resume from",
    )
    args = parser.parse_args()
    train_model(args)
