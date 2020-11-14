import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.utils.data.sampler as samplers
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

import hydra
import hydra.utils as utils
from tqdm import tqdm

from pathlib import Path

from tacotron import Tacotron, TTSDataset, BucketBatchSampler, pad_collate


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
        cfg.top_db * y + cfg.ref_db,
        x_axis="time",
        y_axis="mel",
        sr=cfg.sr,
        hop_length=cfg.hop_length,
        cmap="viridis",
        ax=ax,
    )
    writer.add_figure("mel", fig, global_step)


@hydra.main(config_path="tacotron/config", config_name="train")
def train_model(cfg):
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)

    tacotron = Tacotron(**cfg.model).cuda()
    optimizer = optim.Adam(tacotron.parameters(), lr=cfg.train.optimizer.lr)
    scaler = amp.GradScaler()
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg.train.scheduler.milestones,
        gamma=cfg.train.scheduler.gamma,
    )

    if cfg.resume:
        resume_path = utils.to_absolute_path(cfg.resume)
        global_step = load_checkpoint(
            tacotron=tacotron,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            load_path=resume_path,
        )
    else:
        global_step = 0

    root_path = Path(utils.to_absolute_path(cfg.dataset_dir))
    text_path = Path(utils.to_absolute_path(cfg.text_path))

    dataset = TTSDataset(root_path, text_path)
    sampler = samplers.RandomSampler(dataset)
    batch_sampler = BucketBatchSampler(
        sampler=sampler,
        batch_size=cfg.train.batch_size,
        drop_last=True,
        sort_key=dataset.sort_key,
        bucket_size_multiplier=cfg.train.bucket_size_multiplier,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=pad_collate,
        num_workers=cfg.train.n_workers,
        pin_memory=True,
    )

    n_epochs = cfg.train.n_steps // len(loader) + 1
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
            grad_norm = clip_grad_norm_(tacotron.parameters(), cfg.train.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            average_loss += (loss.item() - average_loss) / i

            if global_step % cfg.train.checkpoint_interval == 0:
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
                log_alignment(alpha, y, cfg.preprocess, writer, global_step)

        writer.add_scalar("loss", average_loss, global_step)
        print(
            f"epoch {epoch} : average loss {average_loss:.4f} : {scheduler.get_last_lr()}"
        )


if __name__ == "__main__":
    train_model()
