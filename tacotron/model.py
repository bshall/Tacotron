import importlib_resources
import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom


class Tacotron(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.input_size = 2 * decoder["input_size"]
        self.attn_rnn_size = decoder["attn_rnn_size"]
        self.decoder_rnn_size = decoder["decoder_rnn_size"]
        self.n_mels = decoder["n_mels"]
        self.reduction_factor = decoder["reduction_factor"]

        self.encoder = Encoder(**encoder)
        self.decoder_cell = DecoderCell(**decoder)

    @classmethod
    def from_pretrained(cls, url, map_location=None, cfg_path=None):
        """
        Loads the Torch serialized object at the given URL
        (uses torch.hub.load_state_dict_from_url).

        Parameters:
            url (string): URL of the weights to download
            map_location:  a function or a dict specifying how to remap
                storage locations (see torch.load).
            cfg_path (Path): path to config file.
                Defaults to tacotron/config.toml
        """
        cfg_ref = (
            importlib_resources.files("tacotron").joinpath("config.toml")
            if cfg_path is None
            else cfg_path
        )
        with cfg_ref.open() as file:
            cfg = toml.load(file)
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location=map_location)
        model = cls(**cfg["model"])
        model.load_state_dict(checkpoint["tacotron"])
        model.eval()
        return model

    def forward(self, x, mels):
        B, N, T = mels.size()
        mels = mels.unbind(-1)

        h = self.encoder(x)

        alpha = F.one_hot(
            torch.zeros(B, dtype=torch.long, device=x.device), h.size(1)
        ).float()
        c = torch.zeros(B, self.input_size, device=x.device)

        attn_hx = (
            torch.zeros(B, self.attn_rnn_size, device=x.device),
            torch.zeros(B, self.attn_rnn_size, device=x.device),
        )

        rnn1_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        rnn2_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        go_frame = torch.zeros(B, N, device=x.device)

        ys, alphas = [], []
        for t in range(0, T, self.reduction_factor):
            y = mels[t - 1] if t > 0 else go_frame
            y, alpha, c, attn_hx, rnn1_hx, rnn2_hx = self.decoder_cell(
                h, y, alpha, c, attn_hx, rnn1_hx, rnn2_hx
            )
            ys.append(y)
            alphas.append(alpha)

        ys = torch.cat(ys, dim=-1)
        alphas = torch.stack(alphas, dim=2)
        return ys, alphas

    def generate(self, x, max_length=10000, stop_threshold=-0.2):
        """
        Generates a log-Mel spectrogram from text.

        Parameters:
            x (Tensor): The text to synthesize converted to a sequence of symbol ids.
                See `text_to_id`.
            max_length (int): Maximum number of frames to generate.
                Defaults to 10000 frames i.e. 125 seconds.
            stop_threshold (float): If a frame is generated with all values exceeding
                `stop_threshold` then generation is stopped.

        Returns:
            Tensor: a log-Mel spectrogram of the synthesized speech.
        """
        h = self.encoder(x)
        B, T, _ = h.size()

        alpha = F.one_hot(torch.zeros(B, dtype=torch.long, device=x.device), T).float()
        c = torch.zeros(B, self.input_size, device=x.device)

        attn_hx = (
            torch.zeros(B, self.attn_rnn_size, device=x.device),
            torch.zeros(B, self.attn_rnn_size, device=x.device),
        )

        rnn1_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        rnn2_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        go_frame = torch.zeros(B, self.n_mels, device=x.device)

        ys, alphas = [], []
        for t in range(0, max_length, self.reduction_factor):
            y = ys[-1][:, :, -1] if t > 0 else go_frame
            y, alpha, c, attn_hx, rnn1_hx, rnn2_hx = self.decoder_cell(
                h, y, alpha, c, attn_hx, rnn1_hx, rnn2_hx
            )
            if torch.all(y[:, :, -1] > stop_threshold):
                break
            ys.append(y)
            alphas.append(alpha)

        ys = torch.cat(ys, dim=-1)
        alphas = torch.stack(alphas, dim=2)
        return ys, alphas


class DynamicConvolutionAttention(nn.Module):
    def __init__(
        self,
        attn_rnn_size,
        hidden_size,
        static_channels,
        static_kernel_size,
        dynamic_channels,
        dynamic_kernel_size,
        prior_length,
        alpha,
        beta,
    ):
        super(DynamicConvolutionAttention, self).__init__()

        self.prior_length = prior_length
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size

        P = betabinom.pmf(np.arange(prior_length), prior_length - 1, alpha, beta)

        self.register_buffer("P", torch.FloatTensor(P).flip(0))
        self.W = nn.Linear(attn_rnn_size, hidden_size)
        self.V = nn.Linear(
            hidden_size, dynamic_channels * dynamic_kernel_size, bias=False
        )
        self.F = nn.Conv1d(
            1,
            static_channels,
            static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False,
        )
        self.U = nn.Linear(static_channels, hidden_size, bias=False)
        self.T = nn.Linear(dynamic_channels, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, s, alpha):
        p = F.conv1d(
            F.pad(alpha.unsqueeze(1), (self.prior_length - 1, 0)), self.P.view(1, 1, -1)
        )
        p = torch.log(p.clamp_min_(1e-6)).squeeze(1)

        G = self.V(torch.tanh(self.W(s)))
        g = F.conv1d(
            alpha.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=s.size(0),
        )
        g = g.view(s.size(0), self.dynamic_channels, -1).transpose(1, 2)

        f = self.F(alpha.unsqueeze(1)).transpose(1, 2)

        e = self.v(torch.tanh(self.U(f) + self.T(g))).squeeze(-1) + p

        return F.softmax(e, dim=-1)


class PreNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.5,
        fixed=False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.p = dropout
        self.fixed = fixed

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training or self.fixed)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training or self.fixed)
        return x


class BatchNormConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bnorm = nn.BatchNorm1d(output_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        g = torch.sigmoid(x2)
        return g * F.relu(x1) + (1.0 - g) * x


class CBHG(nn.Module):
    def __init__(
        self,
        K,
        input_channels,
        channels,
        projection_channels,
        n_highways,
        highway_size,
        rnn_size,
    ):
        super().__init__()

        self.conv_bank = nn.ModuleList(
            [
                BatchNormConv(input_channels, channels, kernel_size)
                for kernel_size in range(1, K + 1)
            ]
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_projections = nn.Sequential(
            BatchNormConv(K * channels, projection_channels, 3),
            BatchNormConv(projection_channels, input_channels, 3, relu=False),
        )

        self.project = (
            nn.Linear(input_channels, highway_size, bias=False)
            if input_channels != highway_size
            else None
        )

        self.highway = nn.Sequential(
            *[HighwayNetwork(highway_size) for _ in range(n_highways)]
        )

        self.rnn = nn.GRU(highway_size, rnn_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        T = x.size(-1)
        residual = x

        x = [conv(x)[:, :, :T] for conv in self.conv_bank]
        x = torch.cat(x, dim=1)

        x = self.max_pool(x)

        x = self.conv_projections(x[:, :, :T])

        x = x + residual
        x = x.transpose(1, 2)

        if self.project is not None:
            x = self.project(x)

        x = self.highway(x)

        x, _ = self.rnn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_symbols, embedding_dim, prenet, cbhg):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, embedding_dim)
        self.pre_net = PreNet(**prenet)
        self.cbhg = CBHG(**cbhg)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_net(x)
        x = self.cbhg(x.transpose(1, 2))
        return x


def zoneout(prev, current, p=0.1):
    mask = torch.empty_like(prev).bernoulli_(p)
    return mask * prev + (1 - mask) * current


class DecoderCell(nn.Module):
    def __init__(
        self,
        prenet,
        attention,
        input_size,
        n_mels,
        attn_rnn_size,
        decoder_rnn_size,
        reduction_factor,
        zoneout_prob,
    ):
        super(DecoderCell, self).__init__()
        self.zoneout_prob = zoneout_prob

        self.prenet = PreNet(**prenet)
        self.dca = DynamicConvolutionAttention(**attention)
        self.attn_rnn = nn.LSTMCell(
            2 * input_size + prenet["output_size"], attn_rnn_size
        )
        self.linear = nn.Linear(2 * input_size + decoder_rnn_size, decoder_rnn_size)
        self.rnn1 = nn.LSTMCell(decoder_rnn_size, decoder_rnn_size)
        self.rnn2 = nn.LSTMCell(decoder_rnn_size, decoder_rnn_size)
        self.proj = nn.Linear(decoder_rnn_size, n_mels * reduction_factor, bias=False)

    def forward(self, h, y, alpha, c, attn_hx, rnn1_hx, rnn2_hx):
        B, N = y.size()

        y = self.prenet(y)
        attn_h, attn_c = self.attn_rnn(torch.cat((c, y), dim=-1), attn_hx)
        if self.training:
            attn_h = zoneout(attn_hx[0], attn_h, p=self.zoneout_prob)

        alpha = self.dca(attn_h, alpha)

        c = torch.matmul(alpha.unsqueeze(1), h).squeeze(1)

        x = self.linear(torch.cat((c, attn_h), dim=-1))

        rnn1_h, rnn1_c = self.rnn1(x, rnn1_hx)
        if self.training:
            rnn1_h = zoneout(rnn1_hx[0], rnn1_h, p=self.zoneout_prob)
        x = x + rnn1_h

        rnn2_h, rnn2_c = self.rnn2(x, rnn2_hx)
        if self.training:
            rnn2_h = zoneout(rnn2_hx[0], rnn2_h, p=self.zoneout_prob)
        x = x + rnn2_h

        y = self.proj(x).view(B, N, 2)
        return y, alpha, c, (attn_h, attn_c), (rnn1_h, rnn1_c), (rnn2_h, rnn2_c)
