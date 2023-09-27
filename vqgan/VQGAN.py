import torch.nn as nn

from vqgan.codebook import VectorQuantizerEMA
from vqgan.encoder import ConvolutionalEncoder
from vqgan.decoder import DeconvolutionalDecoder


class VQGAN(nn.Module):
    def __init__(self, cfg, device):
        super(VQGAN, self).__init__()

        self._output_features_filters = (
            cfg.output_features_filters * 3 if cfg.augment_output_features else cfg.output_features_filters
        )
        self._output_features_dim = cfg.output_features_dim
        self._verbose = cfg.verbose

        self._encoder = ConvolutionalEncoder(
            num_hiddens=cfg.num_hiddens,
            num_residual_layers=cfg.num_residual_layers,
            num_residual_hiddens=cfg.num_hiddens,
            use_kaiming_normal=cfg.use_kaiming_normal,
            input_features_type=cfg.input_features_type,
            features_filters=cfg.input_features_filters * 3 if cfg.augment_input_features else cfg.input_features_filters,
            device=device,
            verbose=self._verbose,
        )

        self._pre_vq_conv = nn.Conv1d(in_channels=cfg.num_hiddens, out_channels=cfg.embedding_dim, kernel_size=3, padding=1)

        self._vq = VectorQuantizerEMA(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            commitment_cost=cfg.commitment_cost,
            decay=cfg.decay,  # has to be > 0.0
            device=device,
        )

        self._decoder = DeconvolutionalDecoder(
            in_channels=cfg.embedding_dim,
            out_channels=self._output_features_filters,
            num_hiddens=cfg.num_hiddens,
            num_residual_layers=cfg.num_residual_layers,
            num_residual_hiddens=cfg.num_residual_hiddens,
            use_kaiming_normal=cfg.use_kaiming_normal,
            use_jitter=cfg.use_jitter,
            jitter_probability=cfg.jitter_probability,
            device=device,
            verbose=self._verbose,
        )

        self._device = device
        self._record_codebook_stats = cfg.record_codebook_stats

    @property
    def vq(self):
        return self._vq

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x):
        z = self._encoder(x)

        z = self._pre_vq_conv(z)

        vq_dict = self._vq(z, record_codebook_stats=self._record_codebook_stats)

        reconstructed_x = self._decoder(vq_dict["quantized"])

        input_features_size = x.size(2)
        output_features_size = reconstructed_x.size(2)

        reconstructed_x = reconstructed_x.view(-1, self._output_features_filters, output_features_size)
        reconstructed_x = reconstructed_x[:, :, : -(output_features_size - input_features_size)]

        return (
            reconstructed_x,
            vq_dict["vq_loss"],
            vq_dict["losses"],
            vq_dict["perplexity"],
            vq_dict["encoding_indices"],
            vq_dict["concatenated_quantized"],
        )
