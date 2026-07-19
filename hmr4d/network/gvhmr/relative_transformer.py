import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hmr4d.configs import MainStore, builds

from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module

from hmr4d.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        # x
        output_dim=151,
        ego_output_dim=None,
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        cam_trans_vel_dim=0,
        gravity_dim=0,
        imgseq_dim=1024,
        ego_imgseq_dim=1024,
        ego_head_dim=25,
        ego_hand_dim=18,
        interaction_dim=0,
        cross_view_fusion=False,
        cross_view_heads=4,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # output
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
        dual_head=True,
    ):
        super().__init__()

        # input
        self.output_dim = output_dim
        self.ego_output_dim = ego_output_dim if ego_output_dim is not None else output_dim
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.cam_trans_vel_dim = cam_trans_vel_dim
        self.gravity_dim = gravity_dim
        self.imgseq_dim = imgseq_dim
        self.ego_imgseq_dim = ego_imgseq_dim
        self.ego_head_dim = ego_head_dim
        self.ego_hand_dim = ego_hand_dim
        self.interaction_dim = interaction_dim
        self.cross_view_fusion = cross_view_fusion
        self.cross_view_heads = cross_view_heads

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input (Kp2d)
        # Main token: map d_obs 2 to 32
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )

        self._build_condition_embedder()

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.pred_cam_head = pred_cam_dim > 0  # keep extra_output for easy-loading old ckpt
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)

        self.avgbeta = avgbeta
        self.dual_head = dual_head
        # Optional dual-head (exo) outputs
        self.final_layer_ego = None
        self.pred_cam_head_ego = None
        self.static_conf_head_ego = None
        if self.dual_head:
            self.final_layer_ego = Mlp(self.latent_dim, out_features=self.ego_output_dim)
            if self.pred_cam_head:
                self.pred_cam_head_ego = Mlp(self.latent_dim, out_features=pred_cam_dim)
            if self.static_conf_head:
                self.static_conf_head_ego = Mlp(self.latent_dim, out_features=static_conf_dim)

        if self.cross_view_fusion:
            self.cross_q_norm = nn.LayerNorm(self.latent_dim)
            self.cross_kv_norm = nn.LayerNorm(self.latent_dim)
            self.cross_attn = nn.MultiheadAttention(
                self.latent_dim,
                num_heads=self.cross_view_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_gate = nn.Parameter(torch.zeros(()))


    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.cam_trans_vel_dim > 0:
            self.cam_trans_vel_embedder = nn.Sequential(
                nn.Linear(self.cam_trans_vel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.gravity_dim > 0:
            self.gravity_embedder = nn.Sequential(
                nn.Linear(self.gravity_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )
        if self.ego_imgseq_dim > 0:
            self.ego_imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.ego_imgseq_dim),
                zero_module(nn.Linear(self.ego_imgseq_dim, latent_dim)),
            )
        if self.ego_head_dim > 0:
            self.ego_head_embedder = nn.Sequential(
                nn.LayerNorm(self.ego_head_dim),
                nn.Linear(self.ego_head_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.ego_hand_dim > 0:
            self.ego_hand_embedder = nn.Sequential(
                nn.LayerNorm(self.ego_hand_dim),
                nn.Linear(self.ego_hand_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.interaction_dim > 0:
            self.interaction_embedder = nn.Sequential(
                nn.LayerNorm(self.interaction_dim),
                nn.Linear(self.interaction_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

    def forward(
        self,
        length,
        obs=None,
        f_cliffcam=None,
        f_cam_angvel=None,
        f_cam_trans_vel=None,
        f_gravity_dir=None,
        f_imgseq=None,
        f_ego_imgseq=None,
        f_ego_head=None,
        f_ego_hand=None,
        f_interaction=None,
        cross_obs=None,
        cross_f_cliffcam=None,
        cross_f_cam_angvel=None,
        cross_f_cam_trans_vel=None,
        cross_f_gravity_dir=None,
        cross_f_imgseq=None,
        cross_f_ego_imgseq=None,
    ):
        """
        Args:
            x: None we do not use it
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C), exo / third-person image feature.
            f_ego_imgseq: (B, L, C), ego PV image feature.
            f_ego_head: (B, L, C), compact CPF/head trajectory condition.
            f_ego_hand: (B, L, C), optional hand/gaze condition.
            f_interaction: (B, L, C), optional ego-exo relative interaction condition.
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_noisyobs: (B, L, C), noisy pose observation
            f_cam_angvel: (B, L, 6), Camera angular velocity
            f_cam_trans_vel: (B, L, 3), camera-local translation velocity.
            f_gravity_dir: (B, L, 3), gravity/up direction in camera coordinates.
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        def encode_stream(
            obs_,
            f_cliffcam_,
            f_cam_angvel_,
            f_cam_trans_vel_=None,
            f_gravity_dir_=None,
            f_imgseq_=None,
            f_ego_imgseq_=None,
            f_ego_head_=None,
            f_ego_hand_=None,
            f_interaction_=None,
        ):
            obs_ = obs_.clone()
            visible_mask = obs_[..., [2]] > 0.5
            obs_[~visible_mask[..., 0]] = 0
            f_obs = self.learned_pos_linear(obs_[..., :2])
            f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
            x_ = self.embed_noisyobs(f_obs.view(B, L, -1))

            f_to_add_ = [self.cliffcam_embedder(f_cliffcam_)]
            if hasattr(self, "cam_angvel_embedder"):
                f_to_add_.append(self.cam_angvel_embedder(f_cam_angvel_))
            if f_cam_trans_vel_ is not None and hasattr(self, "cam_trans_vel_embedder"):
                f_to_add_.append(self.cam_trans_vel_embedder(f_cam_trans_vel_))
            if f_gravity_dir_ is not None and hasattr(self, "gravity_embedder"):
                f_to_add_.append(self.gravity_embedder(f_gravity_dir_))
            if f_imgseq_ is not None and hasattr(self, "imgseq_embedder"):
                f_to_add_.append(self.imgseq_embedder(f_imgseq_))
            if f_ego_imgseq_ is not None and hasattr(self, "ego_imgseq_embedder"):
                f_to_add_.append(self.ego_imgseq_embedder(f_ego_imgseq_))
            if f_ego_head_ is not None and hasattr(self, "ego_head_embedder"):
                f_to_add_.append(self.ego_head_embedder(f_ego_head_))
            if f_ego_hand_ is not None and hasattr(self, "ego_hand_embedder"):
                f_to_add_.append(self.ego_hand_embedder(f_ego_hand_))
            if f_interaction_ is not None and hasattr(self, "interaction_embedder"):
                f_to_add_.append(self.interaction_embedder(f_interaction_))

            for f_delta in f_to_add_:
                x_ = x_ + f_delta
            return x_

        x = encode_stream(
            obs,
            f_cliffcam,
            f_cam_angvel,
            f_cam_trans_vel,
            f_gravity_dir,
            f_imgseq,
            f_ego_imgseq,
            f_ego_head,
            f_ego_hand,
            f_interaction,
        )

        x_cross = None
        if self.cross_view_fusion and cross_obs is not None:
            x_cross = encode_stream(
                cross_obs,
                cross_f_cliffcam if cross_f_cliffcam is not None else f_cliffcam,
                cross_f_cam_angvel if cross_f_cam_angvel is not None else f_cam_angvel,
                cross_f_cam_trans_vel,
                cross_f_gravity_dir,
                cross_f_imgseq,
                cross_f_ego_imgseq,
            )

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        if x_cross is not None:
            x_cat = torch.cat([x, x_cross], dim=0)
            pmask_cat = torch.cat([pmask, pmask], dim=0)
            for block in self.blocks:
                x_cat = block(x_cat, attn_mask=attnmask, tgt_key_padding_mask=pmask_cat)
            x, x_cross = x_cat[:B], x_cat[B:]
            cross_delta, _ = self.cross_attn(
                self.cross_q_norm(x),
                self.cross_kv_norm(x_cross),
                self.cross_kv_norm(x_cross),
                key_padding_mask=pmask,
                need_weights=False,
            )
            x = x + self.cross_gate * cross_delta
        else:
            for block in self.blocks:
                x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Output
        sample = self.final_layer(x)  # (B, L, C)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)
        sample_ego = None
        if self.dual_head and self.final_layer_ego is not None:
            sample_ego = self.final_layer_ego(x)
            if self.avgbeta and sample_ego.size(-1) >= 136:
                betas = (sample_ego[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
                betas = repeat(betas, "b c -> b l c", l=L)
                sample_ego = torch.cat([sample_ego[..., :126], betas, sample_ego[..., 136:]], dim=-1)
        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)
        pred_cam_ego = None
        if self.pred_cam_head_ego is not None:
            pred_cam_ego = self.pred_cam_head_ego(x)
            pred_cam_ego = pred_cam_ego * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam_ego[..., 0], 0.25)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')
        static_conf_logits_ego = None
        if self.static_conf_head_ego is not None:
            static_conf_logits_ego = self.static_conf_head_ego(x)

        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }
        if sample_ego is not None:
            output.update(
                {
                    "pred_x_ego": sample_ego,
                    "pred_cam_ego": pred_cam_ego,
                    "static_conf_logits_ego": static_conf_logits_ego,
                }
            )
        return output


# Add to MainStore
group_name = "network/gvhmr"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
