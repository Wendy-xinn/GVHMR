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
        cross_extrinsic_dim=9,
        cross_local_radius=4,
        cross_teacher_mvp=False,
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
        self.cross_extrinsic_dim = cross_extrinsic_dim
        self.cross_local_radius = int(cross_local_radius)
        self.cross_teacher_mvp = bool(cross_teacher_mvp)

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
            self.partner_cross_q_norm = nn.LayerNorm(self.latent_dim)
            self.partner_cross_kv_norm = nn.LayerNorm(self.latent_dim)
            self.wearer_cross_q_norm = nn.LayerNorm(self.latent_dim)
            self.wearer_cross_kv_norm = nn.LayerNorm(self.latent_dim)
            self.partner_cross_attn = nn.MultiheadAttention(
                self.latent_dim, num_heads=self.cross_view_heads, dropout=dropout, batch_first=True
            )
            self.wearer_cross_attn = nn.MultiheadAttention(
                self.latent_dim, num_heads=self.cross_view_heads, dropout=dropout, batch_first=True
            )
            self.partner_cross_gate = Mlp(self.latent_dim * 2, hidden_features=self.latent_dim, out_features=1)
            self.wearer_cross_gate = Mlp(self.latent_dim * 2, hidden_features=self.latent_dim, out_features=1)
            nn.init.zeros_(self.partner_cross_gate.fc2.weight)
            nn.init.zeros_(self.partner_cross_gate.fc2.bias)
            nn.init.zeros_(self.wearer_cross_gate.fc2.weight)
            nn.init.constant_(self.wearer_cross_gate.fc2.bias, 1.3862944)  # sigmoid -> 0.8
            self.partner_source_proj = zero_module(nn.Linear(self.latent_dim, self.latent_dim))
            self.wearer_source_proj = zero_module(nn.Linear(self.latent_dim, self.latent_dim))
            self.partner_teacher_norm = nn.LayerNorm(self.latent_dim)
            self.wearer_teacher_norm = nn.LayerNorm(self.latent_dim)
            self.cross_interactee_type = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
            self.cross_wearer_type = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
            if self.cross_teacher_mvp:
                # 21 local joint delta rotations + root delta rotation + root delta translation.
                self.teacher_partner_residual_head = Mlp(self.latent_dim, out_features=69)
                self.teacher_wearer_residual_head = Mlp(self.latent_dim, out_features=69)
                nn.init.zeros_(self.teacher_partner_residual_head.fc2.weight)
                nn.init.zeros_(self.teacher_partner_residual_head.fc2.bias)
                nn.init.zeros_(self.teacher_wearer_residual_head.fc2.weight)
                nn.init.zeros_(self.teacher_wearer_residual_head.fc2.bias)


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
        if self.cross_extrinsic_dim > 0:
            self.cross_extrinsic_embedder = nn.Sequential(
                nn.LayerNorm(self.cross_extrinsic_dim),
                nn.Linear(self.cross_extrinsic_dim, latent_dim),
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
        cross_interactee_obs=None,
        cross_interactee_f_cliffcam=None,
        cross_interactee_f_cam_angvel=None,
        cross_interactee_f_cam_trans_vel=None,
        cross_interactee_f_gravity_dir=None,
        cross_interactee_f_imgseq=None,
        cross_interactee_f_extrinsic=None,
        cross_interactee_key_padding_mask=None,
        cross_wearer_obs=None,
        cross_wearer_f_cliffcam=None,
        cross_wearer_f_cam_angvel=None,
        cross_wearer_f_cam_trans_vel=None,
        cross_wearer_f_gravity_dir=None,
        cross_wearer_f_imgseq=None,
        cross_wearer_f_extrinsic=None,
        cross_wearer_key_padding_mask=None,
        ego_partner_visibility=None,
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
            f_cross_extrinsic_=None,
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
            if f_cross_extrinsic_ is not None and hasattr(self, "cross_extrinsic_embedder"):
                f_to_add_.append(self.cross_extrinsic_embedder(f_cross_extrinsic_))

            for f_delta in f_to_add_:
                x_ = x_ + f_delta
            return x_

        x_partner = encode_stream(
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
        x = x_partner
        x_ego_context = None

        x_interactee_src = None
        x_wearer_src = None
        partner_gate = None
        wearer_gate = None
        if self.cross_view_fusion:
            if cross_interactee_obs is not None:
                x_interactee_src = encode_stream(
                    cross_interactee_obs,
                    cross_interactee_f_cliffcam,
                    cross_interactee_f_cam_angvel,
                    cross_interactee_f_cam_trans_vel,
                    cross_interactee_f_gravity_dir,
                    cross_interactee_f_imgseq,
                    f_cross_extrinsic_=cross_interactee_f_extrinsic,
                ) + self.cross_interactee_type
            if cross_wearer_obs is not None:
                x_wearer_src = encode_stream(
                    cross_wearer_obs,
                    cross_wearer_f_cliffcam,
                    cross_wearer_f_cam_angvel,
                    cross_wearer_f_cam_trans_vel,
                    cross_wearer_f_gravity_dir,
                    cross_wearer_f_imgseq,
                    f_cross_extrinsic_=cross_wearer_f_extrinsic,
                ) + self.cross_wearer_type

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        def prepare_source_pmask(mask):
            source_pmask = pmask if mask is None else pmask | mask.to(device=x.device, dtype=torch.bool)
            source_available = (~source_pmask).any(dim=1)
            # An unavailable source is ignored after attention, but every local-attention
            # query still needs at least one unmasked key to keep softmax finite.
            safe_pmask = torch.where(source_available[:, None], source_pmask, pmask)
            return safe_pmask, source_available

        pmask_interactee, interactee_available = prepare_source_pmask(cross_interactee_key_padding_mask)
        pmask_wearer, wearer_available = prepare_source_pmask(cross_wearer_key_padding_mask)

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

        if x_interactee_src is not None or x_wearer_src is not None:
            wearer_obs = obs.new_zeros(obs.shape)
            wearer_cliffcam = f_cliffcam.new_zeros(f_cliffcam.shape)
            x_wearer_tgt = encode_stream(
                wearer_obs,
                wearer_cliffcam,
                f_cam_angvel,
                f_cam_trans_vel,
                f_gravity_dir,
                None,
                None,
                f_ego_head,
                f_ego_hand,
                None,
            )

            streams = [x_partner, x_wearer_tgt]
            stream_pmasks = [pmask, pmask]
            if x_interactee_src is not None:
                streams.append(x_interactee_src)
                stream_pmasks.append(pmask_interactee)
            if x_wearer_src is not None:
                streams.append(x_wearer_src)
                stream_pmasks.append(pmask_wearer)

            x_cat = torch.cat(streams, dim=0)
            pmask_cat = torch.cat(stream_pmasks, dim=0)
            for block in self.blocks:
                x_cat = block(x_cat, attn_mask=attnmask, tgt_key_padding_mask=pmask_cat)
            encoded = list(x_cat.chunk(len(streams), dim=0))
            x_partner = encoded.pop(0)
            x_wearer_tgt = encoded.pop(0)
            local_attn_mask = None
            if self.cross_local_radius >= 0:
                frame_ids = torch.arange(L, device=x.device)
                local_attn_mask = (frame_ids[:, None] - frame_ids[None, :]).abs() > self.cross_local_radius

            def run_local_cross_attention(attn, q_norm, kv_norm, query, source, source_pmask):
                """Attend to valid local source frames without creating all-masked softmax rows."""
                source_valid = ~source_pmask
                source_for_attn = source * source_valid[..., None].to(source.dtype)
                if local_attn_mask is None:
                    local_available = source_valid.any(dim=1, keepdim=True).expand(-1, L)
                else:
                    local_available = (
                        source_valid[:, None, :] & ~local_attn_mask[None, :, :]
                    ).any(dim=-1)
                delta, _ = attn(
                    q_norm(query),
                    kv_norm(source_for_attn),
                    kv_norm(source_for_attn),
                    attn_mask=local_attn_mask,
                    need_weights=False,
                )
                delta = delta * local_available[..., None].to(delta.dtype)
                return delta, local_available

            partner_gate = None
            wearer_gate = None
            if x_interactee_src is not None:
                x_interactee_src = encoded.pop(0)
                partner_delta, partner_local_available = run_local_cross_attention(
                    self.partner_cross_attn,
                    self.partner_cross_q_norm,
                    self.partner_cross_kv_norm,
                    x_partner,
                    x_interactee_src,
                    pmask_interactee,
                )
                partner_local_available = partner_local_available & interactee_available[:, None]
                partner_available_f = partner_local_available[..., None].to(partner_delta.dtype)
                partner_delta = partner_delta * partner_available_f
                partner_gate = torch.sigmoid(self.partner_cross_gate(torch.cat([x_partner, partner_delta], dim=-1)))
                if ego_partner_visibility is not None:
                    visibility = ego_partner_visibility.to(device=x.device, dtype=x_partner.dtype).clamp(0.0, 1.0)
                    if visibility.ndim == 2:
                        visibility = visibility[..., None]
                    # Low target-view visibility deterministically increases reliance on complete exo evidence.
                    partner_gate = partner_gate + (1.0 - visibility) * (1.0 - partner_gate)
                partner_gate = partner_gate * partner_available_f
                aligned_source = self.partner_source_proj(x_interactee_src)
                aligned_source = aligned_source * (~pmask_interactee)[..., None].to(aligned_source.dtype)
                aligned_source = aligned_source * partner_available_f
                x_partner = self.partner_teacher_norm(x_partner + partner_gate * partner_delta + aligned_source)

            if x_wearer_src is not None:
                x_wearer_src = encoded.pop(0)
                wearer_delta, wearer_local_available = run_local_cross_attention(
                    self.wearer_cross_attn,
                    self.wearer_cross_q_norm,
                    self.wearer_cross_kv_norm,
                    x_wearer_tgt,
                    x_wearer_src,
                    pmask_wearer,
                )
                wearer_local_available = wearer_local_available & wearer_available[:, None]
                wearer_available_f = wearer_local_available[..., None].to(wearer_delta.dtype)
                wearer_delta = wearer_delta * wearer_available_f
                wearer_gate = torch.sigmoid(self.wearer_cross_gate(torch.cat([x_wearer_tgt, wearer_delta], dim=-1)))
                wearer_gate = wearer_gate * wearer_available_f
                aligned_source = self.wearer_source_proj(x_wearer_src)
                aligned_source = aligned_source * (~pmask_wearer)[..., None].to(aligned_source.dtype)
                aligned_source = aligned_source * wearer_available_f
                x_wearer_tgt = self.wearer_teacher_norm(x_wearer_tgt + wearer_gate * wearer_delta + aligned_source)

            x = x_partner
            x_ego_context = x_wearer_tgt
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
        ego_context = x_ego_context if x_ego_context is not None else x
        if self.dual_head and self.final_layer_ego is not None:
            sample_ego = self.final_layer_ego(ego_context)
            if self.avgbeta and sample_ego.size(-1) >= 136:
                betas = (sample_ego[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
                betas = repeat(betas, "b c -> b l c", l=L)
                sample_ego = torch.cat([sample_ego[..., :126], betas, sample_ego[..., 136:]], dim=-1)
        source_base = {}
        if self.cross_teacher_mvp:
            for source_name, source_context in (("partner", x_interactee_src), ("wearer", x_wearer_src)):
                if source_context is None:
                    continue
                source_x = self.final_layer(source_context)
                if self.avgbeta:
                    source_mask = pmask_interactee if source_name == "partner" else pmask_wearer
                    source_length = (~source_mask).sum(1).clamp_min(1)
                    source_betas = (source_x[..., 126:136] * (~source_mask[..., None])).sum(1) / source_length[:, None]
                    source_x = torch.cat([source_x[..., :126], repeat(source_betas, "b c -> b l c", l=L), source_x[..., 136:]], dim=-1)
                source_cam = self.pred_cam_head(source_context) if self.pred_cam_head else None
                if source_cam is not None:
                    source_cam = source_cam * self.pred_cam_std + self.pred_cam_mean
                    source_cam[..., 0].clamp_min_(0.25)
                source_static = self.static_conf_head(source_context) if self.static_conf_head else None
                residual_head = self.teacher_partner_residual_head if source_name == "partner" else self.teacher_wearer_residual_head
                target_context = x_partner if source_name == "partner" else x_wearer_tgt
                source_base[source_name] = {
                    "pred_x": source_x,
                    "pred_cam": source_cam,
                    "static_conf_logits": source_static,
                    "residual": residual_head(target_context),
                }

        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)
        pred_cam_ego = None
        if self.pred_cam_head_ego is not None:
            pred_cam_ego = self.pred_cam_head_ego(ego_context)
            pred_cam_ego = pred_cam_ego * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam_ego[..., 0], 0.25)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')
        static_conf_logits_ego = None
        if self.static_conf_head_ego is not None:
            static_conf_logits_ego = self.static_conf_head_ego(ego_context)

        output = {
            "pred_context": x,
            "pred_context_ego": ego_context,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            "cross_partner_gate": partner_gate,
            "cross_wearer_gate": wearer_gate,
            "cross_source_base": source_base,
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
