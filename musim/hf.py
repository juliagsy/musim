import numpy as np
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline


class MusImPipeline(DiffusionPipeline):
    def __init__(self, vae, unet, scheduler, muvis, linear):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler, muvis=muvis, linear=linear)
        self.loss_mse = nn.MSELoss()

    def forward(self, wav, img):
        uncond_t = torch.zeros_like(wav.input_values, device=self.device)
        wav_t = self.muvis(**wav)["last_hidden_state"]

        latents = self.vae.encode(img)
        latents = latents.latent_dist.sample()
        noise = torch.rand_like(latents, device=self.device)
        t = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (img.size(0),), device=self.device
        ).long()
        latents = self.scheduler.add_noise(latents, noise, t)

        uncond_embeddings = self.muvis(uncond_t)["last_hidden_state"]
        wav_embs = torch.cat([uncond_embeddings, wav_t])
        wav_embs = self.linear(wav_embs)

        latent_model_input = torch.cat([latents] * 2)

        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=wav_embs).sample
        loss = self.loss_mse(noise_pred, noise)
        return loss

    @torch.no_grad()
    def __call__(self, wav, img=None, batch_size=1, num_inference_steps=25, guidance_scale=7.5):
        wav_t = self.muvis(**wav)["last_hidden_state"]

        if img is not None:
            latents = self.vae.encode(img)
            latents = latents.latent_dist.sample()
            latents = latents + torch.rand_like(latents).to(self.device)
        else:
            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            )

        latents = latents.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        uncond_embeddings = self.muvis(torch.zeros(1, 1024, 128).to(self.device))["last_hidden_state"]
        wav_embs = torch.cat([uncond_embeddings, wav_t])
        wav_embs = self.linear(wav_embs)

        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=wav_embs).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        return image