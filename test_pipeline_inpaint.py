import torch
from dalle2_pytorch import Unet, Decoder, CLIP

# trained clip from step 1

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# 2 unets for the decoder (a la cascading DDPM)

unet = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 1, 1, 1)
).cuda()


# decoder, which contains the unet(s) and clip

decoder = Decoder(
    clip = clip,
    unet = (unet,),               # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
    image_sizes = (256,),         # resolutions, 256 for first unet, 512 for second. these must be unique and in ascending order (matches with the unets passed in)
    timesteps = 100,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
).cuda()

# mock images (get a lot of this)

images = torch.randn(4, 3, 256, 256).cuda()

# feed images into decoder, specifying which unet you want to train
# each unet can be trained separately, which is one of the benefits of the cascading DDPM scheme

loss = decoder(images, unet_number = 1)
loss.backward()

# do the above for many steps for both unets

mock_image_embed = torch.randn(1, 512).cuda()

# then to do inpainting

inpaint_image = torch.randn(1, 3, 256, 256).cuda()      # (batch, channels, height, width)
inpaint_mask = torch.ones(1, 256, 256).bool().cuda()    # (batch, height, width)

inpainted_images = decoder.sample(
    image_embed = mock_image_embed,
    inpaint_image = inpaint_image,    # just pass in the inpaint image
    inpaint_mask = inpaint_mask       # and the mask
)

inpainted_images.shape # (1, 3, 256, 256)