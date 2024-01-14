def random_shadow(image_to_shadow):
  shadow_func = iaa.AdditiveGaussianNoise(scale=(0.5,0.7))
  shadow_image = shadow_func.augment_image(image_to_shadow)
  return shadow_image

def img_random_shear(image_to_shear):
    shear_func = iaa.Affine(shear=(-10, 10))
    sheared_image = shear_func.augment_image(image_to_shear)
    return sheared_image

def img_random_color_jitter(image_to_jitter):
    jitter_func = iaa.AddToHueAndSaturation((-20, 20), per_channel=True)
    jittered_image = jitter_func.augment_image(image_to_jitter)
    return jittered_image

def img_elastic_transform(image_to_transform):
    transform_func = iaa.ElasticTransformation(alpha=2, sigma=0.25)
    transformed_image = transform_func.augment_image(image_to_transform)
    return transformed_image

def img_perspective_transform(image_to_transform):
    perspective_func = iaa.PerspectiveTransform(scale=(0.01, 0.15))
    perspective_image = perspective_func.augment_image(image_to_transform)
    return perspective_image

def img_random_noise(image_to_noise):
    noise_func = iaa.AdditiveGaussianNoise(loc=0, scale=(0.01*255, 0.05*255))
    noisy_image = noise_func.augment_image(image_to_noise)
    return noisy_image


def random_augment(image_to_augment,steering_angle):
  augment_image = mpimg.imread(image_to_augment)
  if np.random.rand() < 0.5:
    augment_image = zoom(augment_image)
  if np.random.rand() < 0.5:
    augment_image = pan(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_random_brightness(augment_image)
  if np.random.rand() < 0.5:
    augment_image, steering_angle = img_random_flip(augment_image,steering_angle)
  if np.random.rand() < 0.5:
    augment_image = random_shadow(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_random_shear(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_random_color_jitter(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_elastic_transform(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_perspective_transform(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_random_noise(augment_image)
  return augment_image, steering_angle