import numpy as np


def encode_rle(mask):
    """Returns encoded mask (run length) as a string.
    Parameters
    ----------
    mask : np.ndarray, 2d
        Mask that consists of 2 unique values: 0 - denotes background, 1 - denotes object.
    Returns
    -------
    str
        Encoded mask.
    Notes
    -----
    Mask should contains only 2 unique values, one of them must be 0, another value, that denotes
    object, could be different from 1 (for example 255).
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def decode_rle(rle_mask, shape=(320, 240)):
    """Decodes mask from rle string.
    Parameters
    ----------
    rle_mask : str
        Run length as string formatted.
    shape : tuple of 2 int, optional (default=(320, 240))
        Shape of the decoded image.
    Returns
    -------
    np.ndarray, 2d
        Mask that contains only 2 unique values: 0 - denotes background, 1 - denotes object.
    
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1

    return img.reshape(shape)

def plot_images(imgName, path_train_img, path_train_mask):
    """Plots images and masks of object by its index, and paths.
    Parameters
    ----------
    imgName : int
        Name of image
    path_train_img: str
        Path to images directory
    path_train_mask: str
        Path to masks directory
    """

    img = Image.open(path_train_img + '{}.jpg'.format(imgName))
    mask = Image.open(path_train_mask + '{}.png'.format(imgName))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    axes[0].imshow(img[...,[2,1,0]])
    axes[1].imshow(mask)
    plt.show()

    
def plot_image(x,y):
    """Plots images
    Parameters
    ----------
    x : nd.array
        Image itself
    y : nd.array
        Image itself
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    axes[0].imshow(x)
    axes[1].imshow(np.squeeze(y))
    plt.show()

def generator(paths_to_imgs, paths_to_masks, batch_size):
    """Generates batch of images for fit_generator() method
    with augmentations. Images are normalized and 
    resized to (256,256) shape.
    
    6 images of batch are original images,
    batch_size - 6 images are augmentations.
    
    Parameters
    ----------
    paths_to_imgs: str
        Paths to images directory
    paths_to_masks: str
        Paths to masks directory
    batch_size : int
        Size of batch
    Returns
    -------
    x: np.ndarray, (batch_size, 256, 256, 3)
        array of resized images
    y: np.ndarray, (batch_size, 256, 256, 1)
        array of resized images        
        
    """
    import random
    while True:
        x_batch = []
        y_batch = []
        
        #initialize array of temporary paths
        paths_to_imgs_tmp = []
        paths_to_masks_tmp = []
        
        #choose random images from directory
        for i in range(batch_size):
            a = random.randint(1,1222)
            paths_to_imgs_tmp.append(paths_to_imgs[a])
            paths_to_masks_tmp.append(paths_to_masks[a])
            
        #add images to batch
        for path_to_img, path_to_mask in zip(paths_to_imgs_tmp[:6], paths_to_masks_tmp[:6]):
            
            img = cv2.imread(path_to_img)
            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(path_to_mask)
            mask = cv2.resize(mask, (224,224))
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            x_batch.append(img)
            y_batch.append(gray_mask)
            
        #add augmented images to batch
        for path_to_img, path_to_mask in zip(paths_to_imgs_tmp[6:], paths_to_masks_tmp[6:]):
            img = cv2.imread(path_to_img)
            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(path_to_mask)
            mask = cv2.resize(mask, (224,224))
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            
            augmented = aug(image=img, mask=gray_mask)
            image_medium = augmented['image']
            mask_medium = augmented['mask']
            
            x_batch.append(image_medium)
            y_batch.append(mask_medium)     
       
        #normalize images
        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)
        
        
        yield x_batch, np.expand_dims(y_batch,-1)
        
        
def generator_predict(paths_to_imgs, batch_size):
    """Generates batch of images for predict() method.
    Parameters
    ----------
    paths_to_imgs: str
        Paths to images directory
    batch_size : int
        Size of batch
    Returns
    -------
    x: np.ndarray, (batch_size, 256, 256, 3)
        array of resized images        
        
    """
    while True:
        
        x_batch = []
        paths_to_imgs_tmp=[]
        
        
        for i in range(batch_size):
            a=random.randint(1, batch_size)
            paths_to_imgs_tmp.append(paths_to_imgs[a])
        
        
        for path_to_img in paths_to_imgs_tmp:
            
            img = cv2.imread(path_to_img)
            img = cv2.resize(img, (256,256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            x_batch.append(img)
        
        
        x_batch = np.array(x_batch)

        
        yield x_batch

        
def plot_losses(results):
    """Plots a simple loss graph
    Parameters
    ----------
    results: history
        
    """
    plt.figure(figsize=(10,10))
    plt.plot(results.history['dice_loss'])
    plt.plot(results.history['f1_score'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()