import numpy as np

def rotate_image(img, angle, center=None):
    '''Rotates an image (angle in degrees) and expands image to avoid cropping'''
    from cv2 import getRotationMatrix2D, warpAffine, INTER_LINEAR
    
    height, width = img.shape[:2] 

    if center is None:
        center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    
    rotation_mat = getRotationMatrix2D(center, angle, 1)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    
    # Find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # Subtract old image center and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - center[0]
    rotation_mat[1, 2] += bound_h/2 - center[1]
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_img = warpAffine(img, rotation_mat, (bound_w, bound_h), flags=INTER_LINEAR)
    return rotated_img


def rotate_image_pil(img, angle, center=None):
    '''Use PIL to rotate large images that opencv cannot handle without cropping.'''
    
    from PIL import Image
    
    image = Image.fromarray(img)
    if center is not None and not isinstance(center, tuple):
        center = tuple(center)
        
    return np.array(image.rotate(angle, center=center, resample=Image.BILINEAR, expand=True))