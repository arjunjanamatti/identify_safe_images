# Importing necessary functions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

# main_directory = 'C:/Users/Arjun Janamatti/Documents/image_classification/nude_sexy_safe_v1_x320/training/'

class ImageAugment:
    def __init__(self, main_directory, num_images_to_augment):
        self.main_directory = main_directory
        self.dict_1 = {}
        self.sub_dir = os.listdir(self.main_directory)
        self.num_images_to_augment = num_images_to_augment

    def make_directories(self):
        for sub in self.sub_dir:
            self.dict_1[sub] = len(os.listdir(self.main_directory + sub))
            print(sub, len(os.listdir(self.main_directory + sub)))
        try:
            for sub in self.dict_1:
                os.mkdir(self.main_directory + sub + '_aug')
        except Exception as e:
            pass

    def get_file_name(self):
        images_list = []
        for sub in dict_1:
            images_list.append(os.listdir(main_directory + sub))

        name_index = list(dict_1.keys())
        flat_image_name = []
        flat_image_label = []
        for index, i in enumerate(images_list):
            for indes_i, j in enumerate(i):
                if indes_i < self.num_images_to_augment:
                    flat_image_name.append(name_index[index] + '/' + j)
                    flat_image_label.append(name_index[index])

        self.sexy_image_name = [file for file in flat_image_name
                           if 'sexy' in file.split('/')]
        self.nude_image_name = [file for file in flat_image_name
                           if 'nude' in file.split('/')]
        self.safe_image_name = [file for file in flat_image_name
                           if 'safe' in file.split('/')]
        pass

    pass

main_directory = 'C:/Users/Arjun Janamatti/Documents/image_classification/nude_sexy_safe_v1_x320/training/'
dict_1 = {}
sub_dir = os.listdir(main_directory)

for sub in sub_dir:
    dict_1[sub] = len(os.listdir(main_directory+sub))
    print(sub, len(os.listdir(main_directory+sub)))

# making new directories
try:
    for sub in dict_1:
        os.mkdir(main_directory+sub+'_aug')
except Exception as e:
    pass

images_list = []
for sub in dict_1:
    images_list.append(os.listdir(main_directory+sub))

new_images_list = []
name_index = list(dict_1.keys())
flat_image_name = []
flat_image_label = []
for index, i in enumerate(images_list):
    for indes_i, j in enumerate(i):
        if indes_i < 10000:
            flat_image_name.append(name_index[index]+'/'+j)
            flat_image_label.append(name_index[index])

sexy_image_name = [file for file in flat_image_name
                   if 'sexy' in file.split('/')]
nude_image_name = [file for file in flat_image_name
                   if 'nude' in file.split('/')]
safe_image_name = [file for file in flat_image_name
                   if 'safe' in file.split('/')]

def augment_images(list_name, folder_name):
    for image in list_name:
        try:
            # Loading a sample image
            img = load_img(main_directory+image)
            # Converting the input sample image to an array
            x = img_to_array(img)
            # Reshaping the input image
            x = x.reshape((1, ) + x.shape)

            # Generating and saving 5 augmented samples
            # using the above defined parameters.
            i = 0
            unique_code = ((image.split('/')[-1]).split('.')[0]).split('-')[-1]
            for batch in datagen.flow(x, batch_size = 1,
                                      save_to_dir = main_directory+folder_name+'_aug',
                                      save_prefix ='{}_aug_image'.format(folder_name+'_'+unique_code), save_format ='jpeg'):
                i += 1
#                 print(((image.split('/')[-1]).split('.')[0]).split('-')[-1])
                if i > 5:
                    break
        except Exception as e:
            pass

augment_images(sexy_image_name, 'sexy')
augment_images(safe_image_name, 'safe')
augment_images(nude_image_name, 'nude')