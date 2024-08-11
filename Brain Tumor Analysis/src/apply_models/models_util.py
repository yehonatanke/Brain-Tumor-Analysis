from src.utils.util import check_dataset_fields

# Color macros
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def get_image_and_label(dataset, split_name: str = 'train', index: int = 0, field: str = 'image'):
    # Check fields validation
    if not check_dataset_fields(dataset, split_name, index, field):
        return None, None

    # Check if the 'image' field exists in the dataset
    if 'image' not in dataset[split_name].column_names:
        print("'image' field not found in the dataset.")
        return None, None

    # Select an image from the chosen split and index
    image = dataset[split_name][index]['image']
    label = dataset[split_name][index]['label']

    return image, label


def save_image(save_dir, index, plt):
    def file_exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return True
        except FileNotFoundError:
            return False

    base_filename = f'image_{index}.png'
    save_path = f'{save_dir}/{base_filename}'

    # Check if file already exists
    counter = 1
    while file_exists(save_path):
        # If file exists, append a number to the filename
        filename, extension = base_filename.rsplit('.', 1)
        new_filename = f'{filename}_{counter}.{extension}'
        save_path = f'{save_dir}/{new_filename}'
        counter += 1

    plt.savefig(save_path, bbox_inches='tight')
    print(f'Image saved to {save_path}')
    return save_path
