import os
import glob
import argparse


# Count the number of files in each directory and perform a summary in a table

def count_files_illness(parent_dir, illness, mode_to_count='all'):
    
    assert illness in ['altpig', 'dmae', 'excavation', 'membrana', 'nevus']
    assert mode_to_count in ['train', 'test', 'validation', 'all'], 'mode_to_count parameter not recognized'
    
    count = 0
    illness_dir = 'u_{}_symbolic_512'.format(illness)    

    if mode_to_count != 'all':
        filenames_dir = os.path.join(parent_dir, illness_dir, mode_to_count, illness)
        count = len([f for f in os.listdir(filenames_dir) if f.endswith('jpg')])
        
    else:
        for mode in ['test', 'train', 'validation']:
            filenames_dir = os.path.join(parent_dir, illness_dir, mode, illness)
            count += len([f for f in os.listdir(filenames_dir) if f.endswith('jpg')])
    
    return count


def count_files_healthy(parent_dir, mode_to_count='all'):

    assert mode_to_count in ['train', 'test', 'validation', 'all'], 'mode_to_count parameter not recognized'
    
    pathologies = ['altpig', 'dmae', 'excavation', 'membrana', 'nevus']
    count = 0
    
    for pathology in pathologies:        
        pathology_dir = 'u_{}_symbolic_512'.format(pathology)

        if mode_to_count != 'all':
            dir_to_search = os.path.join(parent_dir, pathology_dir, mode_to_count) + '/normal/*' # We only want the 'normal' images in this function
            count += len([f for f in glob.glob(dir_to_search) if f.endswith('.jpg')])

        else:
            for mode in ['train', 'test', 'validation']:
                dir_to_search = os.path.join(parent_dir, pathology_dir, mode) + '/normal/*' # We only want the 'normal' images in this function
                count += len([f for f in glob.glob(dir_to_search) if f.endswith('.jpg')])
    
    return count


if __name__ == '__main__':

    # Create an ArgumentParser object to be able to specify some parameters when executing the file
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir', default=None,
                        help="Directory containing the original datasets")

    # Get the data_dir from the ArgParser object
    args = parser.parse_args()

    for mode in ['train', 'test', 'validation', 'all']:

        print('----- Counting {} images -----'.format(mode))
        pathologies = ['altpig', 'dmae', 'excavation', 'membrana', 'nevus']
        for pathology in pathologies:
            c = count_files_illness(parent_dir=args.parent_dir, illness=pathology, mode_to_count=mode)
            print('Number of images with {}: {}'.format(pathology, c))

        c = count_files_healthy(parent_dir=args.parent_dir, mode_to_count=mode)
        print('Number of images with no pathology detected: {}'.format(c))