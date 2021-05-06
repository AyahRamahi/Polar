from PIL import Image,
from os import listdir,
from os.path import isfile, join, isdir
import cv2

def video_to_frames(input_video, output_folder):
    vidcap = cv2.VideoCapture()
    success,image = vidcap.read(input_video)
    count = 1
    while success:
        cv2.imwrite(output_folder + '/%d.jpg' % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
    print(count)

def convert_JPG_frames_to_TIFF(input_frames_folder, output_frames_folder):
    for image in listdir(input_frames_folder + '/'):
        im = Image.open(join(input_frames_folder + '/', image))
        im.save(output_frames_folder + '/'+ str(count) +'.tif', 'TIFF')


def main():
    train_input_video = 'CCTV/train.mp4'
    train_frames_output_folder = 'CCTV/train_frames'
    train_tiff_frames_output_folder = 'CCTV/train_tiff_frames'

    test_input_video = 'CCTV/test.mp4'
    test_frames_output_folder = 'CCTV/test_frames'
    test_tiff_frames_output_folder = 'CCTV/test_frames_2'

    # get train TIFF frames
    video_to_frames(train_input_video, train_frames_output_folder)
    convert_JPG_frames_to_TIFF(train_frames_output_folder, train_tiff_frames_output_folder)

    # get test TIFF frames
    video_to_frames(test_input_video, test_frames_output_folder)
    convert_JPG_frames_to_TIFF(test_frames_output_folder, test_tiff_frames_output_folder)


if __name__ == '__main__':
    main()