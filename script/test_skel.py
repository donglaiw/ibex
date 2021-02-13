import os,sys
from ibexHelper.skel import CreateSkeletons
from ibexHelper.util import ReadH5

def test_snemi(seg_path, output_path = './'):
    seg = ReadH5(seg_path)
    input_res = [30, 6, 6]
    # save into a pickle file
    CreateSkeletons(seg, output_path, res, return_option='save')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('need an argument to select the test')
    opt = sys.argv[1]
    if opt=='0': # mesh -> skeleton
        if len(sys.argv) < 3:
            print('need an argument for the segmentation file (h5)')
        test_snemi(sys.argv[2])
