import numpy as np
import face_alignment
from skimage import io
import argparse

from fit_lmk3d import fit_lmk3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor
from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from os.path import join, basename, dirname


parser = argparse.ArgumentParser(description='Generate a 3D model from a 2d face image.')
parser.add_argument('-input', required=True, help='name of the input image')
parser.add_argument('-output', help='path of the output file')
parser.add_argument('-g', default='n', choices=['m', 'f', 'n'],
                    help='gender of the model,input m/f stand by male/female, default = none')

args = parser.parse_args()


def ProcessGender():
    return args.g

# process input
def ProcessInput():
    ipdir = args.input
    return ipdir


def ProcessOutput():
    opdir = './output'
    ipdir = args.input.split('/')
    opfname = ipdir[-1].split('.')
    opfname = opfname[0] + '_3d.obj'
    
    if args.output != None:
        opdir = dirname(args.output)
        opfname = basename(args.output)
    
    return opdir, opfname


def get3dlmks():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    ipdir = ProcessInput()
    input = io.imread(ipdir)  #'./input/trump.jpg'
    preds = fa.get_landmarks(input)
    return np.array(preds)


def lmk2flame():
    facelmk = get3dlmks()
    facelmk = facelmk[0][17:]
    unit = 'mm'

    scale_factor = get_unit_factor('m') / get_unit_factor(unit)
    lmk_3d = scale_factor * facelmk
    x_mean = np.mean(lmk_3d[:, 0])
    y_mean = np.mean(lmk_3d[:, 1])
    z_mean = np.mean(lmk_3d[:, 2])
    scale = (np.max(lmk_3d[:, 0]) - np.min(lmk_3d[:, 0])) / 0.1080635
    for i in range(len(lmk_3d)):
        '''temp = lmk_3d[i][0]
        lmk_3d[i][0] = lmk_3d[i][1]
        lmk_3d[i][1] = temp'''
        lmk_3d[i][1] = - lmk_3d[i][1]
        lmk_3d[i][0] = (lmk_3d[i][0] - x_mean) / scale
        lmk_3d[i][1] = (lmk_3d[i][1] - y_mean) / scale
        lmk_3d[i][2] = (lmk_3d[i][2] - z_mean) / scale
    print(lmk_3d)

    # model
    gender = ProcessGender()
    if gender == 'm':
        model_path = './models/male_model.pkl'
    elif gender == 'f':
        model_path = './models/female_model.pkl'
    else:
        model_path = './models/generic_model.pkl'
    model = load_model(
        model_path)  # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print("loaded model from:", model_path)

    # landmark embedding
    lmk_emb_path = './models/flame_static_embedding.pkl'
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("loaded lmk embedding")

    # output
    output_dir, opfname = ProcessOutput()
    safe_mkdir(output_dir)

    # weights
    weights = {}
    # landmark term
    weights['lmk'] = 1.0
    # shape regularizer (weight higher to regularize face shape more towards the mean)
    weights['shape'] = 1e-3
    # expression regularizer (weight higher to regularize facial expression more towards the mean)
    weights['expr'] = 1e-3
    # regularization of head rotation around the neck and jaw opening (weight higher for more regularization)
    weights['pose'] = 1e-2

    # number of shape and expression parameters (we do not recommend using too many parameters for fitting to sparse keypoints)
    shape_num = 100
    expr_num = 50

    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp'] = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3'] = 1e-4
    opt_options['maxiter'] = 2000
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d(lmk_3d=lmk_3d,  # input landmark 3d
                                      model=model,  # model
                                      lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                      weights=weights,  # weights for the objectives
                                      shape_num=shape_num, expr_num=expr_num, opt_options=opt_options)  # options

    # write result
    output_path = join(output_dir, opfname)
    write_simple_obj(mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False)


if __name__ == '__main__':
    lmk2flame()
