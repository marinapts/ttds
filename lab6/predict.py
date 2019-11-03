import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('number', type=str)
    opts = parser.parse_args()

    model = opts.model + opts.number
    out_file = opts.number + opts.out_file
    print(model, out_file)
    svm_learn_cmd = 'svm_multiclass_linux64/svm_multiclass_learn -c 1000 results/feats.train ./outputs/' + model
    svm_predict_cmd = 'svm_multiclass_linux64/svm_multiclass_classify results/feats.test ./outputs/' + model + ' ./outputs/' + out_file

    os.system(svm_learn_cmd)
    os.system(svm_predict_cmd)
