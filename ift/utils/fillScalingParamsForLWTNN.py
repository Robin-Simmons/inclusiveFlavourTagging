import argparse

from ift.utils.utils import populateScalingForLWTNN

from ift.createTrainingFiles.constants import branch_names

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--varFile", type = str, dest = "varFile", default = "vars.json", help = 'LWTNN variables file in which to fill the scaling parameters (JSON).')
    argParser.add_argument("--scaleFile", type = str, dest = "scaleFile", default = "scaling.json", help = 'Saved scaling parameter dictionary (JSON).')

    args = argParser.parse_args()

    features = branch_names['track_features']

    populateScalingForLWTNN(features, args.varFile, args.scaleFile)
