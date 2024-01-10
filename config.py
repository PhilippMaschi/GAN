import re

from torch import nn


def remove_bullet_points(tSeq):
    tSeq = re.sub('\([0-9]+\):', '', tSeq)
    return tSeq

def remove_spaces(tSeq):
    tSeq = tSeq.replace(' ', '')
    return tSeq

def add_commas(tSeq):
    tSeq = tSeq.replace(')', '),')[:-1]
    return tSeq

def add_module(tSeq):
    for item in set(re.findall('[a-zA-Z0-9]+\(', tSeq)):
        tSeq = tSeq.replace(item, f'nn.{item}')
    return tSeq

def tSeq_string_wrapper(tSeq: str):  #tSeq... string of torch Sequential
    tSeq = remove_bullet_points(tSeq)
    tSeq = remove_spaces(tSeq)
    tSeq = add_commas(tSeq)
    tSeq = add_module(tSeq)
    return tSeq

def create_config(model):
    gen = tSeq_string_wrapper(str(model.Gen.model))
    dis = tSeq_string_wrapper(str(model.Dis.model))
    clusterAlgo = model.cluster_algorithm
    clusterLabel = str(model.cluster_label)
    batchSize = str(model.batchSize)
    noiseDim = str(model.dimNoise)
    lrGen = str(model.lr_gen)
    lrDis = str(model.lr_dis)
    nProfilesTrain = str(model.n_profiles_trained_on)
    config = f'Generator\n{gen}\n\n' \
             f'Discriminator\n{dis}\n\n' \
             f'ClusterAlgorithm\n{clusterAlgo}\n\n' \
             f'ClusterLabel\n{clusterLabel}\n\n' \
             f'BatchSize\n{batchSize}\n\n' \
             f'NoiseDimension\n{noiseDim}\n\n' \
             f'GeneratorLearningRate\n{lrGen}\n\n' \
             f'DiscriminatorLearningRate\n{lrDis}\n\n' \
             f'NumberOfProfilesTrainedOn\n{nProfilesTrain}'
    return config

def export_config(config, folderName):
    with open(f'{folderName}/config.txt', 'w') as file:
        file.write(config)

def import_config(model_folder):
    config = '\n\n' + open(model_folder + '/config.txt', 'r').read() + '\n'
    config = re.split(r'\n\n[a-zA-Z]+\n', config)[1:]
    config = {
        'seq_gen': eval(config[0]),
        'seq_dis': eval(config[1]),
        'cluster_algorithm': config[2],
        'cluster_label': int(config[3]),
        'batchSize': int(config[4]),
        'dimNoise': int(config[5]),
        'lr_gen': float(config[6]),
        'lr_dis': float(config[7]),
        'n_profiles_trained_on': int(config[8])
    }
    return config