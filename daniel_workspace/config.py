import re


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
    config = f'Generator\n{gen}\n\n' \
             f'Discriminator\n{dis}\n\n' \
             f'Batch size\n{model.batchSize}\n\n' \
             f'Loss function\n{model.lossFct.__str__()}\n\n' \
             f'Learning rate (Generator)\n{model.lrGen}\n\n' \
             f'Learning rate (Discriminator)\n{model.lrDis}\n\n' \
             f'Betas\n{str(model.betas)}\n\n' \
             f'Device\n{model.device.type}\n\n' \
             f'Epoch count\n{model.epochCount}\n\n' \
             f'Label (real)\n{model.labelReal}\n\n' \
             f'Label (fake)\n{model.labelReal}\n\n' \
             f'Noise dimension\n{model.dimNoise}\n\n'
    return config


def export_config(config, outputPath):
    with open(f'{outputPath}/config.txt', 'w') as file:
        file.write(config)


def config_wrapper(model, outputPath):
    config = create_config(model)
    export_config(config, outputPath)