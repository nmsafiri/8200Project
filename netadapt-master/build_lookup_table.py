import torch
import torch.nn as nn
import nets as models
import functions as fns
from argparse import ArgumentParser

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

NUM_CLASSES = 10

INPUT_DATA_SHAPE = (3, 224, 224)


'''
    `MIN_CONV_FEATURE_SIZE`: The sampled size of feature maps of layers (conv layer)
        along channel dimmension are multiples of 'MIN_CONV_FEATURE_SIZE'.
        
    `MIN_FC_FEATURE_SIZE`: The sampled size of features of FC layers are 
        multiples of 'MIN_FC_FEATURE_SIZE'.
'''
MIN_CONV_FEATURE_SIZE = 8
MIN_FC_FEATRE_SIZE    = 64

'''
    `MEASURE_EXPERIMENT_BATCH_SIZE`: the batch size of input data
        when running forward functions to measure empirical values.
    `MEASURE_EXPERIMENT_SAMPLE_TIMES`: the number of times to run the forward function of 
        a layer in order to get its empirical measurement.
'''
MEASURE_EXPERIMENT_BATCH_SIZE = 128
MEASURE_EXPERIMENT_SAMPLE_TIMES = 500


arg_parser = ArgumentParser(description='Build lookup tables')
arg_parser.add_argument('--resource', metavar='RESOURCE', default='ENERGY',
                    help='which resource to build lookup table for (supports `LATENCY` and `ENERGY`)')
arg_parser.add_argument('--dir', metavar='DIR', default='energy_lut/lut_alexnet.pkl',
                    help='path to saving energy lookup table')
arg_parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')


if __name__ == '__main__':
    
    args = arg_parser.parse_args()
    print(args)
    
    build_lookup_table = True
    lookup_table_path = args.dir
    resource_type = args.resource
    model_arch = args.arch

    if resource_type not in ['LATENCY', 'ENERGY']:
        raise ValueError("Lookup tables only used for `LATENCY` and `ENERGY` resource types")
    
    print('Load', model_arch)
    print('--------------------------------------')
    model = models.__dict__[model_arch](num_classes=NUM_CLASSES)
    network_def = fns.get_network_def_from_model(model, INPUT_DATA_SHAPE)
    for layer_name, layer_properties in network_def.items():
        print(layer_name)
        print('    ', layer_properties, '\n')
    print('-------------------------------------------')
    
    num_w = fns.compute_resource(network_def, 'WEIGHTS')
    flops = fns.compute_resource(network_def, 'FLOPS')
    num_param = fns.compute_resource(network_def, 'WEIGHTS')
    print('Number of FLOPs:      ', flops)
    print('Number of weights:    ', num_w)
    print('Number of parameters: ', num_param)
    print('-------------------------------------------')
    
    model = model.cuda()
    if resource_type == 'LATENCY':
        print('Building latency lookup table for', 
              torch.cuda.get_device_name())
        if build_lookup_table:
            fns.build_lookup_table(network_def, lookup_table_path=lookup_table_path, resource_type=resource_type, 
                min_fc_feature_size=MIN_FC_FEATRE_SIZE, 
                min_conv_feature_size=MIN_CONV_FEATURE_SIZE, 
                measure_experiment_batch_size=MEASURE_EXPERIMENT_BATCH_SIZE,
                measure_experiment_sample_times=MEASURE_EXPERIMENT_SAMPLE_TIMES,
                verbose=True)
        print('-------------------------------------------')
        print('Finish building latency lookup table.')
        print('    Device:', torch.cuda.get_device_name())
        print('    Model: ', model_arch)    
        print('-------------------------------------------')
        
        latency = fns.compute_resource(network_def, 'LATENCY', lookup_table_path)
        print('Computed latency:     ', latency)
        latency = fns.measure_latency(model, 
            [MEASURE_EXPERIMENT_BATCH_SIZE, *INPUT_DATA_SHAPE])
        print('Exact latency:        ', latency)    
    elif resource_type == 'ENERGY':
        print('Building energy lookup table for', 
              torch.cuda.get_device_name())
        if build_lookup_table:
            fns.build_lookup_table(network_def, lookup_table_path=lookup_table_path, resource_type=resource_type, 
                min_fc_feature_size=MIN_FC_FEATRE_SIZE, 
                min_conv_feature_size=MIN_CONV_FEATURE_SIZE, 
                measure_experiment_batch_size=MEASURE_EXPERIMENT_BATCH_SIZE,
                measure_experiment_sample_times=MEASURE_EXPERIMENT_SAMPLE_TIMES,
                verbose=True)
        print('-------------------------------------------')
        print('Finish building energy lookup table.')
        print('    Device:', torch.cuda.get_device_name())
        print('    Model: ', model_arch)    
        print('-------------------------------------------')
        
        energy = fns.compute_resource(network_def, 'ENERGY', lookup_table_path)
        print('Computed energy:     ', energy)
        energy = fns.measure_energy(model, 
            [MEASURE_EXPERIMENT_BATCH_SIZE, *INPUT_DATA_SHAPE])
        print('Exact energy:        ', energy)    

