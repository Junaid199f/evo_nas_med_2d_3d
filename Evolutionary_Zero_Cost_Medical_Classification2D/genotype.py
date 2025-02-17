from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat normal_bottleneck reduce_bottleneck')

# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
#     'conv_7x1_1x7',
#     'conv_1x1_3x3',
# ]

PRIMITIVES = [
'max_pool_3x3', 'avg_pool_3x3',
    'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3',
    'dil_conv_5x5', 'conv_7x1_1x7',
    'inv_res_3x3', 'inv_res_5x5',
    'octave_conv_5x5','octave_conv_3x3',
    'mbconv_k7_t1','mbconv_k5_t1',
    'mbconv_k3_t1']

ATTN_PRIMIVIVES = [
    'Identity',
    'SE',
    'BAM',
    'CBAM',
    'GE_theta-plus',
    'DoubleAttention'
]
NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
    normal_bottleneck='',
    reduce_bottleneck=''
)

NASNet_AB = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
normal_bottleneck = '',
                    reduce_bottleneck = ''
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6],    normal_bottleneck='',
    reduce_bottleneck=''
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5],    normal_bottleneck='',
    reduce_bottleneck='')
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5],    normal_bottleneck='',
    reduce_bottleneck='')

SNAS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                        ('skip_connect', 0), ('dil_conv_3x3', 1),
                        ('skip_connect', 1), ('skip_connect', 0),
                        ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
                reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
                        ('max_pool_3x3', 1), ('skip_connect', 2),
                        ('skip_connect', 2), ('max_pool_3x3', 1),
                        ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6),    normal_bottleneck='',
    reduce_bottleneck='')

PDARTS = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6),    normal_bottleneck='',
    reduce_bottleneck='')
PC_DARTS = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6),    normal_bottleneck='',
    reduce_bottleneck='')
CDARTS = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6),    normal_bottleneck='',
    reduce_bottleneck='')
CARS_I = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 2),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 1),
            ('sep_conv_5x5', 3), ('dil_conv_3x3', 1), ('max_pool_3x3', 4)], reduce_concat=range(2, 6),    normal_bottleneck='',
    reduce_bottleneck='')
CARS_H = Genotype(
    normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0),
            ('skip_connect', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 2),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6),    normal_bottleneck='',
    reduce_bottleneck='')

DARTS = DARTS_V2

EA = Genotype(
    normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('inv_res_5x5', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 1), ('inv_res_5x5', 3), ('inv_res_3x3', 2), ('sep_conv_3x3', 4), ('max_pool_3x3', 3)],
    normal_concat=[5, 6],
    reduce=[('avg_pool_3x3', 0), ('inv_res_3x3', 0), ('inv_res_5x5', 1), ('dil_conv_3x3', 0), ('inv_res_3x3', 0),
            ('dil_conv_5x5', 0), ('max_pool_3x3', 3), ('skip_connect', 3), ('max_pool_3x3', 0), ('sep_conv_3x3', 0)],
    reduce_concat=[2, 4, 5, 6],    normal_bottleneck='',
    reduce_bottleneck='')
EEEA_A = Genotype(
    normal=[('max_pool_3x3', 0), ('inv_res_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('inv_res_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 4), ('skip_connect', 3)],
    normal_concat=[5, 6],
    reduce=[('skip_connect', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0),
            ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('dil_conv_3x3', 3), ('dil_conv_5x5', 0)],
    reduce_concat=[4, 5, 6],    normal_bottleneck='',
    reduce_bottleneck='')
EEEA_B = Genotype(
    normal=[('skip_connect', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('skip_connect', 1),
            ('skip_connect', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('sep_conv_3x3', 4)],
    normal_concat=[5, 6],
    reduce=[('skip_connect', 0), ('inv_res_5x5', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('inv_res_3x3', 1),
            ('skip_connect', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 3)],
    reduce_concat=[4, 5, 6],    normal_bottleneck='',
    reduce_bottleneck='')
EEEA_C = Genotype(
    normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('inv_res_5x5', 0), ('inv_res_3x3', 0), ('dil_conv_5x5', 2),
            ('inv_res_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)],
    normal_concat=[3, 5, 6],
    reduce=[('inv_res_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('inv_res_3x3', 0), ('dil_conv_5x5', 1),
            ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 3), ('inv_res_5x5', 4), ('inv_res_3x3', 2)],
    reduce_concat=[5, 6],    normal_bottleneck='',
    reduce_bottleneck='')

Att_DARTS = Genotype(
    normal=[('sep_conv_3x3', 0, 'DoubleAttention'), ('sep_conv_3x3', 1, 'CBAM'), ('sep_conv_3x3', 0, 'CBAM'),
            ('skip_connect', 2, 'CBAM'), ('skip_connect', 0, 'CBAM'), ('sep_conv_3x3', 1, 'CBAM'),
            ('skip_connect', 0, 'CBAM'), ('skip_connect', 2, 'CBAM')], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0, 'BAM'), ('dil_conv_3x3', 1, 'DoubleAttention'), ('avg_pool_3x3', 0, 'BAM'),
            ('skip_connect', 2, 'CBAM'), ('skip_connect', 2, 'CBAM'), ('avg_pool_3x3', 0, 'BAM'),
            ('skip_connect', 2, 'CBAM'), ('avg_pool_3x3', 0, 'BAM')], reduce_concat=range(2, 6),    normal_bottleneck='',
    reduce_bottleneck='')