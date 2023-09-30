def get_params_range(gain):
        param_dict_0dB={
            'cff':                                               [1, 15, 7., 1, 15],
            'n1':                                                [2, 8, 4, 2, 8],
            'cspace':                                            [0, 1, 1, 0, 1],
            'wtransform':                                        [0, 1, 1, 0, 1],
            'neighborhood':                                      [4, 12, 8, 4, 12],
        }
        return locals()['param_dict_%sdB'%str(gain)]