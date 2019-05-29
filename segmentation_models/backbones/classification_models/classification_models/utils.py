from keras.utils import get_file


def find_weights(weights_collection, model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, weights_collection))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(weights_collection, model, dataset, classes, include_top):
    weights = find_weights(weights_collection, model.name, dataset, include_top)
    if weights:
        weights = weights[0]
        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))
        cache_dir = "/home/ubuntu/data1.5TB/ImageNetWeights"
        weights_path = get_file(weights['name'], weights['url'],
                                cache_subdir=weights['dataset'], md5_hash=weights['md5'], cache_dir=cache_dir)
        print("load weight from {}".format(weights_path))
        # model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))
    pass
