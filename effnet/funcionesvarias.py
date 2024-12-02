from math import ceil

def _make_divisible(v, divisor, min_value):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def adjust_channels(channels, width_mult, min_value):
    return _make_divisible(channels * width_mult, 8, min_value)

def adjust_depth(num_layers, depth_mult):
    return int(ceil(num_layers * depth_mult))

