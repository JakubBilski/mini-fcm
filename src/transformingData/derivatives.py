import copy


def transform(xses_series, max_order):
    transformed_xses_series = []
    input_size = len(xses_series[0][0])
    for xs in xses_series:
        transformed_xs = copy.deepcopy(xs)
        derivatives = xs
        for order in range(1, max_order+1):
            derivatives = [[
                derivatives[i][j]-derivatives[i+1][j]
                for j in range(input_size)]
                for i in range(len(derivatives)-1)]
            for i in range(len(derivatives)):
                transformed_xs[i+order].extend(derivatives[i])
        transformed_xses_series.append(transformed_xs[max_order:])

    return transformed_xses_series
