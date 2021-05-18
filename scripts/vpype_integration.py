import vpype
import numpy as np
from svg import layer_name_to_id, layer_id_to_name


def to_vpype_lines(paths):
    lc = vpype.LineCollection()
    for path in paths:
        lc.append(path[:, 0] + path[:, 1] * 1.j)
    return lc


def from_vpype_lines(lines):
    results = []
    for line in lines:
        xs = np.real(line)
        ys = np.imag(line)
        results.append(np.stack((xs, ys), axis=1))

    return results


def to_vpype_document(layers):
    document = vpype.Document()
    for layer_name, layer in layers.items():
        if len(layer) == 0:
            continue
        lines = to_vpype_lines(layer)
        document.add(lines, layer_name_to_id(layer_name))
    return document


def from_vpype_document(doc):
    layers = {}
    for layer_id in doc.ids():
        paths = from_vpype_lines(doc[layer_id])
        layers[layer_id_to_name(layer_id)] = paths
    return layers
