import os

def reorder_files(list_a, list_b):
    """
    Returns ordered list_a according to the order presented in list_b
    """
    assert len(list_a) == len(list_b), "Input lists must have same length"

    ordering = list_b[:]
    temp = list_a[:]
    out = [None]*len(list_a)

    for i, elt in enumerate(ordering):
        head, tail = os.path.split(elt)
        cur_filename, _ = os.path.splitext(tail)
        for full_path in list_a:
            head_a, tail_a = os.path.split(full_path)
            filename, _ = os.path.splitext(tail_a)
            if filename == cur_filename + "_mask":
                out[i] = full_path

    return out