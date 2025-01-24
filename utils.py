from collections import Counter, defaultdict
from timeit import default_timer as timer
import torch as pt
import numpy as np
import opt_einsum as oe
import sesum.sr as sr


def get_remapped_expression(first_indices, second_indices, unique_indices, out):
    asci_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if len(unique_indices) > len(asci_chars):
        return f"{first_indices},{second_indices}->{out}"
    char_map = {char: asci_chars[i] for i, char in enumerate(unique_indices)}
    first_indices_remapped = "".join(char_map[char] for char in first_indices)
    second_indices_remapped = "".join(char_map[char] for char in second_indices)
    out_remapped = "".join(char_map[char] for char in out)

    remapped_expression = (
        f"{first_indices_remapped},{second_indices_remapped}->{out_remapped}"
    )

    return remapped_expression


def get_stable_unique_indices(first_indices, second_indices):
    unique_indices = first_indices
    visited = set(unique_indices)
    for char in second_indices:
        if char not in visited:
            unique_indices += char
            visited.add(char)
    return unique_indices


def get_output_and_expression(
    histogram, first_indices, second_indices, is_last=False, out=None
):
    visited = set()
    unique_indices = []
    for char in first_indices + second_indices:
        if char not in visited:
            unique_indices.append(char)
            visited.add(char)
        histogram[char] -= 1

    if not is_last:
        out = "".join(char for char in unique_indices if histogram[char] > 0)

    for char in out:
        histogram[char] += 1

    remapped_expression = get_remapped_expression(
        first_indices, second_indices, unique_indices, out
    )

    return out, remapped_expression


def rewrite_to_dim2(format_string, tensors):
    inputs, output = format_string.split("->")
    inputs = inputs.split(",")

    shapes = [tensor.shape for tensor in tensors]
    size_dict = {}
    dim_remap = {}
    unique_indices = set(format_string)
    next_index = len(unique_indices) + 200
    remapped_inputs = []
    remapped_tensors = []
    new_indices_set = set()
    for id, (input, shape) in enumerate(zip(inputs, shapes)):
        needs_padding = False
        paddings = []
        for char, size in zip(input, shape):
            if char in size_dict:
                assert size_dict[char] == size
            size_dict[char] = size
            if size != 2:
                needs_padding = True
                num_new_indices = max(1, int(np.ceil(np.log2(size))))
                pad_size = 2**num_new_indices - size
                paddings.append((0, int(pad_size)))
                if char not in dim_remap:
                    new_indices = "".join(
                        [oe.get_symbol(next_index + i) for i in range(num_new_indices)]
                    )
                    assert new_indices_set.intersection(set(new_indices)) == set()
                    new_indices_set = new_indices_set.union(set(new_indices))
                    assert unique_indices.intersection(set(new_indices)) == set()
                    next_index += num_new_indices
                    dim_remap[char] = new_indices
            else:
                paddings.append((0, 0))
                dim_remap[char] = char

        if needs_padding:
            remapped_input = "".join([dim_remap[char] for char in input])
            remapped_inputs.append(remapped_input)
            remapped_tensors.append(
                np.pad(
                    tensors[len(remapped_tensors)],
                    paddings,
                    mode="constant",
                    constant_values=0,
                ).reshape(tuple([2] * len(remapped_input)))
            )
        else:
            remapped_inputs.append("".join([dim_remap[char] for char in input]))
            remapped_tensors.append(tensors[len(remapped_tensors)])

    remapped_output = "".join([dim_remap[char] for char in output])

    remapped_format_string = ",".join(remapped_inputs) + "->" + remapped_output
    return remapped_format_string, remapped_tensors


def annotate_ssa_path(format_string, ssa_path, tensors):
    """Annote an SSA path with their pairwise einsum format string and additional metadata

    Args:
        format_string (str): The format string representing the einsum expression.
        ssa_path (list): A list of tuples representing the SSA path indices.
        tensors (list): The list of tensors
    Returns:
        list: Annotated SSA path, where each element is a tuple containing the indices,
            pairwise format_string, log size of the output and number of elements in all reminaining tensors
            for each step in the SSA path.
    """
    inputs, output = format_string.split("->")
    inputs = inputs.split(",")
    assert (
        len(inputs) >= 2
    ), "Einsum expressions involving just one Tensor are not supported."
    format_string = format_string.replace(" ", "")
    histogram = Counter(format_string)

    shapes = [tensor.shape for tensor in tensors]
    size_dict = {}
    for input, shape in zip(inputs, shapes):
        for char, size in zip(input, shape):
            if char in size_dict:
                assert size_dict[char] == size
            size_dict[char] = size

    annotated_ssa_path = []

    index = 0
    for first, second in ssa_path:
        index += 1
        t1 = inputs[first]
        t2 = inputs[second]

        t3, pairwise_expression = get_output_and_expression(
            histogram, t1, t2, is_last=index == len(ssa_path), out=output
        )

        t3_size = np.prod([size_dict[char] for char in t3])
        t3_log_size = np.log2(t3_size)
        annotated_ssa_path.append(
            (
                first,
                second,
                pairwise_expression,
                t3_log_size,
                inputs[first],
                inputs[second],
            )
        )
        inputs.append(t3)
    return annotated_ssa_path


def sort_by_size(annotated_ssa_path, tensors, include_meta=True):
    ssa_id = len(tensors)
    with_ssa_id = []
    for tuple in annotated_ssa_path:
        with_ssa_id.append(tuple + (ssa_id,))
        ssa_id += 1

    size_sorted = sorted(with_ssa_id, key=lambda x: x[3])

    new_ssa_id = len(tensors)
    map_ssa_id = {i: i for i in range(new_ssa_id)}
    waiting = defaultdict(lambda: set())
    waits_for = {}
    waiting_tuple = {}
    sorted_ssa_path = []
    for tuple in size_sorted:
        first, second, pairwise_expression, t3_log_size, i1, i2, old_ssa_id = tuple
        has_to_wait = False
        if first not in map_ssa_id:
            has_to_wait = True
            waits_for[first] = old_ssa_id
            waiting[old_ssa_id].add(first)
            waiting_tuple[old_ssa_id] = tuple
        if second not in map_ssa_id:
            has_to_wait = True
            waits_for[second] = old_ssa_id
            waiting[old_ssa_id].add(second)
            waiting_tuple[old_ssa_id] = tuple
        if not has_to_wait:
            new_first = map_ssa_id[first]
            new_second = map_ssa_id[second]
            sorted_ssa_path.append(
                (new_first, new_second, pairwise_expression, t3_log_size, i1, i2)
            )
            map_ssa_id[old_ssa_id] = new_ssa_id
            finished = old_ssa_id
            new_ssa_id += 1
            while finished in waits_for:
                waits_for_finished = waits_for[finished]
                del waits_for[finished]
                waiting[waits_for_finished].remove(finished)
                if len(waiting[waits_for_finished]) == 0:
                    first, second, pairwise_expression, t3_log_size, i1, i2, _ = (
                        waiting_tuple[waits_for_finished]
                    )
                    new_first = map_ssa_id[first]
                    new_second = map_ssa_id[second]
                    sorted_ssa_path.append(
                        (
                            new_first,
                            new_second,
                            pairwise_expression,
                            t3_log_size,
                            i1,
                            i2,
                        )
                    )
                    finished = waits_for_finished
                    map_ssa_id[finished] = new_ssa_id
                    new_ssa_id += 1

    assert len(sorted_ssa_path) == len(annotated_ssa_path)
    if not include_meta:
        return [
            (first, second, expression)
            for (
                first,
                second,
                expression,
                *r,
            ) in sorted_ssa_path
        ]
    return sorted_ssa_path


def hybrid_einsum(
    annotated_ssa_path, tensors, np_dtype, output, density_estimation_start=12
):

    original_dtype = tensors[0].dtype

    step = 0
    ssa_id = len(tensors)
    t3_log_size = 0
    last_ssa_id = len(tensors) + len(annotated_ssa_path)

    # Contract until density estimation start
    while ssa_id < last_ssa_id:
        (
            first,
            second,
            expression,
            t3_log_size,
            *r,
        ) = annotated_ssa_path[step]
        if t3_log_size > density_estimation_start:
            break
        t1, t2 = tensors[first], tensors[second]
        t3 = pt.einsum(expression, t1, t2)
        if t3.dtype != original_dtype:
            # cast to original dtype
            t3 = t3.type(original_dtype)
        tensors.append(t3)
        tensors[first] = None
        tensors[second] = None
        ssa_id += 1
        step += 1

    if ssa_id == last_ssa_id:
        return tensors[-1], False, 0

    d_s = timer()
    non_zeros = {}
    num_elements = 0
    num_nonzeroes = 0
    for id, tensor in enumerate(tensors):
        if tensor is None:
            continue
        numel = tensor.numel()
        num_elements += numel
        non_zero = tensor.count_nonzero()
        non_zeros[id] = non_zero
        num_nonzeroes += non_zero

    density = (num_nonzeroes / num_elements).item()
    density_runtime = timer() - d_s
    density_estimation_threshold = density_estimation_start + 1
    to_count = set()

    while density > 0.05 and density < 0.95 and ssa_id < last_ssa_id:
        while ssa_id < last_ssa_id:
            (
                first,
                second,
                expression,
                t3_log_size,
                i1,
                i2,
            ) = annotated_ssa_path[step]
            if t3_log_size > density_estimation_threshold:
                break
            t1, t2 = tensors[first], tensors[second]
            t3 = pt.einsum(expression, t1, t2)
            num_elements = num_elements + t3.numel() - t1.numel() - t2.numel()
            if first in non_zeros:
                num_nonzeroes = num_nonzeroes - non_zeros[first]
                del non_zeros[first]
            else:
                assert first in to_count
            if second in non_zeros:
                num_nonzeroes = num_nonzeroes - non_zeros[second]
                del non_zeros[second]
            else:
                assert second in to_count
            if t3.dtype != original_dtype:
                # cast to original dtype
                t3 = t3.type(original_dtype)
            tensors.append(t3)
            to_count.add(ssa_id)
            tensors[first] = None
            tensors[second] = None
            to_count.discard(first)
            to_count.discard(second)
            step += 1
            ssa_id += 1
        if ssa_id == last_ssa_id:
            # print(non_zeros, to_count)
            return tensors[-1], False, density_runtime

        d_s = timer()
        for id in to_count:
            non_zero = tensors[id].count_nonzero()
            num_nonzeroes += non_zero
            non_zeros[id] = non_zero
        to_count = set()
        density = (num_nonzeroes / num_elements).item()
        density_runtime += timer() - d_s
        print("density", density_estimation_threshold, density, density_runtime)  #

        density_estimation_threshold += 1

    if ssa_id == last_ssa_id:
        return tensors[-1], False, density_runtime

    if density <= 0.05:
        new_id_tensors = [(id, t) for id, t in enumerate(tensors) if t != None]
        new_tensors = [t[1] for t in new_id_tensors]
        old_id_to_new = {id: new_id for new_id, (id, t) in enumerate(new_id_tensors)}
        new_path = []
        new_inputs = [None] * len(new_tensors)
        new_ssa_id = len(new_tensors)
        for first, second, expression, t3_log_size, i1, i2 in annotated_ssa_path[step:]:
            first_new_id = old_id_to_new[first]
            if first_new_id < len(new_tensors):
                new_inputs[first_new_id] = i1
            second_new_id = old_id_to_new[second]
            if second_new_id < len(new_tensors):
                new_inputs[second_new_id] = i2
            new_path.append((first_new_id, second_new_id))
            old_id_to_new[ssa_id] = new_ssa_id
            ssa_id += 1
            new_ssa_id += 1

        new_format_string = ",".join(new_inputs) + "->" + output
        print("Switched to sparse")
        result = sr.sesum(
            new_format_string,
            *new_tensors,
            path=new_path,
            backend="sparse",
            debug=False,
            dtype=np_dtype,
        )
        return result, True, density_runtime
    else:
        print("Decided for dense")
        while ssa_id < last_ssa_id:
            (first, second, expression, t3_log_size, i1, i2) = annotated_ssa_path[step]
            t1, t2 = tensors[first], tensors[second]
            t3 = pt.einsum(expression, t1, t2)
            if t3.dtype != original_dtype:
                # cast to original dtype
                t3 = t3.type(original_dtype)
            tensors.append(t3)
            tensors[first] = None
            tensors[second] = None
            ssa_id += 1
            step += 1
        return tensors[-1], False, density_runtime


def get_peak_mem(format_string, annotated_ssa_path, *tensors, size_dict=None):
    inputs, output = format_string.split("->")
    inputs = inputs.split(",")
    if size_dict is None:
        shapes = [tensor.shape for tensor in tensors]
        size_dict = {}
        for input, shape in zip(inputs, shapes):
            for char, size in zip(input, shape):
                if char in size_dict:
                    assert size_dict[char] == size
                size_dict[char] = size
    else:
        shapes = [tuple([size_dict[char] for char in input]) for input in inputs]

    current_memory = 0
    peak_mem = 0

    sizes = []
    size_0 = np.prod([size_dict[char] for char in inputs[0]])
    element_size = tensors[0].nbytes / size_0

    for input in inputs:
        size = element_size * np.prod([size_dict[char] for char in input])
        current_memory += size
        sizes.append(size)

    peak_mem = current_memory

    for first, second, expression in annotated_ssa_path:
        t1, t2 = shapes[first], shapes[second]
        _, path_info = oe.contract_path(expression, t1, t2, shapes=True)
        output = expression.split("->")[1]
        output_shape = tuple([size_dict[char] for char in output])
        shapes.append(output_shape)
        output_size = element_size * np.prod([size_dict[char] for char in output])
        sizes.append(output_size)
        current_memory = current_memory + output_size

        if current_memory > peak_mem:
            peak_mem = current_memory
        current_memory = current_memory - sizes[first] - sizes[second]
    return peak_mem
