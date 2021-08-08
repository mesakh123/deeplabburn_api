from typing import Tuple, List, Dict

import torch
from torch import Tensor
from torch.nn import DataParallel


class Bunch(List):

    def tolist(self) -> List:
        return list(self)


class BunchDataParallel(DataParallel):

    def scatter(self, inputs: Tuple, kwargs: Dict, device_ids: List[int]) -> Tuple[List, List]:
        # NOTE:
        #       For example, we are using 2 devices; original `DataParallel` will try to scatter a tensor into
        #       chunks for each element even the tensor is inside a list or tuple.
        #
        #       By this way
        #
        #           # cuda:0
        #           inputs = (
        #               torch.tensor([1, 2, 3, 4]),                 # Modality 1
        #               [
        #                   torch.tensor([10, 20, 30, 40]),         # Modality 2
        #                   torch.tensor([100, 200, 300, 400])      # Modality 3
        #               ]
        #           )
        #
        #           > `inputs` is always a tuple even one or zero parameters were passed into `forward`
        #
        #       is scattered to
        #
        #           scattered_inputs = [
        #               # cuda:0
        #               (
        #                   torch.tensor([1, 2]),               # Modality 1
        #                   [
        #                       torch.tensor([10, 20]),         # Modality 2
        #                       torch.tensor([100, 200])        # Modality 3
        #                   ]
        #               ),
        #               # cuda:1
        #               (
        #                   torch.tensor([3, 4]),               # Modality 1
        #                   [
        #                       torch.tensor([30, 40]),         # Modality 2
        #                       torch.tensor([300, 400])        # Modality 3
        #                   ]
        #               )
        #           ]
        #
        #       However, in some cases, we are hoping to scatter the tuple or list according to its elements, such as
        #
        #           inputs = (
        #               # Modality 1
        #               Bunch([
        #                   torch.tensor(1),
        #                   torch.tensor([2, 2]),
        #                   torch.tensor([[3, 3],
        #                                 [4, 4]])
        #               ]),
        #               [
        #                   # Modality 2
        #                   Bunch([
        #                       torch.tensor(10),
        #                       torch.tensor(20),
        #                       torch.tensor(30),
        #                       torch.tensor(40)
        #                   ]),
        #                   (
        #                       # Modality 3
        #                       Bunch([
        #                           torch.tensor(100),
        #                           torch.tensor(200),
        #                           torch.tensor(300),
        #                           torch.tensor(400)
        #                       ]),
        #                       # Modality 4
        #                       Bunch([
        #                           torch.tensor(1000),
        #                           torch.tensor(2000),
        #                           torch.tensor(3000),
        #                           torch.tensor(4000)
        #                       ])
        #                   )
        #               ]
        #           )
        #
        #       By overwritten `scatter` in `BunchDataParallel`, above one will be scattered to
        #
        #           scattered_inputs = [
        #               # cuda:0
        #               (
        #                   # Modality 1
        #                   Bunch([
        #                       torch.tensor(1),
        #                       torch.tensor([2, 2])
        #                   ]),
        #                   [
        #                       # Modality 2
        #                       Bunch([
        #                           torch.tensor(10),
        #                           torch.tensor(20)
        #                       ]),
        #                       (
        #                           # Modality 3
        #                           Bunch([
        #                               torch.tensor(100),
        #                               torch.tensor(200)
        #                           ]),
        #                           # Modality 4
        #                           Bunch([
        #                               torch.tensor(1000),
        #                               torch.tensor(2000)
        #                           ]),
        #                       )
        #                   ]
        #               ),
        #               # cuda:1
        #               (
        #                   # Modality 1
        #                   Bunch([
        #                       torch.tensor([[3, 3],
        #                                     [4, 4]])
        #                   ]),
        #                   [
        #                       # Modality 2
        #                       Bunch([
        #                           torch.tensor(30),
        #                           torch.tensor(40)
        #                       ]),
        #                       (
        #                           # Modality 3
        #                           Bunch([
        #                               torch.tensor(300),
        #                               torch.tensor(400)
        #                           ]),
        #                           # Modality 4
        #                           Bunch([
        #                               torch.tensor(3000),
        #                               torch.tensor(4000)
        #                           ]),
        #                       )
        #                   ]
        #               )
        #           ]
        #
        #       This can be a better way to handle the arbitrary shape of tensors.

        def scatter_map(obj):
            if isinstance(obj, Bunch):
                num_devices = len(self.device_ids)
                elements = obj
                num_elements = len(elements)
                num_chunks = num_devices
                chunk_sizes = [num_elements // num_chunks] * num_chunks
                for i in range(num_elements % num_chunks):
                    chunk_sizes[i] += 1
                slice_ends = torch.tensor(chunk_sizes).cumsum(dim=0).tolist()
                slice_starts = [0] + slice_ends[:-1]
                elements_chunks = [elements[i:j] for i, j in zip(slice_starts, slice_ends)]
                elements_chunks = [Bunch([element.to(torch.device('cuda', self.device_ids[i]))
                                          for element in elements])
                                   for i, elements in enumerate(elements_chunks)]
                scattered_obj = tuple(elements for elements in elements_chunks if elements)
                return scattered_obj
            if isinstance(obj, tuple):
                return list(zip(*map(scatter_map, obj))) if len(obj) > 0 else []
            if isinstance(obj, list):
                return list(map(list, zip(*map(scatter_map, obj)))) if len(obj) > 0 else []
            if isinstance(obj, dict):
                return list(map(type(obj), zip(*map(scatter_map, obj.items())))) if len(obj) > 0 else []
            return [obj for _ in device_ids]

        try:
            scattered_inputs = scatter_map(inputs)
            scattered_kwargs = scatter_map(kwargs)

            if len(scattered_inputs) < len(scattered_kwargs):
                scattered_inputs.extend([() for _ in range(len(scattered_kwargs) - len(scattered_inputs))])
            elif len(scattered_kwargs) < len(scattered_inputs):
                scattered_kwargs.extend([{} for _ in range(len(scattered_inputs) - len(scattered_kwargs))])
        finally:
            scatter_map = None

        return scattered_inputs, scattered_kwargs

    def gather(self, outputs: List[Tuple], target_device: int, dim: int = 0):
        # NOTE:
        #       Original `DataParallel`
        #
        #           # cuda:0
        #           outputs = [
        #               # cuda:0
        #               (
        #                   torch.tensor([1, 2]),               # Modality 1
        #                   [
        #                       torch.tensor([10, 20]),         # Modality 2
        #                       torch.tensor([100, 200])        # Modality 3
        #                   ]
        #               ),
        #               # cuda:1
        #               (
        #                   torch.tensor([3, 4]),               # Modality 1
        #                   [
        #                       torch.tensor([30, 40]),         # Modality 2
        #                       torch.tensor([300, 400])        # Modality 3
        #                   ]
        #               )
        #           ]
        #
        #       is gathered to
        #
        #           # cuda:0
        #           gathered_outputs = (
        #               torch.tensor([1, 2, 3, 4]),                 # Modality 1
        #               [
        #                   torch.tensor([10, 20, 30, 40]),         # Modality 2
        #                   torch.tensor([100, 200, 300, 400])      # Modality 3
        #               ]
        #           )
        #
        #       By overwritten `gather` in `BunchDataParallel`
        #
        #           outputs = [
        #               # cuda:0
        #               (
        #                   # Modality 1
        #                   Bunch([
        #                       torch.tensor(1),
        #                       torch.tensor([2, 2])
        #                   ]),
        #                   [
        #                       # Modality 2
        #                       Bunch([
        #                           torch.tensor(10),
        #                           torch.tensor(20)
        #                       ]),
        #                       (
        #                           # Modality 3
        #                           Bunch([
        #                               torch.tensor(100),
        #                               torch.tensor(200)
        #                           ]),
        #                           # Modality 4
        #                           Bunch([
        #                               torch.tensor(1000),
        #                               torch.tensor(2000)
        #                           ]),
        #                       )
        #                   ]
        #               ),
        #               # cuda:1
        #               (
        #                   # Modality 1
        #                   Bunch([
        #                       torch.tensor([[3, 3],
        #                                     [4, 4]])
        #                   ]),
        #                   [
        #                       # Modality 2
        #                       Bunch([
        #                           torch.tensor(30),
        #                           torch.tensor(40)
        #                       ]),
        #                       (
        #                           # Modality 3
        #                           Bunch([
        #                               torch.tensor(300),
        #                               torch.tensor(400)
        #                           ]),
        #                           # Modality 4
        #                           Bunch([
        #                               torch.tensor(3000),
        #                               torch.tensor(4000)
        #                           ]),
        #                       )
        #                   ]
        #               )
        #           ]
        #
        #       is gathered to
        #
        #           # cuda:0
        #           gathered_outputs = (
        #               # Modality 1
        #               Bunch([
        #                   torch.tensor(1),
        #                   torch.tensor([2, 2]),
        #                   torch.tensor([[3, 3],
        #                                 [4, 4]])
        #               ]),
        #               [
        #                   # Modality 2
        #                   Bunch([
        #                       torch.tensor(10),
        #                       torch.tensor(20),
        #                       torch.tensor(30),
        #                       torch.tensor(40)
        #                   ]),
        #                   (
        #                       # Modality 3
        #                       Bunch([
        #                           torch.tensor(100),
        #                           torch.tensor(200),
        #                           torch.tensor(300),
        #                           torch.tensor(400)
        #                       ]),
        #                       # Modality 4
        #                       Bunch([
        #                           torch.tensor(1000),
        #                           torch.tensor(2000),
        #                           torch.tensor(3000),
        #                           torch.tensor(4000)
        #                       ])
        #                   )
        #               ]
        #           )

        def gather_map(outputs):
            out = outputs[0]
            if isinstance(out, Bunch):
                return Bunch([o.to(torch.device('cuda', target_device))
                              for out in outputs for o in out])

            if out is None:
                return None
            if isinstance(out, dict):
                if not all((len(out) == len(d) for d in outputs)):
                    raise ValueError('All dicts must have the same number of keys')
                return type(out)(((k, gather_map([d[k] for d in outputs]))
                                  for k in out))
            return type(out)(map(gather_map, zip(*outputs)))

        try:
            gathred_outputs = gather_map(outputs)
        finally:
            gather_map = None

        return gathred_outputs


if __name__ == '__main__':
    def main():
        from threading import Lock

        assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, \
            'This example is expected to run with at least 2 CUDA devices.'

        lock = Lock()
        device_ids = [0, 1]
        device = torch.device('cuda', device_ids[0])

        # region ===== DataParallel Case =====
        def data_parallel_case():
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x: Tensor, x_list: List[Tensor]):
                    with lock:
                        print('x =', x)
                        print('x_list =', x_list)
                    return x, x_list

            num_devices = torch.cuda.device_count()
            print('num_devices:', num_devices)

            print('===== DataParallel =====')
            model = Model()
            model = DataParallel(model, device_ids).to(device)
            inputs = (
                torch.tensor([1, 2, 3, 4]),
                [
                    torch.tensor([10, 20, 30, 40]),
                    torch.tensor([100, 200, 300, 400])
                ]
            )
            print('inputs =', inputs)
            outputs = model.forward(*inputs)
            print('outputs =', outputs)

        data_parallel_case()
        # endregion ==========================

        # region ===== BunchDataParallel Case 1 =====
        def bunch_data_parallel_case_1():
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, input1: Bunch, input2: List):
                    with lock:
                        print('input1 =', input1)
                        print('input2 =', input2)
                    return input1, input2

            print('===== BunchDataParallel (1) =====')
            model = Model()
            model = BunchDataParallel(model, device_ids).to(device)
            inputs = (
                Bunch([
                    torch.tensor(1),
                    torch.tensor([2, 2]),
                    torch.tensor([[3, 3],
                                  [4, 4]])
                ]),
                [
                    Bunch([
                        torch.tensor(10),
                        torch.tensor(20),
                        torch.tensor(30),
                        torch.tensor(40)
                    ]),
                    (
                        Bunch([
                            torch.tensor(100),
                            torch.tensor(200),
                            torch.tensor(300),
                            torch.tensor(400)
                        ]),
                        Bunch([
                            torch.tensor(1000),
                            torch.tensor(2000),
                            torch.tensor(3000),
                            torch.tensor(4000)
                        ])
                    )
                ]
            )
            print('inputs =', inputs)
            outputs = model.forward(*inputs)
            print('outputs =', outputs)

        bunch_data_parallel_case_1()
        # endregion ===============================

        # region ===== BunchDataParallel Case 2 =====
        def bunch_data_parallel_case_2():
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, input1: Bunch):
                    with lock:
                        print('input1 =', input1)
                    return input1

            print('===== BunchDataParallel (2) =====')
            model = Model()
            model = BunchDataParallel(model, device_ids).to(device)
            input1 = Bunch([
                torch.tensor(1),
                torch.tensor([2, 2]),
                torch.tensor([[3, 3],
                              [4, 4]])
            ])
            print('input1 =', input1)
            output = model.forward(input1)
            print('output =', output)

        bunch_data_parallel_case_2()
        # endregion ===============================

        # region ===== BunchDataParallel Case 3 =====
        def bunch_data_parallel_case_3():
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, input1: Bunch):
                    with lock:
                        print('input1 =', input1)
                    return input1

            print('===== BunchDataParallel (3) =====')
            model = Model()
            model = BunchDataParallel(model, device_ids).to(device)
            input1 = Bunch([
                torch.tensor(1)
            ])
            print('input1 =', input1)
            output = model.forward(input1)
            print('output =', output)

        bunch_data_parallel_case_3()
        # endregion ===============================

    main()
