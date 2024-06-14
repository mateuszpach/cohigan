from typing import Dict, List, Callable

import numpy as np
import torch

# All added by the authors


def to_binary(t: torch.Tensor, bits) -> torch.Tensor:
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(device=t.device, dtype=t.dtype)
    binary = t.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=torch.float)
    return binary


def from_binary(t: torch.Tensor) -> torch.Tensor:
    bits = t.shape[1]
    return torch.sum((t * 2 ** torch.arange(bits - 1, -1, -1)), dim=1)


class ConditionEncoding:
    @staticmethod
    def get_dim(c: int) -> int:
        """Return number of dimensions used by the encoding for c classes"""
        ...

    @staticmethod
    def generate_clear_gen_c(n: int, c: int) -> torch.Tensor:
        """Generate tensor of n encodings of c classes"""
        ...

    @staticmethod
    def get_mix_generators() -> Dict[str, Callable[[int, int], List[torch.Tensor]]]:
        """Return dict of mix strategy names with respecting generate methods"""
        ...

    @staticmethod
    def generate_mixed_gen_c_all(n: int, c: int) -> Dict[str, List[torch.Tensor]]:
        """Generate dict of all available mixed encodings.
           Keys are the mix strategies names.
           Values are lists of 2*c-1 (# tree nodes) tensors of n encodings of c classes."""
        ...

    @staticmethod
    def convert_clear_gen_c_to_cube_enc(gen_c: torch.Tensor) -> torch.Tensor:
        """Convert tensor of clear (leaf) encodings to cube encodings"""
        ...

    @staticmethod
    def convert_cube_enc_to_clear_gen_c(cube_enc: torch.Tensor) -> torch.Tensor:
        """Convert tensor of cube encodings to clear (leaf) encodings"""
        ...


class FlatConditionEncoding(ConditionEncoding):
    @staticmethod
    def get_dim(c: int) -> int:
        return c

    @staticmethod
    def generate_clear_gen_c(n: int, c: int) -> torch.Tensor:
        gen_c = torch.zeros((n, c))
        gen_c[torch.arange(0, n), torch.randint(low=0, high=c, size=(n,))] = 1
        return gen_c

    @staticmethod
    def generate_mixed_gen_c_norm_mean(norm_ord: float) -> Callable[[int, int], List[torch.Tensor]]:
        def f(n: int, c: int) -> List[torch.Tensor]:
            gen_c = [torch.zeros((n, c)) for _ in range(2 * c - 1)]
            for i in range(c - 1, 2 * c - 1):
                gen_c[i][:, i - (c - 1)] = 1
            for i in range(c - 2, -1, -1):
                gen_c[i] = gen_c[2 * i + 1] + gen_c[2 * i + 2]
                # gen_c[i] /= torch.linalg.vector_norm(gen_c[i], dim=1, ord=norm_ord).reshape(-1, 1)
                gen_c[i] /= torch.from_numpy(np.linalg.norm(gen_c[i], axis=1, ord=norm_ord)).reshape(-1, 1)
            return gen_c
        return f

    @staticmethod
    def get_mix_generators() -> Dict[str, Callable[[int, int], List[torch.Tensor]]]:
        return {
            'l1_mean': FlatConditionEncoding.generate_mixed_gen_c_norm_mean(1),
            'l2_mean': FlatConditionEncoding.generate_mixed_gen_c_norm_mean(2),
            'linf_mean': FlatConditionEncoding.generate_mixed_gen_c_norm_mean(float('inf'))
        }

    @staticmethod
    def generate_mixed_gen_c_all(n: int, c: int) -> Dict[str, List[torch.Tensor]]:
        return {
            name: mix_generator(n, c)
            for name, mix_generator in FlatConditionEncoding.get_mix_generators().items()
        }

    @staticmethod
    def convert_clear_gen_c_to_cube_enc(gen_c: torch.Tensor) -> torch.Tensor:
        leaves = torch.argmax(gen_c, dim=1)
        c = gen_c.shape[1]
        h = round(np.log2(c))
        return to_binary(leaves, h) * 2 - 1

    @staticmethod
    def convert_cube_enc_to_clear_gen_c(cube_enc: torch.Tensor) -> torch.Tensor:
        binary = (cube_enc + 1) / 2
        n = cube_enc.shape[0]
        c = 2 ** cube_enc.shape[1]
        gen_c = torch.zeros((n, c))
        gen_c[torch.arange(0, n), from_binary(binary).to(dtype=torch.long)] = 1
        return gen_c


class CubeConditionEncoding(ConditionEncoding):
    @staticmethod
    def get_dim(c: int) -> int:
        return round(np.log2(c))

    @staticmethod
    def generate_clear_gen_c(n: int, c: int) -> torch.Tensor:
        h = round(np.log2(c))
        gen_c = torch.randint(low=0, high=2, size=(n, h), dtype=torch.float) * 2 - 1
        return gen_c

    @staticmethod
    def generate_mixed_gen_c_norm_mean(norm_ord: float) -> Callable[[int, int], List[torch.Tensor]]:
        def f(n: int, c: int) -> List[torch.Tensor]:
            h = round(np.log2(c))
            gen_c = [torch.zeros((n, h)) for _ in range(2 * c - 1)]
            for i in range(c - 1, 2 * c - 1):
                gen_c[i] = to_binary(torch.ones((n,), dtype=torch.int) * i - (c - 1), h) * 2 - 1
            for i in range(c - 2, -1, -1):
                gen_c[i] = gen_c[2 * i + 1] + gen_c[2 * i + 2]
                # gen_c[i] /= torch.linalg.vector_norm(gen_c[i], dim=1, ord=norm_ord).reshape(-1, 1
                gen_c[i] /= torch.from_numpy(np.linalg.norm(gen_c[i], axis=1, ord=norm_ord)).reshape(-1, 1)
                # gen_c[i] *= torch.linalg.vector_norm(torch.ones_like(gen_c[i]), dim=1, ord=norm_ord).reshape(-1, 1)
                gen_c[i] *= torch.from_numpy(np.linalg.norm(torch.ones_like(gen_c[i]), axis=1, ord=norm_ord)).reshape(-1, 1)
                # gen_c[i] = torch.nan_to_num(gen_c[i])
                gen_c[i] = torch.from_numpy(np.nan_to_num(gen_c[i]))
            return gen_c
        return f

    @staticmethod
    def generate_mixed_gen_c_0_lca_then_rand_path(n: int, c: int) -> List[torch.Tensor]:
        h = round(np.log2(c))
        gen_c = [torch.zeros((n, h)) for _ in range(2 * c - 1)]
        for i in range(c - 1, 2 * c - 1):
            gen_c[i] = to_binary(torch.ones((n,), dtype=torch.int) * i - (c - 1), h) * 2 - 1
        for i in range(c - 2, -1, -1):
            for j in range(n):
                if torch.randint(low=0, high=2, size=()).item() == 0:
                    gen_c[i][j, :] = gen_c[2 * i + 1][j, :]
                else:
                    gen_c[i][j, :] = gen_c[2 * i + 2][j, :]
        for i in range(h):
            for j in range(2 ** i - 1, 2 ** (i + 1) - 1):
                gen_c[j][:, i] = 0
        return gen_c

    @staticmethod
    def generate_mixed_gen_c_0_lca_then_unif(n: int, c: int) -> List[torch.Tensor]:
        h = round(np.log2(c))
        gen_c = [torch.zeros((n, h)) for _ in range(2 * c - 1)]
        for i in range(c - 1, 2 * c - 1):
            gen_c[i] = to_binary(torch.ones((n,), dtype=torch.int) * i - (c - 1), h) * 2 - 1
        for i in range(c - 2, -1, -1):
            gen_c[i] = (gen_c[2 * i + 1] + gen_c[2 * i + 2]) / 2
        for i in range(h):
            for j in range(2 ** i - 1, 2 ** (i + 1) - 1):
                gen_c[j][:, i] = 0
                gen_c[j][:, i + 1:] = torch.rand((n, h - (i + 1))) * 2 - 1
        return gen_c

    @staticmethod
    def generate_mixed_gen_c_2_rand_paths_mean(n: int, c: int) -> List[torch.Tensor]:
        h = round(np.log2(c))
        gen_c = [torch.zeros((n, h)) for _ in range(2 * c - 1)]
        for i in range(c - 1, 2 * c - 1):
            gen_c[i] = to_binary(torch.ones((n,), dtype=torch.int) * i - (c - 1), h) * 2 - 1
        for i in range(h):
            for x, j in enumerate(range(2 ** i - 1, 2 ** (i + 1) - 1)):
                for k in range(n):
                    window = 2 ** (h - i)
                    l = torch.randint(low=x * window, high=(x + 1) * window - 1, size=()).item()
                    r = torch.randint(low=l, high=(x + 1) * window, size=()).item()
                    gen_c[j][k, :] = (gen_c[c - 1 + l][k, :] + gen_c[c - 1 + r][k, :]) / 2
        return gen_c

    @staticmethod
    def get_mix_generators() -> Dict[str, Callable[[int, int], List[torch.Tensor]]]:
        return {
            'l1_mean': CubeConditionEncoding.generate_mixed_gen_c_norm_mean(1),
            'l2_mean': CubeConditionEncoding.generate_mixed_gen_c_norm_mean(2),
            'linf_mean': CubeConditionEncoding.generate_mixed_gen_c_norm_mean(float('inf')),
            '0_lca_then_rand_path': CubeConditionEncoding.generate_mixed_gen_c_0_lca_then_rand_path,
            '0_lca_then_unif': CubeConditionEncoding.generate_mixed_gen_c_0_lca_then_unif,
            '2_rand_paths_mean': CubeConditionEncoding.generate_mixed_gen_c_2_rand_paths_mean,
        }

    @staticmethod
    def generate_mixed_gen_c_all(n: int, c: int) -> Dict[str, List[torch.Tensor]]:
        return {
            name: mix_generator(n, c)
            for name, mix_generator in CubeConditionEncoding.get_mix_generators().items()
        }

    @staticmethod
    def convert_clear_gen_c_to_cube_enc(gen_c: torch.Tensor) -> torch.Tensor:
        return gen_c

    @staticmethod
    def convert_cube_enc_to_clear_gen_c(cube_enc: torch.Tensor) -> torch.Tensor:
        return cube_enc


class PositiveCubeConditionEncoding(ConditionEncoding):
    @staticmethod
    def get_dim(c: int) -> int:
        return 2 * round(np.log2(c))

    @staticmethod
    def generate_clear_gen_c(n: int, c: int) -> torch.Tensor:
        h = round(np.log2(c))
        path = torch.randint(low=0, high=2, size=(n, h), dtype=torch.float)
        gen_c = torch.zeros((n, 2 * h))
        gen_c[:, torch.arange(0, 2 * h, 2)] = path
        gen_c[:, torch.arange(1, 2 * h, 2)] = 1 - path
        return gen_c

    @staticmethod
    def generate_mixed_gen_c_norm_mean(norm_ord: float) -> Callable[[int, int], List[torch.Tensor]]:
        def f(n: int, c: int) -> List[torch.Tensor]:
            h = round(np.log2(c))
            gen_c = [torch.zeros((n, 2 * h)) for _ in range(2 * c - 1)]
            for i in range(c - 1, 2 * c - 1):
                path = to_binary(torch.ones((n,), dtype=torch.int) * i - (c - 1), h)
                gen_c[i][:, torch.arange(0, 2 * h, 2)] = path
                gen_c[i][:, torch.arange(1, 2 * h, 2)] = 1 - path
            for i in range(c - 2, -1, -1):
                gen_c[i] = gen_c[2 * i + 1] + gen_c[2 * i + 2]
                # gen_c[i] /= torch.linalg.vector_norm(gen_c[i], dim=1, ord=norm_ord).reshape(-1, 1)
                gen_c[i] /= torch.from_numpy(np.linalg.norm(gen_c[i], axis=1, ord=norm_ord)).reshape(-1, 1)
                unit_vec = torch.zeros_like(gen_c[i])
                unit_vec[:, torch.arange(0, 2 * h, 2)] = 1
                # gen_c[i] *= torch.linalg.vector_norm(unit_vec, dim=1, ord=norm_ord).reshape(-1, 1)
                gen_c[i] *= torch.from_numpy(np.linalg.norm(unit_vec, axis=1, ord=norm_ord)).reshape(-1, 1)
            return gen_c
        return f

    @staticmethod
    def get_mix_generators() -> Dict[str, Callable[[int, int], List[torch.Tensor]]]:
        return {
            'l1_mean': PositiveCubeConditionEncoding.generate_mixed_gen_c_norm_mean(1),
            'l2_mean': PositiveCubeConditionEncoding.generate_mixed_gen_c_norm_mean(2),
            'linf_mean': PositiveCubeConditionEncoding.generate_mixed_gen_c_norm_mean(float('inf'))
        }

    @staticmethod
    def generate_mixed_gen_c_all(n: int, c: int) -> Dict[str, List[torch.Tensor]]:
        return {
            name: mix_generator(n, c)
            for name, mix_generator in PositiveCubeConditionEncoding.get_mix_generators().items()
        }

    @staticmethod
    def convert_clear_gen_c_to_cube_enc(gen_c: torch.Tensor) -> torch.Tensor:
        c = gen_c.shape[1]
        h = round(np.log2(c))
        path = gen_c[:, torch.arange(0, 2 * h, 2)]
        return path * 2 - 1

    @staticmethod
    def convert_cube_enc_to_clear_gen_c(cube_enc: torch.Tensor) -> torch.Tensor:
        binary = (cube_enc + 1) / 2
        n = cube_enc.shape[0]
        h = cube_enc.shape[1]
        gen_c = torch.zeros((n, 2 * h))
        gen_c[:, torch.arange(0, 2 * h, 2)] = binary
        gen_c[:, torch.arange(1, 2 * h, 2)] = 1 - binary
        return gen_c


def get_encoding(encoding_name) -> ConditionEncoding:
    if encoding_name == 'flat':
        return FlatConditionEncoding()
    elif encoding_name == 'cube':
        return CubeConditionEncoding()
    elif encoding_name == 'positive-cube':
        return PositiveCubeConditionEncoding()
    else:
        raise Exception(f'{encoding_name} encoding does not exist')


if __name__ == "__main__":
    print('flat')
    flat_clear_gen_c = FlatConditionEncoding.generate_clear_gen_c(8, 4)
    print(flat_clear_gen_c)
    print(FlatConditionEncoding.convert_clear_gen_c_to_cube_enc(flat_clear_gen_c))
    for name, gen_c in FlatConditionEncoding.generate_mixed_gen_c_all(1, 8).items():
        print('flat ->', name)
        print(gen_c)

    print('cube')
    cube_clear_gen_c = CubeConditionEncoding.generate_clear_gen_c(8, 4)
    print(cube_clear_gen_c)
    print(CubeConditionEncoding.convert_clear_gen_c_to_cube_enc(cube_clear_gen_c))
    for name, gen_c in CubeConditionEncoding.generate_mixed_gen_c_all(4, 8).items():
        print('cube ->', name)
        print(gen_c)

    print('positive-cube')
    positive_cube_clear_gen_c = PositiveCubeConditionEncoding.generate_clear_gen_c(8, 4)
    print(positive_cube_clear_gen_c)
    print(PositiveCubeConditionEncoding.convert_clear_gen_c_to_cube_enc(positive_cube_clear_gen_c))
    for name, gen_c in PositiveCubeConditionEncoding.generate_mixed_gen_c_all(1, 8).items():
        print('positive-cube ->', name)
        print(gen_c)
