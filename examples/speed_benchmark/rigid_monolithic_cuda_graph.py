# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compare ordinary and CUDA-graph launches of the monolithic rigid constraint solve."""

import argparse
import time

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--num-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of timed simulation steps")
    parser.add_argument("--no-graph", action="store_true", help="Launch each rigid solver stage separately")
    args = parser.parse_args()

    # CUDA graph replay is CUDA-specific, so this benchmark intentionally has no CPU backend.
    gs.init(backend=gs.cuda, performance_mode=True)
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_monolithic_cuda_graph=not args.no_graph,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
    scene.build(n_envs=args.num_envs)

    for _ in range(10):
        scene.step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        scene.step()
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    positions = box.get_pos()
    if not torch.isfinite(positions).all():
        raise RuntimeError("Rigid positions became non-finite")
    print(f"{args.num_envs * args.steps / elapsed:,.1f} world-steps/s ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()
