# Release contents

This repository contains the runnable Isaac Sim demonstration only:

- factory scene generation and visualization scripts;
- biped S62 URDF, required meshes, and imported USD asset;
- HIK-Q2-400D URDF and the two meshes used by the logistics preview;
- machine, bin, rack, and aluminum-tube STL assets;
- single-cell, all-cell, and complete AGV pipeline demonstrations.

The publish tree intentionally excludes logs, PID files, Python caches,
SolidWorks source assemblies/parts, ROS launch scaffolding, exporter logs,
temporary scenes, and unused robot meshes.

`scenes/humanoid_loading_factory.usd` is generated locally and ignored by Git
to keep the repository reproducible, compact, and centered on source assets.
