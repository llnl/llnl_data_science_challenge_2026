"""Registration-aware 3-D visualization of individual lattice struts."""

__all__ = ["extract_strut", "visualize_strut"]


def __getattr__(name):
    # Lazy imports keep ``python -m stage3_visualization.strut_visualizer``
    # free of runpy warnings and avoid importing plotting libraries for callers
    # that only need package metadata.
    if name in __all__:
        from .strut_visualizer import extract_strut, visualize_strut
        return {"extract_strut": extract_strut, "visualize_strut": visualize_strut}[name]
    raise AttributeError(name)
