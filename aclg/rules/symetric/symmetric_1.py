from aclg.rules.split.split_basic import split_vertical, split_horizontal
from aclg.dataclass.component import Component

"""
symmetric_1 refer to cutting at the center of the component
"""

def split_symmetric_1_vertical(
        component: Component
) -> list[Component]:

    components = split_vertical(component, ratio=0.5)

    for comp in components:
        comp.generate_rule = "symmetric_1"
        comp.symmetry_axis = "vertical"

    return components


def split_symmetric_1_horizontal(
        component: Component
) -> list[Component]:

    components = split_horizontal(component, ratio=0.5)

    for comp in components:
        comp.generate_rule = "symmetric_1"
        comp.symmetry_axis = "horizontal"

    return components