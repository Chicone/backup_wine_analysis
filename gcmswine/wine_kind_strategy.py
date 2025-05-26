# wine_kind_strategy.py

from collections import Counter
import re

class WineKindStrategy:
    def extract_labels(self, labels):
        return labels  # Default: no transformation

    def get_custom_order(self, labels, year_labels=None):
        return sorted(set(labels))

    def use_composite_labels(self, labels):
        return False

# Defienes the interface and default behaviour
class PressWineStrategy(WineKindStrategy):
    def extract_labels(self, labels):
        return [label[0] if label else None for label in labels]  # 'A1' -> 'A'

    def get_custom_order(self, labels, year_labels=None):
        if year_labels is not None and len(year_labels) > 0 and any(y is not None for y in year_labels):
            return list(Counter(year_labels).keys())
        return ["A", "B", "C"]

    def use_composite_labels(self, labels):
        return len(labels[0]) > 1 if labels and labels[0] else False

class BordeauxWineStrategy(WineKindStrategy):
    def __init__(self, class_by_year=False):
        self.class_by_year = class_by_year

    def extract_labels(self, labels):
        if self.class_by_year:
            return [re.search(r'\d{4}', label).group(0) if re.search(r'\d{4}', label) else None for label in labels]
        else:
            return [label[0] if label else None for label in labels]  # 'A2021B' -> 'A'B' -> 'A'

    def get_custom_order(self, labels, year_labels=None):
        return sorted(set(labels))

    def use_composite_labels(self, labels):
        return len(labels[0]) > 1 if len(labels) > 0 and labels[0] else False

class PinotNoirWineStrategy(WineKindStrategy):
    def __init__(self, region, get_custom_order_func):
        self.region = region
        self.get_custom_order_func = get_custom_order_func

    def extract_labels(self, labels):
        return labels  # No simplification for Pinot Noir

    def get_custom_order(self, labels, year_labels=None):
        return self.get_custom_order_func(self.region)

    def use_composite_labels(self, labels):
        return False

def get_strategy_by_wine_kind(wine_kind, region=None, get_custom_order_func=None, class_by_year=False):
    if wine_kind == "press":
        return PressWineStrategy()
    elif wine_kind == "bordeaux":
        return BordeauxWineStrategy(class_by_year=class_by_year)
    elif wine_kind == "pinot_noir":
        return PinotNoirWineStrategy(region, get_custom_order_func)
    else:
        return WineKindStrategy()
