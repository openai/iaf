class HParams(object):

    def __init__(self, **kwargs):
        self._items = {}
        for k, v in kwargs.items():
            self._set(k, v)

    def _set(self, k, v):
        self._items[k] = v
        setattr(self, k, v)

    def parse(self, str_value):
        hps = HParams(**self._items)
        for entry in str_value.strip().split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, sep, value = entry.partition("=")
            if not sep:
                raise ValueError("Unable to parse: %s" % entry)
            default_value = hps._items[key]
            if isinstance(default_value, bool):
                hps._set(key, value.lower() == "true")
            elif isinstance(default_value, int):
                hps._set(key, int(value))
            elif isinstance(default_value, float):
                hps._set(key, float(value))
            else:
                hps._set(key, value)
        return hps
