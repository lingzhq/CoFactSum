
def _is_json_serializable(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(_is_json_serializable(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and _is_json_serializable(v) for k, v in data.items())
    return False
