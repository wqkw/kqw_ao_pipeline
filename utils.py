

def get_prompt(filename):
    """Load file contents from prompts/ folder as a string"""
    with open(f"prompts/{filename}", "r") as f:
        return f.read()


def print_truncated(data):
    """Print nested lists/dicts with items truncated at 200 characters"""
    def truncate_item(item):
        if isinstance(item, str) and len(item) > 200:
            return item[:200] + "..."
        elif isinstance(item, dict):
            return {k: truncate_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [truncate_item(x) for x in item]
        else:
            return item

    truncated = truncate_item(data)
    if isinstance(data, list):
        for item in truncated:
            print(item)
    else:
        print(truncated)
