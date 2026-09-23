def normalize(input=None, prompt=None):
    return (input or "").strip()


def clean_rows(rows):
    return [normalize(input=row, prompt=None) for row in rows]
