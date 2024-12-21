class TwoTransforms:
    def __init__(self, transform, transform_st=None):
        self.transform = transform
        if transform_st is None:
            transform_st = transform
        self.transform_st = transform_st

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_st(sample)
        return x1, x2
