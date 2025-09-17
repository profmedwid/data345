class vector:
    def __init__(self, data):
        try:
            # Convert input data to a float type for computation
            self.data = [float(x) for x in data]
        except (TypeError,ValueError) as e:
            raise type(e)(f"Input must be an iterable of numbers. Problem: {e}") from None

        # Insist that the list of elements is not empty
        if len(self.data) == 0:
            raise ValueError("Vector cannot be empty")

    def __repr__(self):
        return f"vector({self.data})"

    def __getitem__(self,key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        # Check that we are adding vectors to vectors
        if not isinstance(other, vector):
            raise TypeError(f"Can only add vector to vector, got {type(other).__name__}")

        # Check that the vectors are the same length
        if len(self.data) != len(other.data):
            raise ValueError(f"Vector dimensions must match: {len(self.data)} vs {len(other.data)}")

        return vector(data=[a + b for a,b in zip(self.data, other.data)])

    def __mul__(self, other):
        # Check that we are multiplying by a scalar.
        if not isinstance(other,(int, float)):
            raise TypeError(f"Cannot multiply vector by {type(other).__name__}")
        return vector(data=[a*other for a in self.data])

    def __rmul__(self, other):
        return self*other

    def __sub__(self, other):
        return self + (-1)*other

    def dot(self,other):
        # Check we are only using vectors
        if not isinstance(other, vector):
            raise TypeError(f"Expected two vectors, got {type(other.__name__)}")

        if len(self.data) != len(other.data):
            raise ValueError(f"Vector dimensions must match: {len(self.data)} vs {len(other.data)}")

        return sum([a*b for a,b in zip(self.data, other.data)])

    @property
    def norm(self):
        # Since v.v = ||v||^2, pass back (v.v)^0.5
        return self.dot(self)**0.5
