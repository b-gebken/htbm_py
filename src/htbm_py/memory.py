class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.sample_pts = []
        self.oracle_vals = []

    def add(self, new_sample_pts, new_oracle_vals):
        self.sample_pts.extend(new_sample_pts)
        self.oracle_vals.extend(new_oracle_vals)

        new_length = len(self.sample_pts)
        if new_length > self.max_size:
            self.sample_pts = self.sample_pts[new_length-self.max_size:]
            self.oracle_vals = self.oracle_vals[new_length-self.max_size:]