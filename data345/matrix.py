import numpy as np
import sympy as sp

class matrix:
    def __init__(self,data):
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same length")
        
        self.data = [[float(x) for x in row] for row in data]
        self.shape = (len(data), len(data[0]))
    
    def __repr__(self):
        m, n = self.shape
        row_display = 6
        col_display = 6

        def _format_entry(val):
            if abs(val) < 1e-10:
                return "0.0"
            if isinstance(val, float):
                return f"{val:.6g}"  # up to 6 significant digits
            return str(val)

        # Format all entries in advance
        formatted_data = [[_format_entry(x) for x in row] for row in self.data]

        # If truncating columns, keep only first/last few
        if n > col_display:
            display_data = [row[:3] + ["..."] + row[-3:] for row in formatted_data]
        else:
            display_data = formatted_data

        # Compute column widths for alignment
        col_widths = [max(len(row[j]) for row in display_data if j < len(row))
                    for j in range(len(display_data[0]))]

        def format_row(row):
            return "[" + "  ".join(val.rjust(col_widths[j]) for j, val in enumerate(row)) + "]"

        # If truncating rows, keep only first/last few
        if m > row_display:
            shown = [format_row(r) for r in display_data[:3]] + ["..."]
            shown += [format_row(r) for r in display_data[-3:]]
        else:
            shown = [format_row(r) for r in display_data]

        body = "\n\t".join(shown)
        return f"matrix({body})"

    # def __repr__(self):
    #     m, n = self.shape
    #     # Set limits for displaying large matrices
    #     row_display = 6
    #     col_display = 6

    #     def _format_entry(val):
    #         if abs(val) < 1e-10:
    #             return "0.0"
    #         else:
    #             return f"{val:6g}"
            
    #     # Truncate rows to first and last 3 if more than 6 columns
    #     def format_row(row):
    #         if n > col_display:
    #             return "[" + "  ".join(map(_format_entry, row[:3])) + " ... " + "  ".join(map(str, row[-3:])) + "]"
    #         else:
    #             return "[" + "  ".join(map(_format_entry, row)) + "]"
        
    #     # Truncate columns to first and last 3 if more than 6 rows
    #     if m > row_display:
    #         shown = [format_row(row) for row in self.data[:3]] + ["..."]
    #         shown += [format_row(row) for row in self.data[-3:]]
    #     else:
    #         shown = [format_row(row) for row in self.data]

    #     body = "\n\t".join(shown)
    #     return f"matrix({body})"

    def __getitem__(self,key):
        if isinstance(key, tuple):
            i, j = key
            return self.data[i][j]
        return self.data[key]
    
    # def __setitem__(self, key, value):
    #     if not isinstance(value,(int, float)):
    #         raise ValueError(f"Cannot assign {type(value).__name__} to matrix entry")
        
    #     if isinstance(key, tuple) and len(key)==2:
    #         i, j = key
    #         self.data[i][j] = value
    #     else:
    #         self.data[key] = value
    
    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} vs {other.shape}")
        m, n = self.shape
        return matrix([[self[i,j] + other[i,j] for j in range(n)] for i in range(m)])
    
    def __mul__(self, other):
        # Insist on scalars being numeric
        if not isinstance(other,(int, float)):
            raise ValueError(f"Cannot multiply matrix by {type(other.__name__)}")
        return matrix([[other*x for x in row] for row in self.data])
    
    def __rmul__(self,other):
        return self*other
    
    def __sub__(self, other):
        return self + (-1*other)
    
    def row(self,key):
        return vector(self[key])
    
    def col(self,key):
        return vector([row[key] for row in self.data])
    
    def __matmul__(self,other):
        if not isinstance(other, matrix):
            return NotImplemented
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch for matrix multiplication, {self.shape} vs {other.shape}")
        m = self.shape[0]
        n = other.shape[1]
        result = [[self.row(i).dot(other.col(j)) for j in range(n)] for i in range(m)]
        return matrix(result)
    
    @property
    def T(self):
        m, n = self.shape
        return matrix([[self[i,j] for i in range(m)] for j in range(n)])
    
    def row_swap(self, i, j):
        """Swap row i and row j, in place."""
        self.data[i], self.data[j] = self.data[j], self.data[i]
    
    def row_scale(self, i, scalar):
        """Multiply row i by scalar, in place."""
        if not isinstance(scalar,(int,float)):
            raise TypeError(f"Can only scale row by number, got {type(scalar).__name__}")
        if scalar == 0:
            raise ValueError(f"Scaling by zero would annihilate row {i}")
        self.data[i] = [scalar*x for x in self.data[i]]
    
    def row_add(self, i, j, scalar=1):
        """Replace row i with row i + scalar*row j, in place."""
        self.data[i] = [a + scalar*b for a,b in zip(self.data[i],self.data[j])]
    
    # -------------------------------
    # Helper functions for solver backend
    # -------------------------------
    def _as_numpy(self):
        # Convert to NumPy array for numerical solving
        return np.array(self.data, dtype=float)
    
    def _as_sympy(self):
        # Convert to SymPy matrix for symbolic solving
        return sp.Matrix(self.data)
    
    def solve(self, b, mode="numeric"):
        if mode == "numeric":
            A = self._as_numpy()
            b = np.array(b, dtype=float)
            x = np.linalg.solve(A, b)
            return vector(x.tolist())
        elif mode == "symbolic":
            # Returns a SymPy matrix which has functionality for converting floats to rationals
            A = sp.nsimplify(self._as_sympy(), rational=True) 
            b = sp.nsimplify(sp.Matrix(b.data), rational=True)
            return A.LUsolve(b)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
    
    def rref(self):
        return sp.nsimplify(self._as_sympy().rref()[0], rational=True)
    
    # ---------------------------------------------
    # Simple Gaussian elimination tool
    # ---------------------------------------------
    # def gaussian_elimination(self, rref=False, augment=None):
    #     """Return the row echelon form of the matrix (does not modify original matrix)."""
        
    #     if augment is not None:
    #         if isinstance(augment, vector):
    #             b = matrix([[x] for x in augment.data])
    #         else:
    #             b = augment
    #         if self.shape[0] != b.shape[0]:
    #             raise ValueError(f"Row mismatch in augmentation, {self.shape[0]} vs {b.shape[0]}")
    #         A = matrix([row_A + row_b for row_A, row_b in zip(self.data, b.data)])
    #     else:
    #         A = matrix([row[:] for row in self.data])
        
    #     m,n = A.shape
    #     pivot_row = 0

    #     for pivot_col in range(n):
    #         if pivot_row >= m:
    #             break

    #         # Find row with nonzero pivot
    #         max_row = None
    #         for r in range(pivot_row, m):
    #             if A[r, pivot_col] != 0:
    #                 max_row = r
    #                 break
    #         if max_row is None:
    #             continue
            
    #         # Swap into pivot row
    #         if max_row != pivot_row:
    #             A.row_swap(pivot_row, max_row)
            
    #         pivot_val = A[pivot_row, pivot_col]
    #         A.row_scale(pivot_row, 1/pivot_val)

    #         # Eliminate below
    #         for r in range(pivot_row+1, m):
    #             if A[r, pivot_col] != 0:
    #                 factor = A[r, pivot_col]
    #                 A.row_add(r, pivot_row, -factor)
            
    #         pivot_row += 1
        
    #     if rref==True:
    #     # Convert to RREF by eliminating above pivot
    #         for pivot_row in reversed(range(m)):
    #             # Find pivot column
    #             pivot_col = None
    #             for j in range(n):
    #                 if A[pivot_row, j] == 1:
    #                     pivot_col = j
    #                     break
    #             if pivot_col is None:
    #                 continue
            
    #         # Once pivot is found, eliminate nonzero entries above
    #             for r in range(0, pivot_row):
    #                 if A[r, pivot_col] != 0:
    #                     factor = A[r, pivot_col]
    #                     A.row_add(r, pivot_row, -factor)
                
    #     return A
    def gaussian_elimination(self, rref=False, augment=None):
        """Return the (reduced) row echelon form of the matrix (does not modify original)."""

        if augment is not None:
            if isinstance(augment, vector):
                b = matrix([[x] for x in augment.data])
            else:
                b = augment
            if self.shape[0] != b.shape[0]:
                raise ValueError(f"Row mismatch in augmentation, {self.shape[0]} vs {b.shape[0]}")
            A = matrix([row_A + row_b for row_A, row_b in zip(self.data, b.data)])
        else:
            A = matrix([row[:] for row in self.data])

        m, n = A.shape
        pivot_row = 0
        pivotable_cols = self.shape[1]  # don’t pivot into augmentation

        for pivot_col in range(pivotable_cols):
            if pivot_row >= m:
                break

            # Partial pivoting
            max_row = max(range(pivot_row, m), key=lambda r: abs(A[r, pivot_col]))
            if abs(A[max_row, pivot_col]) < 1e-12:
                continue

            if max_row != pivot_row:
                A.row_swap(pivot_row, max_row)

            pivot_val = A[pivot_row, pivot_col]
            A.row_scale(pivot_row, 1/pivot_val)

            for r in range(pivot_row+1, m):
                if A[r, pivot_col] != 0:
                    factor = A[r, pivot_col]
                    A.row_add(r, pivot_row, -factor)

            pivot_row += 1

        if rref:
            # Backward elimination
            for prow in reversed(range(m)):
                pivot_col = None
                for j in range(n):
                    if abs(A[prow, j]) > 1e-12:
                        pivot_col = j
                        break
                if pivot_col is None:
                    continue

                # normalize pivot (in case it drifted numerically)
                A.row_scale(prow, 1 / A[prow, pivot_col])

                for r in range(prow):
                    if A[r, pivot_col] != 0:
                        factor = A[r, pivot_col]
                        A.row_add(r, prow, -factor)

        return A
