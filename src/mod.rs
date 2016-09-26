pub mod ext;
pub mod utils;

use rand::distributions::{IndependentSample, Range};
use rand::{random, SeedableRng, StdRng};

use matrixmultiply;
use self::utils::vec_bin_op;
use self::ext::traits::ToMatrix2d;

#[derive(Clone)]
pub struct Matrix2d {
    n_rows: usize,
    n_cols: usize,
    rs: usize,
    cs: usize,
    matrix: Vec<f64>
}

impl Matrix2d {
    pub fn new(n_rows: usize, n_cols: usize) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            rs: n_cols,
            cs: 1,
            matrix: (0..n_rows*n_cols).map(|_| 0.0).collect::<Vec<f64>>()
        }
    }

    pub fn from_vec(vec: &Vec<Vec<f64>>) -> Matrix2d {
        Matrix2d {
            n_rows: vec.len(),
            n_cols: vec[0].len(),
            rs: vec[0].len(),
            cs: 1,
            matrix: vec.iter().flat_map(|el| el.iter().cloned() ).collect::<Vec<f64>>(),
        }
    }

    pub fn fill_rng(n_rows: usize, n_cols: usize) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            rs: n_cols,
            cs: 1,
            matrix: (0..n_rows*n_cols)
                .map(|_| random::<f64>()).collect::<Vec<f64>>()
        }
    }

    pub fn get_col(&self, n_col: usize) -> Option<Vec<f64>> {
        if n_col > self.n_cols - 1 {
            return None;
        }
        Some((0..self.n_rows)
            .map(|row| self.matrix[row * self.rs + n_col * self.cs])
            .collect::<Vec<f64>>())
    }

    pub fn get_row(&self, n_row: usize) -> Option<Vec<f64>> {
        if n_row > self.n_rows - 1 {
            return None;
        }
        // Some(&self.get_matrix()[n_row * self.rs .. n_row * self.rs + self.n_cols])
        Some((0..self.n_cols)
            .map(|col| self.matrix[n_row * self.rs + col * self.cs])
            .collect::<Vec<f64>>())
    }

    pub fn transpose(&self) -> Matrix2d {
        Matrix2d {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            rs: self.cs,
            cs: self.rs,
            matrix: self.matrix.clone()
        }
    }

    #[inline]
    pub fn get_cols(&self) -> usize {
        self.n_cols
    }

    #[inline]
    pub fn get_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn get_row_stride(&self) -> usize {
        self.rs
    }

    #[inline]
    pub fn get_col_stride(&self) -> usize {
        self.cs
    }

    #[inline]
    pub fn get_matrix(&self) -> &Vec<f64> {
        &self.matrix
    }

    pub fn get_matrix_mut(&mut self) -> &mut Vec<f64> {
        &mut self.matrix
    }

    pub fn dot(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if self.n_cols == m.get_rows() {
            let mut c = vec![0.; self.n_rows * m.get_cols()];
            // amazing magic happens here
            unsafe {
                matrixmultiply::dgemm(self.n_rows, self.n_cols, m.get_cols(),
                    1., self.get_matrix().as_ptr(), self.rs as isize, self.cs as isize,
                    m.get_matrix().as_ptr(), m.get_row_stride() as isize, m.get_col_stride() as isize,
                    0., c.as_mut_ptr(), m.get_cols() as isize, 1);
            }

            return Some(Matrix2d {
                n_rows: self.n_rows,
                n_cols: m.get_cols(),
                rs: m.get_cols(),
                cs: 1,
                matrix: c,
            });
        }
        None
    }

    pub fn apply_fn<F>(&self, f: F) -> Matrix2d
        where F: Fn(f64) -> f64
    {
        let len = self.matrix.len();
        let xs = &self.get_matrix()[..len];

        let mut out_vec = Vec::with_capacity(len);
        unsafe {
            out_vec.set_len(len);
        }

        {
            let out_slice = &mut out_vec[..len];

            for i in 0..len {
                out_slice[i] = f(xs[i]);
            }
        }

        Matrix2d {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            rs: self.rs,
            cs: self.cs,
            matrix: out_vec
        }
    }

    pub fn scale(&self, scalar: f64) -> Matrix2d {
        let len = self.matrix.len();
        let xs = &self.get_matrix()[..len];

        let mut out_vec = Vec::with_capacity(len);
        unsafe {
            out_vec.set_len(len);
        }

        {
            let out_slice = &mut out_vec[..len];

            for i in 0..len {
                out_slice[i] = xs[i] * scalar;
            }
        }

        Matrix2d {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            rs: self.rs,
            cs: self.cs,
            matrix: out_vec

        }
    }

    pub fn mult(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if  self.get_cols() == m.get_cols() &&
            self.get_rows() == m.get_rows() {
            return Some(
                Matrix2d {
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                    rs: self.rs,
                    cs: self.cs,
                    matrix: vec_bin_op(self.get_matrix(), m.get_matrix(), |x, y| x * y)
            });
        }
        None
    }

    pub fn subtract(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if  self.get_cols() == m.get_cols() &&
            self.get_rows() == m.get_rows() {
            return Some(
                Matrix2d {
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                    rs: self.rs,
                    cs: self.cs,
                    matrix: vec_bin_op(self.get_matrix(), m.get_matrix(), |x, y| x - y)
                });
        }
        None
    }

    pub fn addition(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if  self.get_cols() == m.get_cols() &&
            self.get_rows() == m.get_rows() {
            return Some(
                Matrix2d {
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                    rs: self.rs,
                    cs: self.cs,
                    matrix: vec_bin_op(self.get_matrix(), m.get_matrix(), |x, y| x + y)
                });
        }
        None
    }

    pub fn ravel(&self) -> Vec<f64> {
        self.matrix.clone()
    }

    pub fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        if self.matrix.len() / n_cols == n_rows {
            return Some(
                Matrix2d {
                    n_rows: n_rows,
                    n_cols: n_cols,
                    rs: n_cols,
                    cs: 1,
                    matrix: self.matrix.clone()
                }
            );
        }
        None
    }

    pub fn reshape_from_vec(vec: &Vec<f64>, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        if vec.len() / n_cols == n_rows {
            return Some(
                Matrix2d {
                    n_rows: n_rows,
                    n_cols: n_cols,
                    rs: n_cols,
                    cs: 1,
                    matrix: vec.clone()
                }
            );
        }
        None
    }

    pub fn normalize(&self) -> Matrix2d {
        let mut maxes = Vec::new();
        let mut matrix_clone = self.get_matrix().clone();
        for idx in 0..self.n_cols {
            maxes.push(self.get_col(idx).unwrap().iter()
                .fold(0f64, |acc, &x| {
                        if acc < x.abs() {
                            return x.abs();
                        }
                        return acc;
                    }));
        }

        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                matrix_clone[row * self.rs + col * self.cs] /= maxes[col];
            }
        }

        Matrix2d {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            rs: self.rs,
            cs: self.cs,
            matrix: matrix_clone.clone()
        }
    }

    pub fn shuffle<'a>(&self, seed: &'a [usize]) -> Matrix2d
     {
        let mut rng: StdRng = StdRng::from_seed(seed);
        let sample = Range::new(0, self.get_rows());

        let mut out_matrix = Matrix2d {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            rs: self.rs,
            cs: self.cs,
            matrix: vec![0.; self.n_rows * self.n_cols]
        };

        for row in 0..self.n_rows {
            let rnd_row = sample.ind_sample(&mut rng);
            for col in 0..self.n_cols {
                out_matrix.get_matrix_mut()[row * self.rs + col * self.cs] =
                    self.matrix[rnd_row * self.rs + col * self.cs];
            }
        }

        out_matrix
    }

    pub fn mini_batch(&self, batch_size: usize) -> Vec<Matrix2d> {
        let mut all_rows = Vec::new();
        for row in 0..self.n_rows {
            all_rows.push(self.get_row(row).unwrap());
        }

        let slice = &all_rows[..];
        let mut out_vec = Vec::new();
        for c in slice.chunks(batch_size) {
            out_vec.push(c.to_vec().to_matrix_2d().unwrap());
        }
        out_vec
    }
}
