use Matrix2d;

pub trait ToMatrix2d {
    fn to_matrix_2d(&self) -> Option<Matrix2d>;
    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d>;
}
