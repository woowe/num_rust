use Matrix2d;
use ext::traits::ToMatrix2d;

use std::fmt;
use std::cmp::PartialEq;
use std::ops::{Neg, Sub};

impl PartialEq for Matrix2d {
    fn eq(&self, other: &Matrix2d) -> bool {
        self.n_cols == other.get_cols() &&
        self.n_rows == other.get_rows() &&
        &self.matrix == other.get_matrix()
    }
}

impl fmt::Debug for Matrix2d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output_string = format!("\nMatrix2d {{\n    n_rows: {},\n    n_cols: {},\n    rs: {}\n    cs: {}\n    matrix: ",
                                        self.n_rows,
                                        self.n_cols,
                                        self.rs,
                                        self.cs);

        let spacing = (0..self.get_cols())
            .map(|idx| {
                let col = self.get_col(idx).unwrap();

                col.iter().map(|x| x.to_string().len()).max().unwrap()
            })
            .collect::<Vec<usize>>();

        for idx in 0..self.n_rows {
            if idx > 0 {
                output_string = format!("{}            ", output_string);
            }

            // output_string = format!("{}{}", output_string, format!("{} [ ", idx));
            output_string.push('[');
            output_string.push(' ');
            let row = self.get_row(idx).unwrap();
            for (i, x) in row.iter().enumerate() {
                let x_str = x.to_string();
                let tmp = format!("{}{}",
                                  x_str,
                                  (0..(spacing[i] - x_str.len())).map(|_| ' ').collect::<String>());
                output_string = format!("{}{}", output_string, tmp);
                if i < self.get_cols() - 1 {
                    output_string = format!("{}, ", output_string);
                }
            }

            output_string.push(' ');
            output_string.push(']');
            if idx < self.get_rows() - 1 {
                output_string.push('\n');
            }
        }

        write!(f, "{},\n }}\n", output_string)
    }
}

impl Neg for Matrix2d {
    type Output = Matrix2d;

    fn neg(self) -> Matrix2d {
        self.scale(-1f64)
    }
}

impl Sub for Matrix2d {
    type Output = Option<Matrix2d>;

    fn sub(self, _rhs: Matrix2d) -> Option<Matrix2d> {
        self.subtract(&_rhs)
    }
}

impl ToMatrix2d for Vec<Vec<f64>> {
    fn to_matrix_2d(&self) -> Option<Matrix2d> {
        if self.len() > 0 {
            let col_len = self[0].len();
            for row in self.iter() {
                if col_len != row.len() {
                    return None;
                }
            }
            return Some(Matrix2d::from_vec(self));
        }
        None
    }

    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        self.to_matrix_2d().expect("Provided vec is of len <= 0").reshape(n_rows, n_cols)
    }
}

impl ToMatrix2d for Vec<f64> {
    fn to_matrix_2d(&self) -> Option<Matrix2d> {
        if self.len() > 0 {
            return Some(Matrix2d::from_vec(&self.iter()
                .map(|i| vec![*i])
                .collect::<Vec<Vec<f64>>>()));
        }
        None
    }

    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        Matrix2d::reshape_from_vec(&self, n_rows, n_cols)
    }
}

impl ToMatrix2d for [f64] {
    fn to_matrix_2d(&self) -> Option<Matrix2d> {
        if self.len() > 0 {
            return Some(Matrix2d::from_vec(&self.iter()
                .map(|i| vec![*i])
                .collect::<Vec<Vec<f64>>>()));
        }
        None
    }

    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        Matrix2d::reshape_from_vec(&self.to_vec(), n_rows, n_cols)
    }
}
