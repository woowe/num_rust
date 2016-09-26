extern crate num_rust;

use num_rust::ext::traits::ToMatrix2d;

#[test]
fn to_matrix_2d_vec() {
    let vec = vec![1f64, 2f64, 3f64].to_matrix_2d().expect("1d vec to matrix problem");
    let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().expect("2d vec to matrix problem");

    assert!(vec == m);
}

#[test]
fn to_matrix_2d_transpose() {
    let m   = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();
    let tm  = vec![vec![1f64, 2f64, 3f64]].to_matrix_2d().unwrap();

    assert!(m.transpose() == tm);
}

#[test]
fn get_matrix2d_col_size() {
    let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

    assert!(m.get_cols() == 1usize);
}

#[test]
fn get_matrix2d_row_size() {
    let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

    assert!(m.get_rows() == 3usize);
}

#[test]
fn get_matrix2d_col() {
    let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

    assert!(m.get_col(0).unwrap() == vec![1f64, 2f64, 3f64]);
}

#[test]
fn get_matrix2d_row() {
    let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

    assert!(m.get_row(0).unwrap() == &[1f64]);
}

#[test]
fn dot() {
    let m = vec![vec![5f64, 8f64, -4f64], vec![6f64, 9f64, -5f64], vec![4f64, 7f64, -2f64]].to_matrix_2d().unwrap();
    let m1 = vec![2f64, -3f64, 1f64].to_matrix_2d().unwrap();

    let pm = vec![-18f64, -20f64, -15f64].to_matrix_2d().unwrap();

    assert!(m.dot(&m1).unwrap() == pm);

    let m = vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0]].to_matrix_2d().unwrap(); // 3 x4
    let m1 = vec![vec![1.0, 2.0, 3.0]].to_matrix_2d().unwrap(); // 1 x 3

    let pm = vec![vec![0.0, 0.0, 0.0, 0.0]].to_matrix_2d().unwrap(); // 1 x 4
    assert!(m1.dot(&m).unwrap() == pm);
}

#[test]
fn dot_transpose() {
    let m = vec![vec![1., 2.], vec![3., 4.], vec![5., 6.]].to_matrix_2d().unwrap();
    let tm = m.transpose();
    let dtm = vec![vec![5., 11., 17.], vec![11., 25., 39.], vec![17., 39., 61.]].to_matrix_2d().unwrap();

    assert!(m.dot(&tm).unwrap() == dtm);
}

#[test]
fn apply_fn() {
    let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();
    let sm = vec![vec![1f64], vec![4f64], vec![9f64]].to_matrix_2d().unwrap();

    fn squared(x: f64) -> f64 {
        x * x
    }

    let c_squared = |x: f64| -> f64 { x * x };


    assert!(m.apply_fn(squared) == sm);
    assert!(m.apply_fn(c_squared) == sm);
}

#[test]
fn scale() {
    let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();
    let sm = vec![vec![2f64], vec![4f64], vec![6f64]].to_matrix_2d().unwrap();

    assert!(m.scale(2f64) == sm);
}

#[test]
fn subtract() {
    let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
    let m1 = vec![vec![0f64, -4f64, 3f64], vec![9f64, -4f64, -3f64]].to_matrix_2d().unwrap();
    let sm = vec![vec![-1f64, 6f64, -3f64], vec![-9f64, 7f64, 9f64]].to_matrix_2d().unwrap();


    assert!(m.subtract(&m1).unwrap() == sm);
}

#[test]
fn add() {
    let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
    let m1 = vec![vec![0f64, -4f64, 3f64], vec![9f64, -4f64, -3f64]].to_matrix_2d().unwrap();
    let sm = vec![vec![-1f64, -2f64, 3f64], vec![9f64, -1f64, 3f64]].to_matrix_2d().unwrap();

    assert!(m.addition(&m1).unwrap() == sm);
}

#[test]
fn mult() {
    let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
    let m1 = vec![vec![0f64, -4f64, 3f64], vec![9f64, -4f64, -3f64]].to_matrix_2d().unwrap();
    let sm = vec![vec![0f64, -8f64, 0f64], vec![0f64, -12f64, -18f64]].to_matrix_2d().unwrap();


    assert!(m.mult(&m1).unwrap() == sm);
}

#[test]
fn ravel() {
    let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
    let rvec = vec![-1f64, 2f64, 0f64, 0f64, 3f64, 6f64];

    assert!(m.ravel() == rvec);
}

#[test]
fn reshape_2d() {
    let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].reshape(3, 2).unwrap();
    let rm = vec![vec![-1f64, 2f64], vec![0f64, 0f64], vec![3f64, 6f64]].to_matrix_2d().unwrap();

    assert!(m == rm);
}

#[test]
fn reshape_1d() {
    let m = vec![-1f64, 2f64, 0f64, 0f64, 3f64, 6f64].reshape(3, 2).unwrap();
    let rm = vec![vec![-1f64, 2f64], vec![0f64, 0f64], vec![3f64, 6f64]].to_matrix_2d().unwrap();

    assert!(m == rm);
}
